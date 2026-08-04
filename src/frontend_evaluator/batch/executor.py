"""Batch executor for running multiple evaluations."""

import time
import asyncio
from typing import List, Optional
from pathlib import Path

from .task_queue import TaskQueue
from ..events.models import EvaluationTask
from ..events.storage import EventStorage
from ..events.emitter import EventEmitter
from ..agent.runtime_factory import create_evaluator_runtime
from ..parser import ArtifactParser
from ..sandbox import get_sandbox_environment
from ..sandbox.agent_browser_executor import get_executor
from ..agent import FrontendEvaluator
from ..utils.config import Config
from ..utils.logger import logger


class BatchExecutor:
    """Executes multiple evaluations in batch with optional web visualization."""

    def __init__(
        self,
        max_parallel: int = 1,
        enable_web: bool = False,
        web_port: int = 8000,
    ):
        """Initialize batch executor.

        Args:
            max_parallel: Maximum parallel tasks
            enable_web: Whether to enable web dashboard
            web_port: Port for web server
        """
        self.max_parallel = max_parallel
        self.enable_web = enable_web
        self.web_port = web_port
        self.queue = TaskQueue(max_parallel)
        self.event_storage = EventStorage() if enable_web else None
        self.web_server = None
        self.web_app = None
        self.config = Config()

    async def execute(
        self,
        markdown_files: List[str],
        query: str,
    ) -> List[EvaluationTask]:
        """Execute batch evaluation.

        Args:
            markdown_files: List of markdown file paths
            query: User query for all evaluations

        Returns:
            List of completed tasks
        """
        logger.info(f"Starting batch evaluation: {len(markdown_files)} files")
        logger.info(f"Max parallel: {self.max_parallel}")
        logger.info(f"Web enabled: {self.enable_web}")

        # Add all tasks
        for md_file in markdown_files:
            self.queue.add_task(md_file, query)

        # Start web server if enabled
        if self.enable_web and self.event_storage:
            await self._start_web_server()

        # Execute all tasks
        start_time = time.time()
        tasks = await self.queue.run_all(self._execute_single_task, self.event_storage)
        duration = time.time() - start_time

        # Print summary
        self._print_summary(tasks, duration)

        # Stop web server if running
        if self.web_server:
            await self._stop_web_server()

        return tasks

    async def _execute_single_task(
        self,
        task: EvaluationTask,
        event_storage: Optional[EventStorage],
    ):
        """Execute a single evaluation task.

        Args:
            task: Task to execute
            event_storage: Optional event storage
        """
        logger.info(f"Starting task {task.id}: {task.markdown_file}")

        # Mark as running
        task.start()
        if event_storage:
            event_storage.save_task(task)
            # Broadcast task update if web enabled
            if self.web_app:
                from ..web.server import broadcast_task_update
                await broadcast_task_update(self.web_app, task)

        try:
            # Parse markdown
            parser = ArtifactParser()
            with open(task.markdown_file, "r") as f:
                markdown_content = f.read()
            files = parser.parse(markdown_content)

            if not files:
                raise ValueError("No files found in markdown")

            # Create event emitter
            event_emitter = None
            if event_storage:
                event_emitter = EventEmitter(task.id, enabled=True)
                event_emitter.on(lambda event: event_storage.save_event(event))

            # Sandbox creation is NOT supported in the public (local-only) release.
            # eval_open.py runs locally via CdpPlaywrightExecutor and does not use
            # this batch path. Remote sandbox providers (E2B/Docker/Remote/OpenSandbox)
            # were removed.
            raise NotImplementedError(
                "Remote sandbox providers are not included in the public release. "
                "Use eval_open.py with EXECUTOR_BACKEND=playwright (local)."
            )

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.fail(str(e))

        finally:
            # Save final task state
            if event_storage:
                event_storage.save_task(task)
                # Broadcast task update if web enabled
                if self.web_app:
                    from ..web.server import broadcast_task_update
                    await broadcast_task_update(self.web_app, task)

    async def _start_web_server(self):
        """Start web server for visualization."""
        try:
            import uvicorn
            from ..web.server import create_app

            logger.info(f"Starting web server on port {self.web_port}")

            # Create FastAPI app
            self.web_app = create_app(self.event_storage)

            # Create uvicorn config
            config = uvicorn.Config(
                self.web_app,
                host="0.0.0.0",
                port=self.web_port,
                log_level="warning",
            )

            # Create server and store reference for graceful shutdown
            self._uvicorn_server = uvicorn.Server(config)

            # Start server in background task
            self.web_server = asyncio.create_task(self._uvicorn_server.serve())

            # Give server time to start
            await asyncio.sleep(1)

            logger.info(f"✓ Web dashboard available at http://localhost:{self.web_port}")

        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            self.web_server = None
            self.web_app = None

    async def _stop_web_server(self):
        """Stop web server gracefully."""
        if not self.web_server:
            return
        logger.info("Stopping web server")
        try:
            # Signal uvicorn to shutdown gracefully
            if hasattr(self, "_uvicorn_server"):
                self._uvicorn_server.should_exit = True
                # Wait briefly for graceful shutdown
                try:
                    await asyncio.wait_for(self.web_server, timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self.web_server.cancel()
            else:
                self.web_server.cancel()
        except Exception as e:
            logger.debug(f"Web server stop: {e}")

    def _print_summary(self, tasks: List[EvaluationTask], duration: float):
        """Print evaluation summary.

        Args:
            tasks: List of tasks
            duration: Total duration in seconds
        """
        total = len(tasks)
        passed = len([t for t in tasks if t.verdict is True])
        failed = len([t for t in tasks if t.verdict is False])
        errors = len([t for t in tasks if t.status == "failed"])

        print("\n" + "=" * 60)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total tasks:     {total}")
        print(f"Passed:          {passed}")
        print(f"Failed:          {failed}")
        print(f"Errors:          {errors}")
        print(f"Duration:        {duration:.1f}s")
        print(f"Avg per task:    {duration/total:.1f}s")
        print("=" * 60)

        # Print individual results
        print("\nIndividual Results:")
        for i, task in enumerate(tasks, 1):
            status = "✓ PASSED" if task.verdict is True else "✗ FAILED"
            if task.status == "failed":
                status = "⚠ ERROR"

            filename = Path(task.markdown_file).name
            print(f"  [{i}/{total}] {filename}: {status}")

            if task.reason:
                print(f"        Reason: {task.reason}")
            if task.error:
                print(f"        Error: {task.error}")

        print()
