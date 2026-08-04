"""Task queue for managing batch evaluations."""

import asyncio
import uuid
from typing import List, Set, Optional
from ..events.models import EvaluationTask
from ..events.storage import EventStorage
from ..utils.logger import logger


class TaskQueue:
    """Manages a queue of evaluation tasks with concurrency control."""

    def __init__(self, max_parallel: int = 1):
        """Initialize task queue.

        Args:
            max_parallel: Maximum number of tasks to run in parallel
        """
        self.max_parallel = max_parallel
        self.tasks: List[EvaluationTask] = []
        self.running: Set[str] = set()
        self.semaphore = asyncio.Semaphore(max_parallel)

    def add_task(self, markdown_file: str, query: str) -> str:
        """Add a task to the queue.

        Args:
            markdown_file: Path to markdown file
            query: User query

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = EvaluationTask.create(task_id, markdown_file, query)
        self.tasks.append(task)
        logger.info(f"Added task {task_id}: {markdown_file}")
        return task_id

    async def run_all(
        self,
        executor_factory,
        event_storage: Optional[EventStorage] = None,
    ) -> List[EvaluationTask]:
        """Run all tasks with concurrency control.

        Args:
            executor_factory: Async function that executes a single task
            event_storage: Optional storage for events

        Returns:
            List of completed tasks
        """
        # Save all tasks to storage
        if event_storage:
            for task in self.tasks:
                event_storage.save_task(task)

        # Run tasks with semaphore
        async def run_with_semaphore(task: EvaluationTask):
            async with self.semaphore:
                self.running.add(task.id)
                try:
                    await executor_factory(task, event_storage)
                finally:
                    self.running.remove(task.id)

        # Execute all tasks
        await asyncio.gather(*[run_with_semaphore(task) for task in self.tasks])

        return self.tasks

    def get_status(self) -> dict:
        """Get current queue status.

        Returns:
            Status dictionary
        """
        return {
            "total": len(self.tasks),
            "pending": len([t for t in self.tasks if t.status == "pending"]),
            "running": len(self.running),
            "completed": len([t for t in self.tasks if t.status == "completed"]),
            "failed": len([t for t in self.tasks if t.status == "failed"]),
        }
