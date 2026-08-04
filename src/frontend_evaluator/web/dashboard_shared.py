"""Shared dashboard fragments for agent detail rendering."""

SHARED_AGENT_DETAIL_CSS = """
.ag-card{margin:10px 0;border:1px solid var(--border);border-radius:12px;overflow:hidden;background:#fff}
.ag-hdr{padding:10px 12px;background:#f8fbf5;display:flex;align-items:center;justify-content:space-between;gap:10px;cursor:pointer}
.ag-hdr:hover{background:#eef6ea}
.ag-id{font-size:12px;font-weight:700;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ag-role{font-size:11px;color:var(--muted);margin-top:4px}
.ag-score{font-size:16px;font-weight:700;color:var(--accent)}
.ag-body{padding:12px}
.ag-end{font-size:12px;color:var(--muted);margin-bottom:8px;line-height:1.5}
.ag-expand-btn{font-size:11px;color:var(--accent);font-weight:700;display:flex;align-items:center;gap:6px}
.ag-expand-btn::before{content:'▶';font-size:9px;transition:transform .18s ease}
.ag-card.expanded .ag-expand-btn::before{transform:rotate(90deg)}
.traj{display:none;border-top:1px solid var(--border);background:#fafcf8}
.ag-card.expanded .traj{display:block}
.traj-empty{padding:12px;color:var(--muted);font-size:12px;text-align:center}
.traj-step{border-bottom:1px solid var(--border);padding:10px 12px}
.traj-step:last-child{border-bottom:none}
.ts-hdr{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.ts-idx{background:var(--accent);color:#fff;font-size:10px;font-weight:700;padding:2px 6px;border-radius:5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ts-tool{font-size:12px;font-weight:700;color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ts-success{width:16px;height:16px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700}
.ts-success.ok{background:#dff3e6;color:var(--ok)}
.ts-success.fail{background:#fde4df;color:var(--bad)}
.ts-time{font-size:10px;color:var(--muted);margin-left:auto}
.ts-task{font-size:10px;color:var(--muted);background:var(--surface2,#f8fafc);border:1px solid var(--border);border-radius:4px;padding:1px 5px;max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-args-toggle{color:var(--accent);font-size:11px;cursor:pointer;margin:4px 0;display:inline-block}
.ts-args-toggle:hover{text-decoration:underline}
.ts-args{background:#fff;border:1px solid var(--border);border-radius:6px;padding:6px 8px;margin:4px 0;font-size:11px;max-height:140px;overflow:auto}
.ts-args.collapsed{display:none}
.ts-result{font-size:11px;color:var(--muted);line-height:1.5;margin-top:4px;white-space:pre-wrap}
.ts-screenshot{margin-top:8px}
.ts-screenshot img{max-width:100%;border:1px solid var(--border);border-radius:6px;cursor:pointer}
.dim-item{background:#f8fbf5;border:1px solid var(--border);border-radius:8px;padding:9px 10px;margin-bottom:8px}
.dim-hdr{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:4px}
.dim-id{font-size:11px;font-weight:700;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--accent)}
.dim-score{font-size:12px;font-weight:700}
.dim-reason{font-size:12px;color:var(--muted);line-height:1.5}
.ev-item{background:#fff;border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:11px;color:var(--muted);margin-top:4px}
.agent-paths{margin-bottom:10px}
.path-panel{background:var(--surface2,#f8fafc);border:1px solid var(--border);border-radius:8px;padding:8px 10px}
.path-summary{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.path-chip{background:#ede9fe;color:#5b21b6;border-radius:999px;padding:2px 7px;font-size:10px;font-weight:600}
.path-groups{display:grid;gap:6px}
.path-group{background:#fff;border:1px solid var(--border);border-radius:6px;padding:8px 10px}
.path-group-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--light,var(--muted));margin-bottom:5px}
.path-row{display:grid;grid-template-columns:minmax(112px,152px) 1fr;gap:10px;align-items:start;padding:4px 0;border-top:1px solid var(--border)}
.path-row:first-child{border-top:none;padding-top:0}
.path-key{font-size:11px;font-weight:700;color:#4f46e5;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.path-value{font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all;line-height:1.5}
.path-value a{color:var(--accent);text-decoration:none}
.path-value a:hover{text-decoration:underline}
.sb{margin-top:8px;padding:8px 9px;background:#fff;border:1px solid var(--border);border-radius:8px}
.sb-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px}
.sb-chip{background:#eef6ea;color:var(--accent-2);border-radius:999px;padding:2px 7px;font-size:10px;font-weight:700}
.sb-sub-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:6px 0 4px}
.sb-row{font-size:11px;color:var(--muted);padding:3px 7px;background:#fffdfb;border:1px solid var(--border);border-radius:6px;margin-bottom:4px}
"""


SHARED_AGENT_DETAIL_JS = """
function scoreBreakdownHtml(scoreBreakdown){
  if(!scoreBreakdown||typeof scoreBreakdown!=='object')return '';
  const weights=scoreBreakdown.weights&&typeof scoreBreakdown.weights==='object'?scoreBreakdown.weights:{};
  const chips=[
    `method=${esc(scoreBreakdown.method)}`,
    `final=${fmt(scoreBreakdown.final_score)}`,
    `verdict=${fmt(scoreBreakdown.verdict_score)}`,
    `obj=${fmt(scoreBreakdown.objective_score)}(w=${fmt(weights.objective)})`,
    `subj=${fmt(scoreBreakdown.subjective_score)}(w=${fmt(weights.subjective)})`,
  ].map(item=>`<span class="sb-chip">${item}</span>`).join('');
  const objectiveDetails=Array.isArray(scoreBreakdown.objective_details)?scoreBreakdown.objective_details:[];
  const subjectiveDetails=Array.isArray(scoreBreakdown.subjective_details)?scoreBreakdown.subjective_details:[];
  const objectiveHtml=objectiveDetails.length?`<div class="sb-sub-title">Checks (${objectiveDetails.length})</div>${objectiveDetails.map((item,index)=>`<div class="sb-row">${index+1}. <b>${esc(item.check_id)}</b> ${esc(item.status)} w=${fmt(item.weight)} ${esc(item.reported)}</div>`).join('')}`:'';
  const subjectiveHtml=subjectiveDetails.length?`<div class="sb-sub-title">Subcriteria (${subjectiveDetails.length})</div>${subjectiveDetails.map((item,index)=>`<div class="sb-row">${index+1}. <b>${esc(item.subcriterion_id)}</b> rating=${esc(item.rating)} score=${fmt(item.score)} w=${fmt(item.weight)}</div>`).join('')}`:'';
  return `<div class="sb"><div class="sb-chips">${chips}</div>${objectiveHtml}${subjectiveHtml}</div>`;
}

function parsePathEvidenceLine(line){
  const text=String(line||'').trim();
  if(!text)return null;
  const separator=text.indexOf(': ');
  const rawKey=separator>=0?text.slice(0,separator):'detail';
  const rawValue=separator>=0?text.slice(separator+2):text;
  const lowerKey=rawKey.toLowerCase();
  const lowerValue=rawValue.toLowerCase();
  let group='Runtime';
  let source='runtime';
  let label='detail';
  if(lowerKey.startsWith('trajectory.steps[')){
    group='Trajectory';
    source='trajectory';
  }else if(lowerKey==='trajectory_ref'){
    group='Reference';
    source='reference';
  }else if(lowerKey==='evidence_ref'){
    group='Reference';
    source='handoff';
  }else if(lowerKey.includes('url')||lowerValue.startsWith('http://')||lowerValue.startsWith('https://')||lowerValue.startsWith('ws://')||lowerValue.startsWith('wss://')){
    group='URLs';
    source=lowerKey.startsWith('trajectory')?'trajectory':'runtime';
  }else if(lowerKey.includes('artifacts')||lowerKey.includes('process_log_dir')||lowerKey.includes('workspace_root')||lowerKey.includes('path')||lowerKey.includes('dir')){
    group='Runtime';
    source=lowerKey.startsWith('trajectory')?'trajectory':'runtime';
  }
  if(lowerKey==='workspace_root')label='workspace';
  else if(lowerKey==='artifacts_path')label='artifacts json';
  else if(lowerKey==='process_log_dir')label='process logs';
  else if(lowerKey==='app_url')label='app url';
  else if(lowerKey==='preview_url')label='preview url';
  else if(lowerKey==='cdp_url')label='cdp url';
  else if(lowerKey==='trajectory_ref')label='trajectory ref';
  else if(lowerKey==='evidence_ref')label='evidence ref';
  else if(lowerKey.startsWith('trajectory.steps[')){
    const stepMatch=rawKey.match(/^trajectory\.steps\[(\d+)\]\.(.+)$/);
    if(stepMatch){
      const stepIndex=Number(stepMatch[1])+1;
      const tail=stepMatch[2].replace(/^args\./,'').replace(/^observation\./,'').replace(/^result\./,'');
      const tailLabel=tail.split('.').pop().split('[')[0].replace(/_/g,' ');
      label=`step ${stepIndex} · ${tailLabel}`;
    }else{
      label='trajectory';
    }
  }else{
    label=rawKey.split('.').pop().split('[')[0].replace(/_/g,' ');
  }
  return {group,source,key:rawKey,label,value:rawValue};
}

function renderPathValue(value){
  const text=esc(String(value||''));
  if(/^https?:\/\//i.test(value)||/^wss?:\/\//i.test(value)){
    return `<a href="${esc(String(value))}" target="_blank" rel="noopener">${text}</a>`;
  }
  return text;
}

function renderPathEvidence(pathEvidence){
  const entries=(Array.isArray(pathEvidence)?pathEvidence:[])
    .map(parsePathEvidenceLine)
    .filter(Boolean);
  if(!entries.length)return '';
  const sourceCounts=new Map();
  for(const entry of entries){
    sourceCounts.set(entry.source,(sourceCounts.get(entry.source)||0)+1);
  }
  const summaryHtml=[`<span class="path-chip">${entries.length} entries</span>`,...Array.from(sourceCounts.entries()).map(([source,count])=>`<span class="path-chip">${esc(source)} ${count}</span>`)].join('');
  const groupOrder=['Runtime','URLs','Reference','Trajectory'];
  const groups=groupOrder.map(group=>({group,items:entries.filter(entry=>entry.group===group)})).filter(group=>group.items.length);
  const groupsHtml=groups.map(group=>`<div class="path-group"><div class="path-group-title">${esc(group.group)}</div>${group.items.map(item=>`<div class="path-row"><div class="path-key">${esc(item.label||item.key)}</div><div class="path-value">${renderPathValue(item.value)}</div></div>`).join('')}</div>`).join('');
  return `<div class="section agent-paths"><div class="label">Agent Paths</div><div class="path-panel"><div class="path-summary">${summaryHtml}</div><div class="path-groups">${groupsHtml}</div></div></div>`;
}

function getAgentTasks(agent){
  return Array.isArray(agent&&agent.tasks)?agent.tasks.filter(task=>task&&typeof task==='object'):[];
}

function getAgentTaskResults(agent){
  const taskResults=Array.isArray(agent&&agent.task_results)?agent.task_results.filter(task=>task&&typeof task==='object'):[];
  if(taskResults.length)return taskResults;
  return getAgentTasks(agent).map(task=>({
    task_id:task.task_id,
    title:task.title,
    passed:task.passed,
    verdict:task.verdict||task.status,
    reason:task.reason,
    steps:task.steps,
  }));
}

function getTaskTrajectorySteps(task){
  const trajectory=task&&typeof task.trajectory==='object'?task.trajectory:{};
  return Array.isArray(trajectory.steps)?trajectory.steps.filter(step=>step&&typeof step==='object'):[];
}

function getAgentStepCount(agent){
  const tasks=getAgentTasks(agent);
  if(tasks.length){
    return tasks.reduce((total,task)=>total+getTaskTrajectorySteps(task).length,0);
  }
  const trajectory=agent&&typeof agent.trajectory==='object'?agent.trajectory:null;
  const steps=trajectory&&Array.isArray(trajectory.steps)?trajectory.steps:[];
  return steps.length;
}

function renderSharedAgents(agents, taskIndex){
  if(!Array.isArray(agents)||!agents.length){return '<div class="section"><div class="label">Agents</div><div class="value">No agent report available.</div></div>';}
  return `<div class="section"><div class="label">Roles / Agents</div>${agents.map((agent,agentIndex)=>{
    const plannedTasks=Array.isArray(agent.planned_tasks)?agent.planned_tasks:[];
    const dimensions=Array.isArray(agent.dimensions)?agent.dimensions:[];
    const dimensionsHtml=dimensions.length?dimensions.map(dimension=>`
      <div class="dim-item">
        <div class="dim-hdr"><span class="dim-id">[${esc(dimension.dimension_id||'dimension')}]</span><span class="dim-score">${fmt(dimension.score)}${dimension.verdict?` · ${esc(dimension.verdict)}`:''}</span></div>
        <div class="dim-reason">${esc(dimension.reason||'-')}</div>
        ${Array.isArray(dimension.evidence)&&dimension.evidence.length?`<div>${dimension.evidence.map(evidence=>`<div class="ev-item">${typeof evidence==='string'?esc(evidence):esc(JSON.stringify(evidence))}</div>`).join('')}</div>`:''}
        ${scoreBreakdownHtml(dimension.score_breakdown)}
      </div>`).join(''):'<div class="value">No dimension results.</div>';
    const taskCount=getAgentTasks(agent).length;
    return `<div class="ag-card" id="ag-${taskIndex}-${agentIndex}">
      <div class="ag-hdr">
        <div>
          <div class="ag-id">${esc(agent.agent_id||agent.role||'unknown')}</div>
          <div class="ag-role">status ${esc(agent.status||'-')}${agent.end_reason?` · ${esc(agent.end_reason)}`:''}</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          ${statusBadge(agent.status||'unknown')}
          <span class="ag-score">${fmt(agent.score)}</span>
          <span class="ag-expand-btn">${taskCount?`${taskCount} task${taskCount!==1?'s':''}`:'No task'}</span>
        </div>
      </div>
      <div class="ag-body">
        ${dimensions.length?`<div class="section"><div class="label">Dimensions</div>${dimensionsHtml}</div>`:''}
      </div>
    </div>`;
  }).join('')}</div>`;
}
"""