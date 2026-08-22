let evidence = null;
let currentPhase = 0;
let playTimer = null;

const fmt = (n, digits=2) => Number(n).toLocaleString(undefined,{maximumFractionDigits:digits});
const pct = (n, digits=2) => `${(Number(n)*100).toFixed(digits)}%`;
const setText = (id, value) => { const el=document.getElementById(id); if(el) el.textContent=value; };

function showTab(id){
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.id===id));
  document.querySelectorAll('#nav button').forEach(x=>x.classList.toggle('active',x.dataset.tab===id));
  const active=document.querySelector(`#nav button[data-tab="${id}"]`);
  setText('page-title',active?active.textContent.replace(/^\d+\s*/, ''):id);
  window.scrollTo({top:0,behavior:'smooth'});
  if(id==='motion') loadPhase(currentPhase);
}

document.querySelectorAll('#nav button').forEach(button=>button.addEventListener('click',()=>showTab(button.dataset.tab)));

async function loadEvidence(){
  try{
    const [summary,contract]=await Promise.all([fetch('/api/summary').then(r=>r.json()),fetch('/api/output-contract').then(r=>r.json())]);
    evidence=summary;
    setText('release-version',`${summary.release.name} ${summary.release.version}`); setText('release-status',summary.release.status);
    setText('metric-source',summary.cohort.source_labels); setText('overview-eligible',summary.cohort.eligible); setText('overview-generated',`${summary.generation.accepted_trees}/${summary.generation.requested_trees}`); setText('overview-snapshots',summary.motion.snapshot_count); setText('overview-tests',summary.tests.passed);
    setText('f-source',summary.cohort.source_labels); setText('f-extracted',summary.cohort.extracted); setText('f-resolved',summary.cohort.resolved); setText('f-anatomy',summary.cohort.anatomy_pass); setText('f-eligible',summary.cohort.eligible); setText('cohort-proof',`${summary.cohort.source_labels} → ${summary.cohort.extracted} → ${summary.cohort.resolved} → ${summary.cohort.anatomy_pass} → ${summary.cohort.eligible}`);
    const tbody=document.querySelector('#exclusion-table tbody'); Object.entries(summary.cohort.exclusion_reasons).forEach(([key,value])=>{const row=document.createElement('tr');row.innerHTML=`<td>${key.replaceAll('_',' ')}</td><td>${value}</td>`;tbody.appendChild(row)});
    setText('coord-change',`${summary.source_integrity.coordinate_change_mm} mm`); setText('segment-change',`${summary.source_integrity.segment_length_change_mm} mm`); setText('residual-count',fmt(summary.source_integrity.pointwise_residual_records,0));
    setText('pca-cases',summary.pca.matrix_shape[0]); setText('pca-modes',summary.pca.modes); setText('pca-var',pct(summary.pca.variance,2)); setText('bspline-delta',`${Number(summary.bspline.maximum_numerical_difference_mm).toExponential(2)} mm`);
    setText('max-displacement',`${fmt(summary.demo.maximum_point_displacement_mm,4)} mm`); setText('healthy-pulse',pct(summary.demo.observed_global_radius_change_fraction,4));
    const pulses=summary.demo.observed_lesion_pulsatility; setText('focal-pulse',`${pct(pulses.focal_lad.radius_change_fraction,4)} · ${pct(pulses.focal_lad.lesion_to_healthy_ratio,2)} healthy`); setText('diffuse-pulse',`${pct(pulses.diffuse_lcx.radius_change_fraction,4)} · ${pct(pulses.diffuse_lcx.lesion_to_healthy_ratio,2)} healthy`); setText('tandem-pulse',`${pct(pulses.tandem_lad.radius_change_fraction,4)} · ${pct(pulses.tandem_lad.lesion_to_healthy_ratio,2)} healthy`);
    setText('vtk-status',summary.demo.vtk_readback_pass?'PASS':'FAIL'); setText('pvd-path',summary.demo_cases.focal_lad.pvd);
    setText('v-generated',`${summary.generation.accepted_trees}/${summary.generation.requested_trees}`); setText('v-attempts',summary.generation.sampling_attempts); setText('v-comparisons',`${summary.generation.population_comparisons_passed}/50`); setText('v-warnings',summary.generation.population_comparison_warnings); setText('v-tests',`${summary.tests.passed}/${summary.tests.passed+summary.tests.failed}`); setText('v-integrity',`${summary.integrity.unchanged}/${summary.integrity.total}`);
    setText('exact-dupes',summary.novelty.exact_duplicate_count); setText('near-dupes',summary.novelty.near_duplicate_under_0_1mm_count); setText('baseline-rms',`${fmt(summary.novelty.mean_rms_displacement_from_baseline_mm,3)} mm`); setText('holdout-accept',`${summary.holdout.generated_anatomy_acceptance_count}/${summary.holdout.total_holdout_predictions} (${pct(summary.holdout.generated_anatomy_acceptance_rate,2)})`); setText('holdout-leakage',`${summary.holdout.source_leakage_count} / ${summary.holdout.baseline_leakage_count}`);
    setText('qa-tests',summary.tests.passed); setText('qa-failed',summary.tests.failed); setText('qa-compile',summary.tests.compileall_pass?'PASS':'FAIL'); setText('qa-pip',summary.tests.pip_check_pass?'PASS':'FAIL'); setText('qa-vtk',summary.demo.vtk_readback_pass?'PASS':'FAIL'); setText('qa-hash',summary.integrity.status);
    setText('output-shapes',`geometry_cine.npy ${JSON.stringify(contract.geometry_cine_shape)}\ngeometry_static.npy ${JSON.stringify(contract.geometry_static_shape)}`);
    const files=document.getElementById('file-accordion'); Object.entries(contract.files).forEach(([name,description])=>{const d=document.createElement('details');d.innerHTML=`<summary><code>${name}</code></summary><p>${description}</p>`;files.appendChild(d)});
    document.getElementById('loading').classList.add('hidden');
    loadPhase(0);
  }catch(error){document.getElementById('loading').textContent=`Evidence load failed: ${error}`;}
}

async function loadPhase(index){
  currentPhase=Number(index); setText('phase-label',`Phase ${currentPhase} / 9`); document.getElementById('phase-slider').value=currentPhase;
  try{const data=await fetch(`/api/phase?case=focal_lad&index=${currentPhase}`).then(r=>r.json()); drawTree(data);}catch(error){console.error(error)}
}
function drawTree(data){
  const canvas=document.getElementById('tree-canvas'),ctx=canvas.getContext('2d'),colors={LMCA:'#a4adb7',LAD:'#f35b5b',LCX:'#24c6c9'};ctx.clearRect(0,0,canvas.width,canvas.height);
  const all=Object.values(data.branches).flat(); if(!all.length)return; const xs=all.map(p=>p[0]),zs=all.map(p=>p[2]),minX=Math.min(...xs),maxX=Math.max(...xs),minZ=Math.min(...zs),maxZ=Math.max(...zs),pad=55,sx=(canvas.width-2*pad)/(maxX-minX||1),sz=(canvas.height-2*pad)/(maxZ-minZ||1),scale=Math.min(sx,sz),tx=(canvas.width-(maxX-minX)*scale)/2,tz=(canvas.height-(maxZ-minZ)*scale)/2;
  ctx.lineCap='round';Object.entries(data.branches).forEach(([branch,pts])=>{ctx.strokeStyle=colors[branch];ctx.lineWidth=branch==='LMCA'?8:7;ctx.beginPath();pts.forEach((p,i)=>{const x=tx+(p[0]-minX)*scale,y=canvas.height-(tz+(p[2]-minZ)*scale);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();});ctx.fillStyle='#dbe8f3';ctx.font='16px Segoe UI';ctx.fillText(`Cardiac phase ${Number(data.phase).toFixed(3)} · fixed correspondence`,20,30);
}
function nextPhase(){loadPhase((currentPhase+1)%10)} function previousPhase(){loadPhase((currentPhase+9)%10)}
function togglePlay(){if(playTimer){clearInterval(playTimer);playTimer=null;setText('play-button','Play')}else{playTimer=setInterval(nextPhase,650);setText('play-button','Pause')}}

async function runVerification(mode){
  showTab('qa');const box=document.getElementById('verification-result');box.className='result-panel';box.textContent=mode==='full'?'Running 8-stage full verification. This can take about one minute…':'Running fast read-only verification…';
  try{const data=await fetch(`/api/verify?mode=${mode}`).then(r=>r.json());box.classList.add(data.status==='PASS'?'pass':'fail');box.textContent=`OVERALL VERIFICATION: ${data.status}\n\n`+data.checks.map(c=>`${c.status==='PASS'?'✓':'✗'} ${c.name}: ${c.detail}`).join('\n');}catch(error){box.classList.add('fail');box.textContent=`FAIL: ${error}`}
}

async function generateCase(){
  const button=document.getElementById('generate-button'),box=document.getElementById('generation-result');button.disabled=true;box.className='result-panel';box.textContent='Generating and validating a fresh case in the isolated runtime directory…';
  const payload={seed:Number(document.getElementById('gen-seed').value),pca_scale:Number(document.getElementById('gen-scale').value),disease_mode:document.getElementById('gen-mode').value,disease_branch:document.getElementById('gen-branch').value,position:Number(document.getElementById('gen-position').value),length:Number(document.getElementById('gen-length').value),severity:Number(document.getElementById('gen-severity').value),heart_rate:Number(document.getElementById('gen-heart').value)};
  try{const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();box.classList.add(data.status==='PASS'?'pass':'fail');box.innerHTML=`<b>${data.status}</b> · tensor ${JSON.stringify(data.shape)}<br>Output: <code>${data.output}</code><br>PVD: <code>${data.pvd}</code>${data.preview?`<br><img src="${data.preview}?t=${Date.now()}" style="max-width:680px;width:100%;margin-top:12px;border-radius:8px">`:''}`;}catch(error){box.classList.add('fail');box.textContent=`FAIL: ${error}`}finally{button.disabled=false}
}
loadEvidence();
