/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Replay: session replay with playback controls
   ═══════════════════════════════════════════════════════════════════════════ */

let RP = { steps: [], idx: 0, playing: false, speed: 1, timer: null };

async function loadReplay() {
  // Sessions are loaded by sessions.js into rp-sess
}

async function startReplay() {
  const sid = $('rp-sess')?.value;
  if (!sid) return;
  const data = await api('/session/' + encodeURIComponent(sid));
  const obs = data?.observations || data?.session?.observations || [];
  if (!obs.length) { $('rp-steps').innerHTML = '<div class="empty">No observations to replay</div>'; return }

  RP.steps = obs.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  RP.idx = 0; RP.playing = false;
  $('rp-total').textContent = RP.steps.length;
  renderReplayStep();
}

function renderReplayStep() {
  if (!RP.steps.length) return;
  const step = RP.steps[RP.idx];
  $('rp-current').textContent = RP.idx + 1;
  const pct = ((RP.idx + 1) / RP.steps.length * 100);
  $('rp-fill').style.width = pct + '%';

  $('rp-steps').innerHTML = `<div class="rp-step">
    <div style="font-size:9px;color:var(--text3);margin-bottom:4px">Step ${RP.idx + 1} / ${RP.steps.length} · ${fmtD(step.timestamp)}</div>
    <div style="font-size:12px;word-break:break-word">${esc((step.content || '').slice(0, 500))}</div>
    ${step.action ? `<div style="margin-top:4px"><span class="aeact ${step.action}">${esc(step.action)}</span></div>` : ''}
  </div>`;
}

function rpPlay() {
  if (RP.playing) { rpPause(); return }
  RP.playing = true;
  $('rp-play-btn').textContent = 'PAUSE';
  RP.timer = setInterval(() => {
    if (RP.idx < RP.steps.length - 1) { RP.idx++; renderReplayStep() }
    else { rpPause() }
  }, 1000 / RP.speed);
}

function rpPause() {
  RP.playing = false;
  $('rp-play-btn').textContent = 'PLAY';
  if (RP.timer) { clearInterval(RP.timer); RP.timer = null }
}

function rpPrev() { if (RP.idx > 0) { RP.idx--; renderReplayStep() } }
function rpNext() { if (RP.idx < RP.steps.length - 1) { RP.idx++; renderReplayStep() } }
function rpSpeed(s) { RP.speed = s; $('rp-speed-label').textContent = s + 'x'; if (RP.playing) { rpPause(); rpPlay() } }

window.loadReplay = loadReplay;
window.startReplay = startReplay;
window.rpPlay = rpPlay;
window.rpPause = rpPause;
window.rpPrev = rpPrev;
window.rpNext = rpNext;
window.rpSpeed = rpSpeed;
