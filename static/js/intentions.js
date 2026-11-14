/* Active focus + prospective intentions console */

async function loadIntentions() {
  const sid = $('focus-sid')?.value?.trim() || 'default';
  const [focus, data] = await Promise.all([
    api('/active-memory?session_id=' + encodeURIComponent(sid)),
    api('/intentions?status=pending&limit=50'),
  ]);
  if (focus) {
    if ($('focus-goal')) $('focus-goal').value = focus.goal || '';
    if ($('focus-notes')) $('focus-notes').value = focus.notes || '';
  }
  const list = $('int-list');
  if (!list) return;
  const items = data?.intentions || [];
  if (!items.length) {
    list.innerHTML = '<div class="empty">No pending intentions</div>';
    return;
  }
  list.innerHTML = items.map(it => `
    <div class="mc" style="display:flex;gap:8px;align-items:flex-start;justify-content:space-between">
      <div>
        <div class="mt">${esc(it.text || '')}</div>
        <div class="mi">${it.cue ? 'cue: ' + esc(it.cue) + ' · ' : ''}id ${esc(it.id || '')}</div>
      </div>
      <div style="display:flex;gap:4px;flex-shrink:0">
        <button class="btn sm" onclick="setIntentionStatus('${esc(it.id)}','done')">Done</button>
        <button class="btn sm" onclick="setIntentionStatus('${esc(it.id)}','cancel')">Cancel</button>
      </div>
    </div>`).join('');
}

async function saveFocus() {
  const session_id = $('focus-sid')?.value?.trim() || 'default';
  const goal = $('focus-goal')?.value || '';
  const notes = $('focus-notes')?.value || '';
  const r = await post('/active-memory', { session_id, goal, notes });
  addLog('info', r ? 'Active focus saved' : 'Failed to save focus');
  if (r) loadIntentions();
}

async function addIntention() {
  const text = $('int-text')?.value?.trim();
  if (!text) return;
  const cue = $('int-cue')?.value?.trim() || '';
  const session_id = $('focus-sid')?.value?.trim() || '';
  const r = await post('/intentions', { text, cue, session_id });
  if (r?.ok) {
    if ($('int-text')) $('int-text').value = '';
    if ($('int-cue')) $('int-cue').value = '';
    addLog('info', 'Intention added');
    loadIntentions();
  } else {
    addLog('error', 'Failed to add intention');
  }
}

async function setIntentionStatus(id, status) {
  const r = await post('/intentions/' + encodeURIComponent(id) + '/status', { status });
  if (r?.ok) loadIntentions();
}

window.loadIntentions = loadIntentions;
window.saveFocus = saveFocus;
window.addIntention = addIntention;
window.setIntentionStatus = setIntentionStatus;
