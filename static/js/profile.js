/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Profile: user persona view and edit
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadProfile() {
  const [profile, config] = await Promise.all([api('/profile'), api('/config')]);

  if (profile) {
    const fields = profile.profile || profile;
    const keys = Object.keys(fields);
    if (!keys.length) { $('profile-fields').innerHTML = '<div class="empty">No profile data</div>'; return }
    $('profile-fields').innerHTML = keys.map(k =>
      `<div class="pf"><span class="pfk">${esc(k)}</span><span class="pfv">${esc(typeof fields[k] === 'string' ? fields[k] : JSON.stringify(fields[k]))}</span></div>`
    ).join('');
  } else { $('profile-fields').innerHTML = '<div class="empty">No profile data</div>' }

  // Config info
  if (config) {
    const em = config.embedding_model || {};
    const st = config.settings || {};
    const redisOk = config.redis_connected;
    let cfg = '<div style="font-size:11px">';
    cfg += `<div class="dr"><span class="dk">Version</span><span class="dv">${esc(config.version || '—')}</span></div>`;
    cfg += `<div class="dr"><span class="dk">Redis</span><span class="dv" style="color:${redisOk ? 'var(--green)' : 'var(--red, #f55)'}">${redisOk ? 'Connected' : 'Disconnected'}</span></div>`;
    cfg += `<div class="dr"><span class="dk">Embedding</span><span class="dv">${esc(em.provider || '—')} / ${esc(em.model || '—')} (${em.dims || '?'}d)</span></div>`;
    cfg += `<div class="dr"><span class="dk">Consolidate every</span><span class="dv">${config.auto_consolidate_every || 50} stores</span></div>`;
    cfg += `<div class="dr"><span class="dk">Dedup threshold</span><span class="dv">${st.dedup_similarity_threshold ?? '—'}</span></div>`;
    cfg += `<div class="dr"><span class="dk">Token budget</span><span class="dv">${st.default_token_budget ?? '—'}</span></div>`;
    cfg += '</div>';
    $('profile-config').innerHTML = cfg;
  }
}

window.loadProfile = loadProfile;
