const background = document.querySelector('.hero-background');
const toggle = document.querySelector('.background-toggle');
const videos = [...document.querySelectorAll('video:not(.hero-background)')].filter(video => !video.closest('#adaptation'));
function updateToggle() {
  toggle.textContent = background.paused ? 'Play background' : 'Pause background';
  toggle.setAttribute('aria-pressed', String(!background.paused));
}
background.addEventListener('play', updateToggle);
background.addEventListener('pause', updateToggle);
toggle.addEventListener('click', () => {
  if (background.paused) background.play().catch(updateToggle);
  else background.pause();
});
// Keep the mobile hero still; playback remains available on request.
if (matchMedia('(prefers-reduced-motion: no-preference) and (min-width: 561px)').matches) background.play().catch(updateToggle);
for (const video of videos) video.addEventListener('play', () => {
  background.pause();
  for (const other of videos) {
    if (other !== video && !other.paused) other.pause();
  }
});
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (!entry.isIntersecting && !entry.target.paused) entry.target.pause();
  });
  for (const video of [...videos, background]) observer.observe(video);
}

// One clock and shared controls keep the reference and terrain clips comparable.
const terrainVideos = [...document.querySelectorAll('#adaptation video')];
const terrainControls = document.querySelector('.adaptation-controls');
const terrainToggle = terrainControls.querySelector('[data-play]');
const terrainStatus = terrainControls.querySelector('[role="status"]');
let terrainWanted = true;
let terrainStarting = false;
let terrainAttempt = 0;
let terrainBlocked = false;
terrainControls.hidden = false;
for (const video of terrainVideos) {
  video.controls = false;
  video.muted = true;
  video.preload = 'auto';
  video.addEventListener('waiting', pauseTerrain);
  video.addEventListener('ended', restartTerrain);
}
function pauseTerrain() {
  terrainAttempt++;
  terrainStarting = false;
  for (const video of terrainVideos) video.pause();
}
function restartTerrain() {
  pauseTerrain();
  for (const video of terrainVideos) if (video.readyState > 0) video.currentTime = 0;
  updateTerrain();
}
function updateTerrain() {
  const failed = terrainVideos.some(video => video.error);
  const ready = terrainVideos.filter(video => video.readyState >= 3 && !video.seeking).length;
  terrainToggle.textContent = failed ? 'Retry videos' : terrainWanted ? 'Pause all' : 'Play all';
  let status = terrainWanted ? 'Synchronized · muted' : 'Paused';
  if (ready < terrainVideos.length && terrainWanted) status = `Loading videos · ${ready} / ${terrainVideos.length} ready`;
  if (terrainBlocked) status = 'Tap Play all to start.';
  if (failed) status = 'A video could not load. Retry to play all six together.';
  if (terrainStatus.textContent !== status) terrainStatus.textContent = status;
  if (failed || !terrainWanted || document.hidden) {
    if (terrainVideos.some(video => !video.paused)) pauseTerrain();
    return;
  }
  if (terrainStarting) return;
  if (ready < terrainVideos.length) {
    if (terrainVideos.some(video => !video.paused)) pauseTerrain();
    return;
  }
  const time = terrainVideos[0].currentTime;
  for (const video of terrainVideos) {
    if (Math.abs(video.currentTime - time) > 0.12) video.currentTime = time;
  }
  if (terrainVideos.some(video => video.seeking) || terrainVideos.every(video => !video.paused)) return;
  terrainStarting = true;
  const attempt = ++terrainAttempt;
  Promise.all(terrainVideos.map(video => video.play())).catch(() => {
    if (attempt !== terrainAttempt) return;
    pauseTerrain();
    terrainWanted = false;
    terrainBlocked = true;
  }).finally(() => {
    if (attempt === terrainAttempt) terrainStarting = false;
  });
}
terrainToggle.addEventListener('click', () => {
  if (terrainVideos.some(video => video.error)) {
    for (const video of terrainVideos) if (video.error) video.load();
    terrainWanted = true;
  } else terrainWanted = !terrainWanted;
  terrainBlocked = false;
  updateTerrain();
});
terrainControls.querySelector('[data-restart]').addEventListener('click', restartTerrain);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) pauseTerrain();
  else updateTerrain();
});
setInterval(updateTerrain, 250);
updateTerrain();

// Closing a gallery also stops any media hidden inside it.
for (const gallery of document.querySelectorAll('.more-clips')) {
  gallery.addEventListener('toggle', () => {
    if (!gallery.open) for (const video of gallery.querySelectorAll('video')) if (!video.paused) video.pause();
  });
}

// Page UV keeps P-BFM separate from other projects on roboparty.github.io.
// Local previews do not load the counter; unavailable counts remain a dash.
if (location.hostname === 'roboparty.github.io' && location.pathname.startsWith('/P-BFM/')) {
  const counter = document.createElement('script');
  counter.src = 'https://busuanzi.9420.ltd/js';
  counter.async = true;
  counter.dataset.style = 'comma';
  document.body.append(counter);
}
