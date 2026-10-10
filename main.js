const background = document.querySelector('.hero-background');
const toggle = document.querySelector('.background-toggle');
const videos = [...document.querySelectorAll('video:not(.hero-background)')];
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
  for (const other of videos) if (other !== video) other.pause();
});
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (!entry.isIntersecting) entry.target.pause();
  });
  for (const video of [...videos, background]) observer.observe(video);
}

// Closing a gallery also stops any media hidden inside it.
for (const gallery of document.querySelectorAll('.more-clips')) {
  gallery.addEventListener('toggle', () => {
    if (!gallery.open) for (const video of gallery.querySelectorAll('video')) video.pause();
  });
}

const comparison = document.querySelector('#comparison-video');
for (const button of document.querySelectorAll('[data-comparison]')) {
  button.addEventListener('click', () => {
    if (button.getAttribute('aria-pressed') === 'true') return;
    const terrain = button.dataset.comparison;
    const label = terrain === 'stairs' ? 'Stairs' : 'Box obstacles';
    comparison.pause();
    comparison.src = `pbfm_site_media/adaptation-${terrain}.mp4`;
    comparison.poster = `posters/adaptation-${terrain}.jpg`;
    comparison.setAttribute('aria-label', `Matched reference motion on the left and P-BFM execution on ${label.toLowerCase()} on the right`);
    comparison.load();
    document.querySelector('#comparison-terrain').textContent = label;
    for (const option of document.querySelectorAll('[data-comparison]')) {
      option.setAttribute('aria-pressed', String(option === button));
    }
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
