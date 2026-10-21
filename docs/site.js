const slides = [...document.querySelectorAll('.slide')];
const track = document.querySelector('#track');
const viewport = document.querySelector('#viewport');
let current = 0;
let lastClick = -Infinity;
let suppressClickUntil = 0;

function go(index, updateHash = true) {
  current = Math.max(0, Math.min(slides.length - 1, index));
  track.style.transform = `translateX(-${current * 100}%)`;
  slides.forEach((slide, i) => {
    slide.inert = i !== current;
    slide.setAttribute('aria-hidden', String(i !== current));
  });
  if (updateHash) history.replaceState(null, '', `#${slides[current].id}`);
}

function fromHash() {
  const index = slides.findIndex(slide => `#${slide.id}` === location.hash);
  go(index < 0 ? 0 : index, false);
}

document.querySelector('.brand').addEventListener('click', event => {
  event.preventDefault();
  go(0);
});
document.querySelector('.return-home').addEventListener('click', () => go(0));
addEventListener('hashchange', fromHash);
fromHash();

viewport.addEventListener('click', event => {
  if (event.target.closest('a, button') || !getSelection().isCollapsed) return;
  const now = performance.now();
  if (now < suppressClickUntil || now - lastClick < 550) return;
  lastClick = now;
  go(current + 1);
});

addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  const keys = ['ArrowRight', 'PageDown', 'ArrowLeft', 'PageUp', 'Home', 'End'];
  if (keys.includes(event.key)) {
    event.preventDefault();
    go(event.key === 'Home' ? 0 : event.key === 'End' ? slides.length - 1 : current + (['ArrowRight', 'PageDown'].includes(event.key) ? 1 : -1));
  } else if (['Enter', ' '].includes(event.key) && event.target === viewport) {
    event.preventDefault();
    go(current + 1);
  }
});

let wheelTotal = 0, lastWheel = 0, lastTurn = -Infinity;
viewport.addEventListener('wheel', event => {
  if (event.ctrlKey) return;
  event.preventDefault();
  const now = performance.now(), quiet = now - lastWheel > 180;
  lastWheel = now;
  if (now - lastTurn < 650 || (!quiet && Number.isFinite(lastTurn) && wheelTotal === 0)) return;
  if (quiet) wheelTotal = 0;
  const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
  wheelTotal += delta * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
  if (Math.abs(wheelTotal) > 45) {
    go(current + Math.sign(wheelTotal));
    wheelTotal = 0;
    lastTurn = now;
  }
}, {passive: false});

let touchStart = null;
viewport.addEventListener('touchstart', event => {
  touchStart = event.touches.length === 1 ? {x: event.touches[0].clientX, y: event.touches[0].clientY} : null;
}, {passive: true});
viewport.addEventListener('touchend', event => {
  if (!touchStart || !event.changedTouches.length) return;
  const dx = event.changedTouches[0].clientX - touchStart.x;
  const dy = event.changedTouches[0].clientY - touchStart.y;
  touchStart = null;
  if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy) * 1.3) {
    suppressClickUntil = performance.now() + 400;
    go(current + (dx < 0 ? 1 : -1));
  }
}, {passive: true});
viewport.addEventListener('touchcancel', () => { touchStart = null; });
