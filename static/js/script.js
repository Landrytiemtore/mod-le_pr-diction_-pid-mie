/* ═══════════════════════════════════════════════════
   SysSurv BF — script.js
   Animations, counters, région cards, scroll effects
═══════════════════════════════════════════════════ */

'use strict';

/* ── Données régions ── */
const REGIONS = [
  { name: 'Centre',          city: 'Ouagadougou',  status: 'critique', pop: '3.2M hab.' },
  { name: 'Hauts-Bassins',   city: 'Bobo-Dioulasso',status: 'alerte',   pop: '1.8M hab.' },
  { name: 'Sahel',           city: 'Dori',          status: 'stable',   pop: '1.1M hab.' },
  { name: 'Est',             city: 'Fada N\'Gourma', status: 'stable',  pop: '1.3M hab.' },
  { name: 'Boucle du Mouhoun',city: 'Dédougou',     status: 'stable',   pop: '1.4M hab.' },
  { name: 'Cascades',        city: 'Banfora',        status: 'alerte',   pop: '0.8M hab.' },
  { name: 'Centre-Nord',     city: 'Kaya',           status: 'stable',   pop: '1.5M hab.' },
];

const STATUS_LABELS = {
  critique: { label: 'Critique', dot: 'dot-red',   text: '🔴 Alerte critique' },
  alerte:   { label: 'Alerte',   dot: 'dot-amber', text: '🟡 Surveillance renforcée' },
  stable:   { label: 'Stable',   dot: 'dot-green', text: '🟢 Situation stable' },
};

/* ── Navbar scroll effect ── */
function initNavbar() {
  const navbar = document.getElementById('navbar');
  if (!navbar) return;
  const onScroll = () => {
    navbar.classList.toggle('scrolled', window.scrollY > 60);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

/* ── Intersection Observer pour les révélations ── */
function initReveal() {
  const targets = document.querySelectorAll('[data-reveal]');
  if (!targets.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
          // Décalage en cascade pour les éléments dans une grille
          const siblings = entry.target.parentElement
            ? [...entry.target.parentElement.children].filter(el => el.hasAttribute('data-reveal'))
            : [];
          const idx = siblings.indexOf(entry.target);
          const delay = Math.min(idx * 80, 400);

          setTimeout(() => {
            entry.target.classList.add('revealed');
          }, delay);

          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );

  targets.forEach(el => observer.observe(el));
}

/* ── Counter animation ── */
function animateCounter(el, target, duration = 1400) {
  const start = performance.now();
  const isLarge = target > 999;

  const easeOut = (t) => 1 - Math.pow(1 - t, 3);

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const current = Math.round(easeOut(progress) * target);

    el.textContent = isLarge ? current.toLocaleString('fr-FR') : current;

    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = isLarge ? target.toLocaleString('fr-FR') : target;
  }

  requestAnimationFrame(update);
}

function initCounters() {
  const counterEls = document.querySelectorAll('[data-count]');
  if (!counterEls.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const target = parseInt(entry.target.dataset.count, 10);
          animateCounter(entry.target, target);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  counterEls.forEach(el => observer.observe(el));
}

/* ── Génération des cartes régions ── */
function renderRegions() {
  const grid = document.getElementById('regionsGrid');
  if (!grid) return;

  REGIONS.forEach((region, i) => {
    const info = STATUS_LABELS[region.status];
    const card = document.createElement('div');
    card.className = 'region-card';
    card.setAttribute('data-reveal', '');
    card.style.transitionDelay = `${i * 60}ms`;

    card.innerHTML = `
      <div class="rc-flag">Région sanitaire</div>
      <div class="rc-name">${region.name}</div>
      <div class="rc-city">
        <span class="rc-dot ${info.dot}"></span>${region.city} · ${region.pop}
      </div>
      <div style="margin-top:12px;font-size:.75rem;font-weight:600;color:${
        region.status === 'critique' ? '#e24b4a' :
        region.status === 'alerte'   ? '#f59e0b' : '#6ee7b7'
      }">${info.text}</div>
    `;

    grid.appendChild(card);
  });

  // Initialise le reveal sur les cartes fraîchement insérées
  initReveal();
}

/* ── Smooth scroll pour les ancres ── */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (!target) return;
      e.preventDefault();
      const navH = document.getElementById('navbar')?.offsetHeight || 72;
      const top = target.getBoundingClientRect().top + window.scrollY - navH - 16;
      window.scrollTo({ top, behavior: 'smooth' });
    });
  });
}

/* ── Active nav link suivi de scroll ── */
function initActiveLinks() {
  const sections = ['fonctionnalites', 'regions', 'stats', 'technologie']
    .map(id => document.getElementById(id))
    .filter(Boolean);

  const links = document.querySelectorAll('.nav-links a');

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          links.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
          });
        }
      });
    },
    { threshold: 0.4 }
  );

  sections.forEach(section => observer.observe(section));
}

/* ── Dashboard URL dynamique (port configurable) ── */
function initDashboardLinks() {
  const DASHBOARD_URL = 'http://localhost:8501'; // ← modifier ici si besoin
  document.querySelectorAll('a[href="http://localhost:8501"]').forEach(el => {
    el.href = DASHBOARD_URL;
  });
}

/* ── Hero reveal au chargement ── */
function initHeroReveal() {
  document.querySelectorAll('.hero [data-reveal]').forEach((el, i) => {
    setTimeout(() => el.classList.add('revealed'), 200 + i * 120);
  });
}

/* ══════════════════════════════════════
   INIT
══════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initHeroReveal();
  renderRegions();      // Génère les cartes avant d'observer
  initReveal();         // Observe tous les [data-reveal] du DOM
  initCounters();
  initSmoothScroll();
  initActiveLinks();
  initDashboardLinks();
});
