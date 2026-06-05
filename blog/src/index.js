/**
 * SONIC-O1 Blog: Main Entry Point
 */

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initScrollSpy();
  initSmoothScroll();
  initIframeResize();
});

/**
 * TOC scroll spy
 */
function initScrollSpy() {
  const tocLinks = document.querySelectorAll('d-contents a[href^="#"]');
  const sections = [];

  tocLinks.forEach(link => {
    const id = link.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if (el) sections.push({ link, el });
  });

  if (sections.length === 0) return;

  function onScroll() {
    const scrollY = window.scrollY + 140;
    let active = sections[0];

    for (let i = sections.length - 1; i >= 0; i--) {
      if (sections[i].el.offsetTop <= scrollY) {
        active = sections[i];
        break;
      }
    }

    tocLinks.forEach(l => {
      l.style.fontWeight = '';
      l.style.color = '';
    });
    if (active) {
      active.link.style.fontWeight = '700';
      active.link.style.color = '#EB088A';
    }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
      const targetId = link.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, '', '#' + targetId);
      }
    });
  });
}

/**
 * Auto-resize iframes
 */
function initIframeResize() {
  const resizeIframe = (iframe) => {
    try {
      const doc = iframe.contentDocument;
      const body = doc?.body;
      const html = doc?.documentElement;
      if (!body && !html) return;

      // Use the larger scroll height in case either body/html is taller.
      const height = Math.max(
        body?.scrollHeight ?? 0,
        html?.scrollHeight ?? 0
      );

      // Add a small buffer to account for borders/padding differences.
      iframe.style.height = (height + 20) + 'px';
    } catch (e) {
      // If cross-origin or not ready, just keep the initial height.
    }
  };

  const clearIframeBg = (iframe) => {
    try {
      const doc = iframe.contentDocument;
      if (!doc?.head) return;
      if (!doc.getElementById('__clr__')) {
        const s = doc.createElement('style');
        s.id = '__clr__';
        s.textContent = 'html,body{background:transparent!important}';
        doc.head.appendChild(s);
      }
    } catch (e) {}
  };

  // Capture-phase listener catches every iframe load, even ones added after DOMContentLoaded.
  document.addEventListener('load', (e) => {
    if (e.target.tagName === 'IFRAME') {
      clearIframeBg(e.target);
      resizeIframe(e.target);
      setTimeout(() => resizeIframe(e.target), 150);
      setTimeout(() => resizeIframe(e.target), 500);
      setTimeout(() => resizeIframe(e.target), 1000);
    }
  }, true);

  // Also handle any iframes already loaded when this runs.
  document.querySelectorAll('d-article iframe').forEach(iframe => {
    clearIframeBg(iframe);
    resizeIframe(iframe);
  });

  // Also handle viewport changes (responsive layout might affect diagram height).
  window.addEventListener('resize', () => {
    document.querySelectorAll('d-article iframe').forEach(resizeIframe);
  });
}