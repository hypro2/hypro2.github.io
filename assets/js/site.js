// 테마 토글 / 코드 복사 / 목차 / 검색. 라이브러리 없음.
(function () {
  'use strict';

  // ── 테마 토글 ────────────────────────────────────────────────
  document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  });

  // ── 코드 복사 ────────────────────────────────────────────────
  document.querySelectorAll('div.highlight').forEach(function (block) {
    block.classList.add('code-wrap');
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy';
    btn.textContent = '복사';
    btn.addEventListener('click', function () {
      var src = block.querySelector('pre code') || block.querySelector('pre');
      if (!src) return;
      navigator.clipboard.writeText(src.innerText).then(function () {
        btn.textContent = '복사됨';
        setTimeout(function () { btn.textContent = '복사'; }, 1500);
      }, function () {
        btn.textContent = '실패';
        setTimeout(function () { btn.textContent = '복사'; }, 1500);
      });
    });
    block.appendChild(btn);
  });

  // ── 목차 ─────────────────────────────────────────────────────
  var toc = document.querySelector('[data-toc]');
  if (toc) {
    var heads = document.querySelectorAll('article h2, article h3');
    if (heads.length >= 2) {
      var nav = toc.querySelector('nav');
      var links = {};
      heads.forEach(function (h, i) {
        if (!h.id) h.id = 'h-' + i;
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        if (h.tagName === 'H3') a.className = 'toc-h3';
        nav.appendChild(a);
        links[h.id] = a;
      });
      toc.hidden = false;
      toc.open = window.innerWidth >= 1024;

      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          for (var id in links) links[id].classList.remove('is-active');
          links[e.target.id].classList.add('is-active');
        });
      }, { rootMargin: '0px 0px -75% 0px' });
      heads.forEach(function (h) { observer.observe(h); });
    }
  }

  // ── 검색 ─────────────────────────────────────────────────────
  var indexPromise = null;
  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch('/search.json').then(function (r) { return r.json(); }).catch(function () { return []; });
    }
    return indexPromise;
  }

  document.querySelectorAll('[data-search]').forEach(function (input) {
    var out = input.parentNode.querySelector('[data-search-results]');
    input.addEventListener('focus', loadIndex, { once: true });
    input.addEventListener('input', function () {
      var q = input.value.trim().toLowerCase();
      if (q.length < 2) { out.innerHTML = ''; return; }
      loadIndex().then(function (posts) {
        var hits = posts.filter(function (p) {
          return (p.title + ' ' + p.tags + ' ' + p.body).toLowerCase().indexOf(q) !== -1;
        }).slice(0, 8);
        // ponytail: 단순 부분일치. 글이 수백 개가 되면 그때 lunr 검토.
        if (!hits.length) { out.innerHTML = '<div class="search-empty">결과 없음</div>'; return; }
        out.innerHTML = hits.map(function (p) {
          return '<a href="' + p.url + '">' + esc(p.title) + '<small>' + p.date + '</small></a>';
        }).join('');
      });
    });
  });

  function esc(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
})();
