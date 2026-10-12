/* ==========================================================================
   076923.github.io — 사이트 공용 스크립트 (2026-10 리디자인)
   1) 라이트/다크 전환 (+ utterances 댓글 테마 연동)
   2) "/" 키로 헤더 검색창 포커스
   3) 코드 블록 언어 라벨 + 복사 버튼
   4) 좌측 사이드바 스크롤 따라오기(긴 목록도 끝까지 보이게)
   5) 목차의 읽은 정도 표시
   ========================================================================== */
(function () {
  'use strict';

  var root = document.documentElement;

  /* 1) 테마 전환 ------------------------------------------------------------ */
  function currentTheme() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function syncUtterances(theme) {
    var frame = document.querySelector('iframe.utterances-frame');
    if (!frame || !frame.contentWindow) { return; }
    frame.contentWindow.postMessage(
      { type: 'set-theme', theme: theme === 'dark' ? 'github-dark' : 'github-light' },
      'https://utteranc.es'
    );
  }

  function updateToggle(theme) {
    var buttons = document.querySelectorAll('.theme-toggle');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute('aria-label', theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환');
      buttons[i].setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
    }
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch (e) {}
    updateToggle(theme);
    syncUtterances(theme);
  }

  /* 부드러운 전환: 토글 버튼에서 원이 퍼지며 새 테마가 드러남 (View Transitions API).
     미지원 브라우저는 색상 transition, '동작 줄이기' 설정 사용자는 즉시 전환 */
  function switchTheme(theme, origin) {
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { applyTheme(theme); return; }

    if (document.startViewTransition) {
      var x = window.innerWidth - 40, y = 40;
      if (origin && origin.getBoundingClientRect) {
        var r = origin.getBoundingClientRect();
        x = r.left + r.width / 2;
        y = r.top + r.height / 2;
      }
      // 버튼 중심에서 화면의 가장 먼 모서리까지의 거리 = 최종 반지름
      var radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));

      var transition = document.startViewTransition(function () { applyTheme(theme); });
      transition.ready.then(function () {
        root.animate(
          { clipPath: ['circle(0px at ' + x + 'px ' + y + 'px)', 'circle(' + radius + 'px at ' + x + 'px ' + y + 'px)'] },
          { duration: 550, easing: 'cubic-bezier(.4, 0, .2, 1)', pseudoElement: '::view-transition-new(root)' }
        );
      }).catch(function () {});
      return;
    }

    root.classList.add('theme-transition');
    applyTheme(theme);
    window.setTimeout(function () { root.classList.remove('theme-transition'); }, 400);
  }

  updateToggle(currentTheme());
  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('.theme-toggle') : null;
    if (!btn) { return; }
    switchTheme(currentTheme() === 'dark' ? 'light' : 'dark', btn);
  });

  /* 2) "/" 로 검색창 포커스 ---------------------------------------------------- */
  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) { return; }
    var tag = (document.activeElement && document.activeElement.tagName) || '';
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(tag) || (document.activeElement && document.activeElement.isContentEditable)) { return; }
    var input = document.getElementById('site-search-input') || document.getElementById('search');
    if (!input || input.offsetParent === null) { return; }
    e.preventDefault();
    input.focus();
  });

  /* 3) 코드 블록 헤더 ------------------------------------------------------- */
  var blocks = document.querySelectorAll('div.highlighter-rouge, figure.highlight');
  for (var b = 0; b < blocks.length; b++) {
    (function (block) {
      if (block.querySelector('.code-head')) { return; }
      var lang = 'code';
      var m = block.className.match(/language-([\w+#.-]+)/);
      var dataLang = block.querySelector('code[data-lang]');
      if (m) { lang = m[1]; } else if (dataLang) { lang = dataLang.getAttribute('data-lang'); }

      var head = document.createElement('div');
      head.className = 'code-head';
      var label = document.createElement('span');
      label.textContent = lang;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'code-copy';
      btn.innerHTML = '<i class="far fa-copy" aria-hidden="true"></i><span>복사</span>';
      btn.addEventListener('click', function () {
        var code = block.querySelector('.rouge-code pre') || block.querySelector('pre code') || block.querySelector('pre');
        var text = code ? code.innerText : '';
        var done = function () {
          btn.querySelector('span').textContent = '복사됨';
          setTimeout(function () { btn.querySelector('span').textContent = '복사'; }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {});
        }
      });
      head.appendChild(label);
      head.appendChild(btn);
      block.insertBefore(head, block.firstChild);
    })(blocks[b]);
  }

  /* 4) 좌측 사이드바: 목록 전체를 보이면서 스크롤을 따라오게 -----------------------
     화면에 다 들어가면 일반 sticky(헤더 아래 1.5rem). 화면보다 길면 스크롤 방향에 따라
     top 을 [화면높이 - 사이드바높이 - 간격, 헤더아래] 범위에서 움직여 양 끝이 모두 보이게 한다. */
  var sidebar = document.querySelector('.sidebar.sticky');
  if (sidebar) {
    var masthead = document.querySelector('.masthead');
    var topLimit = 0, bottomLimit = 0, curTop = null, lastY = window.pageYOffset;
    var measure = function () {
      var gap = parseFloat(getComputedStyle(document.documentElement).fontSize) * 1.5;
      topLimit = (masthead ? masthead.getBoundingClientRect().height : 0) + gap;
      bottomLimit = Math.min(topLimit, window.innerHeight - sidebar.offsetHeight - gap);
      if (curTop === null) { curTop = topLimit; }
      curTop = Math.max(bottomLimit, Math.min(topLimit, curTop));
      sidebar.style.top = curTop + 'px';
    };
    var follow = function () {
      var y = window.pageYOffset;
      var delta = y - lastY;
      lastY = y;
      if (bottomLimit >= topLimit) { return; }
      curTop = Math.max(bottomLimit, Math.min(topLimit, curTop - delta));
      sidebar.style.top = curTop + 'px';
    };
    measure();
    window.addEventListener('scroll', follow, { passive: true });
    window.addEventListener('resize', measure);
    window.addEventListener('load', measure);
  }

  /* 5) 읽은 정도 ------------------------------------------------------------ */
  var content = document.querySelector('.page__content');
  var fill = document.querySelector('.toc__progress-fill');
  var value = document.querySelector('.toc__progress-value');
  if (content && fill) {
    var ticking = false;
    var update = function () {
      ticking = false;
      var rect = content.getBoundingClientRect();
      var total = rect.height - window.innerHeight;
      var ratio = total > 0 ? Math.min(1, Math.max(0, -rect.top / total)) : 1;
      var pct = Math.round(ratio * 100);
      fill.style.width = pct + '%';
      if (value) { value.textContent = pct + '%'; }
    };
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }
})();
