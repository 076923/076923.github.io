---
layout: none
---

function stemWord(w) {
  return w
  .replace(/^[^\w가-힣]+/, '')
  .replace(/[^\w가-힣]+$/, '');
}

var koStemmer = function (token) {
  return token.update(function (word) {
    return stemWord(word);
  })
}

var idx = lunr(function () {
  this.field('title')
  this.field('excerpt')
  this.field('categories')
  this.field('tags')
  this.field('date')
  this.field('keywords')
  this.ref('id')

  this.pipeline.remove(lunr.trimmer)
  this.pipeline.add(koStemmer)
  this.pipeline.remove(lunr.stemmer)

  for (var item in store) {
    this.add({
      title: store[item].title,
      excerpt: store[item].excerpt,
      categories: store[item].categories,
      tags: store[item].tags,
      date: store[item].date,
      keywords: store[item].keywords,
      id: item
    })
  }
});

$(document).ready(function() {
  var $input = $('input#search');
  var $resultdiv = $('#results');
  var lastQuery = null;
  var timer = null;

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function runSearch() {
    var query = $input.val().trim().toLowerCase();
    if (query === lastQuery) { return; }
    lastQuery = query;

    if (query === '') {
      $resultdiv.empty();
      return;
    }

    var result = idx.query(function (q) {
      query.split(lunr.tokenizer.separator).forEach(function (term) {
        if (term === '') { return; }
        q.term(term, { boost: 100 });
        q.term(term, { usePipeline: false, wildcard: lunr.Query.wildcard.TRAILING, boost: 10 });
        q.term(term, { usePipeline: false, editDistance: 1, boost: 1 });
      });
    });

    $resultdiv.empty();
    $resultdiv.prepend('<p class="results__found">' + result.length + ' {{ site.data.ui-text[site.locale].results_found | default: "Result(s) found" }}</p>');

    for (var item in result) {
      var ref = result[item].ref;
      var doc = store[ref];
      var excerpt = escapeHtml(doc.excerpt.split(' ').splice(0, 80).join(' ')) + '...';
      var searchitem =
        '<div class="list__item">' +
          '<article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">' +
            '<h2 class="archive__item-title" itemprop="headline">' +
              '<a href="' + doc.url + '" rel="permalink">' + escapeHtml(doc.title) + '</a>' +
              '<span class="search__date">' + escapeHtml(doc.date) + '</span>' +
            '</h2>' +
            (doc.teaser ? '<div class="archive__item-teaser"><img src="' + doc.teaser + '" alt="" loading="lazy"></div>' : '') +
            '<p class="archive__item-excerpt" itemprop="description">' + excerpt + '</p>' +
          '</article>' +
        '</div>';
      $resultdiv.append(searchitem);
    }
  }

  // Enter 즉시 검색, 그 외 입력은 300ms 디바운스 후 자동 검색
  $input.on('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      clearTimeout(timer);
      runSearch();
    }
  });
  $input.on('input', function () {
    clearTimeout(timer);
    timer = setTimeout(runSearch, 300);
  });

  // URL 의 ?q= 로 들어온 경우 바로 검색
  var params = new URLSearchParams(window.location.search);
  if (params.get('q')) {
    $input.val(params.get('q'));
    runSearch();
  }
});
