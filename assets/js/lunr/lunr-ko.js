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
  $('input#search').on('keyup', function () {
    var resultdiv = $('#results');
    var query = $(this).val().toLowerCase();

    if (window.event.keyCode == 13) {
      var result =
        idx.query(function (q) {
          query.split(lunr.tokenizer.separator).forEach(function (term) {
            q.term(term, { boost: 100 })
            if(query.lastIndexOf(" ") != query.length-1){
              q.term(term, {  usePipeline: false, wildcard: lunr.Query.wildcard.TRAILING, boost: 10 })
            }
            if (term != ""){
              q.term(term, {  usePipeline: false, editDistance: 1, boost: 1 })
            }
          })
        });

      resultdiv.empty();
      resultdiv.prepend('<p class="results__found">'+result.length+' {{ site.data.ui-text[site.locale].results_found | default: "Result(s) found" }}</p>');
      for (var item in result) {
        var ref = result[item].ref;
        if(store[ref].teaser){
          var searchitem =
            '<div class="list__item">'+
              '<article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">'+
                '<h2 class="archive__item-title" itemprop="headline">'+
                  '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
                '</h2>'+
                '<div class="archive__item-teaser">'+
                  '<img src="'+store[ref].teaser+'" alt="">'+
                '</div>'+
                '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt.split(" ").splice(0,80).join(" ")+'...</p>'+
              '</article>'+
            '</div>';
        }
        else{
          var searchitem =
            '<div class="list__item">'+
              '<article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">'+
                '<h2 class="archive__item-title" itemprop="headline">'+
                  '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
                  '<span class="search__date">'+ store[ref].date +'</span>' +
                '</h2>'+
                '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt.split(" ").splice(0,80).join(" ")+'...</p>'+
              '</article>'+
            '</div>';
        }
        resultdiv.append(searchitem);
      }
    }
  });
});