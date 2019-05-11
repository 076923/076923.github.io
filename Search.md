---
bg: "search.jpg"
layout: page
title: "Search"
crawlertitle: "076923 : Search"
permalink: /search/
summary: "Search on the site."
active: Search
date: 11/20/2017 3:37:34 PM 
---

# 검색 # 
----------
<br>
<br>
<div id="search-container">
<center>
<input type="text" id="search-input" placeholder="Search..." style="width:90%;">
</center>
<ul id="results-container"></ul>
</div>

<!-- Script pointing to jekyll-search.js -->
<script src="{{site.baseurl}}/dest/jekyll-search.js" type="text/javascript"></script>

<script type="text/javascript">
      SimpleJekyllSearch({
        searchInput: document.getElementById('search-input'),
        resultsContainer: document.getElementById('results-container'),
        json: '{{ site.baseurl }}/search2.json',
        searchResultTemplate: '<li><a href="{url}" title="{desc}">{title}</a></li>',
        noResultsText: '검색결과가 없습니다.',
        limit: 10,
        fuzzy: false,
        exclude: ['Welcome']
      })
</script>

<br>
<br>

**사이트 내부에 있는 게시물을 검색할 수 있습니다.** **제목 및 내용에 대해 검색이 가능합니다.**

예) `houghcircle`, `random`, `dictionary`

<br>
<br>
<center>
    {% include boxad-4.html %}
</center>


