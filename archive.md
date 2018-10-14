---
bg: "tag.jpg"
layout: page
permalink: /posts/
title: "Category"
crawlertitle: "076923 : Category"
summary: "Posts about Category"
active: Category
---

<center>
    {% include boxad-3.html %}
</center>

<br>
<h3>
<b><a href="https://076923.github.io/posts/#computer%20vision" class="fab fa-whmcs" style="text-decoration:none"> Theory </a></b>
<br>
<b><a href="https://076923.github.io/posts/#c#" class="fab fa-cuttlefish" style="text-decoration:none"> C# </a></b> - 
<a href="https://076923.github.io/posts/#c#-opencv" class="far fa-bookmark" style="text-decoration:none">OPENCV  </a>
<a href="https://076923.github.io/posts/#c#-tesseract" class="far fa-bookmark" style="text-decoration:none">TESSERACT  </a>
<a href="https://076923.github.io/posts/#c#-dynamixel" class="far fa-bookmark" style="text-decoration:none">DYNAMIXEL  </a>
<br>
<b><a href="https://076923.github.io/posts/#python" class="fab fa-python" style="text-decoration:none"> PYTHON </a></b> - 
<a href="https://076923.github.io/posts/#python-tkinter" class="far fa-bookmark" style="text-decoration:none">TKINTER  </a>
<a href="https://076923.github.io/posts/#python-numpy" class="far fa-bookmark" style="text-decoration:none">NUMPY  </a>
<a href="https://076923.github.io/posts/#python-opencv" class="far fa-bookmark" style="text-decoration:none">OPENCV  </a>
</h3>
<br>    



{% for tag in site.tags %}
  {% assign t = tag | first %}
  {% assign posts = tag | last %}

  {% for post in posts  limit: 1 %}
    {% if post.tags contains t %}
      {% if post.categories contains "posts" %}
      
<h2 class="category-key" id="{{ t | downcase }}">{{ t | capitalize }}</h2>

  {% endif %}
  {% endif %}
  {% endfor %}

  <ul class="year">
    {% for post in posts %}
      {% if post.tags contains t %}
        {% if post.categories contains "posts" %}
        <li>
          {% if post.lastmod %}
            <a href="{{ post.url }}">{{ post.title }}</a> - [<a href="{{ post.url }}#disqus_thread" data-disqus-identifier="{{ post.id }}">0 Comments</a>]
            <span class="date">{{ post.lastmod | date: "%Y-%m-%d"  }}</span>
          {% else %}
            <a href="{{ post.url }}">{{ post.title }} </a> - [<a href="{{ post.url }}#disqus_thread" data-disqus-identifier="{{ post.id }}">0 Comments</a>]
            <span class="date">{{ post.date | date: "%Y-%m-%d"  }}</span>
          {% endif %}
        </li>
      {% endif %}
      {% endif %}
    {% endfor %}
  </ul>
{% endfor %}

<center>
    <br>
    {% include boxad-4.html %}
</center>

