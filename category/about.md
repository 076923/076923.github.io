---
layout: post
permalink: /about/
sidebar: true
comments: true
title: "About"
classes: page--about
---
{%- comment -%}
  About 페이지 — 프로필 카드 + 블로그 통계. 값은 _config.yml(author) · navigation.yml(오픈채팅 링크) · 포스트 데이터에서 자동으로 채워진다.
{%- endcomment -%}
{%- assign nav_extras = site.data.navigation.docs | where: "title", "Extras" | first -%}
{%- assign kakao = nav_extras.children | where: "title", "Kakao Group Chat" | first -%}
{%- assign gh_user = site.repository | split: "/" | first -%}
{%- assign first_year = site.posts.last.date | date: "%Y" -%}
<section class="about-hero">
<div class="about-hero__body">
<p class="about-hero__eyebrow">{{ site.subtitle }}</p>
<h2 class="about-hero__name">윤대희 <span>{{ site.author.name }}</span></h2>
<p class="about-hero__lead">{{ first_year }}년부터 프로그래밍과 AI 기술을 정리하고 있습니다.</p>
<div class="about-hero__links">
<a class="about-btn" href="mailto:{{ site.author.email }}"><i class="fas fa-envelope" aria-hidden="true"></i>{{ site.author.email }}</a>
{%- if gh_user %}
<a class="about-btn" href="https://github.com/{{ gh_user }}" target="_blank" rel="noopener noreferrer"><i class="fab fa-github" aria-hidden="true"></i>GitHub</a>
{%- endif %}
{%- if kakao %}
<a class="about-btn" href="{{ kakao.url }}" target="_blank" rel="noopener noreferrer"><i class="fas fa-comment" aria-hidden="true"></i>카카오톡 오픈채팅</a>
{%- endif %}
<a class="about-btn" href="{{ '/guest/' | relative_url }}"><i class="fas fa-pen" aria-hidden="true"></i>방명록</a>
</div>
</div>
</section>
<ul class="about-stats">
<li><strong>{{ site.posts.size }}</strong><span>포스트</span></li>
<li><strong>{{ site.tags.size }}</strong><span>강좌 · 주제</span></li>
<li><strong>{{ site.categories.size }}</strong><span>카테고리</span></li>
<li><strong>{{ first_year }}</strong><span>블로그 시작</span></li>
</ul>
