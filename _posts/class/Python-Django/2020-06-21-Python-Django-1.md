---
layout: post
title: "Python Django 강좌 : 제 1강 - Django 소개 및 설치"
tagline: "Python Django Introduction and Installation"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Django']
keywords: Python, Python Django, Python Django 3, Python Django MTV, Python Django Model, Python Django Template, Python Django View
ref: Python
category: posts
permalink: /posts/Python-Django-1/
comments: true
---

## Django란? ##
----------

`장고(Django)`는 Python 기반의 오픈 소스 웹 애플리케이션 프레임워크(Web Application Framework)입니다.

장고를 활용해 **UI**, **UX**와 관련된 `프론트엔트(Front-end)` 개발과 **Server**, **DB** 등과 관련된 `백엔드(Back-end)` 개발을 진행할 수 있습니다.

장고의 기본 구성은 `모델(Model)`, `템플릿(Template)`, `뷰(View)`로 **MTV 패턴**을 따릅니다.

`모델`은 데이터에 관한 정보**(저장, 접근, 검증, 작동 등)**를 처리하고 `논리 및 규칙`을 직접 관리합니다.

`템플릿`은 **데이터가 어떻게 표시**되는 지를 정의합니다. 템플릿은 사용자에게 실제로 보여지는 웹 페이지나 문서를 작성합니다.

`뷰`는 **어떤 데이터**를 표시할지 정의하며, `HTTP 응답 상태 코드(response)`를 반환합니다. 또한, **웹 페이지**, **리디렉션**, **문서** 등의 형태가 가능합니다.

즉, 데이터베이스에서 관리할 데이터를 정의(Model)하며, 사용자가 보는 화면(Template)과 애플리케이션의 처리 논리(View)를 정의합니다.

장고는 **Instagram, Disqus, Mozilla, Pinterest, Bitbucket** 등에서도 사용하고 있습니다.

<br>
<br>

## Django 설치 ##
----------

{% highlight django %}

pip install django==3.0.7

{% endhighlight %}

`Django` 프레임워크는 `pip`를 통하여 설치할 수 있습니다.

본 강좌는 **Django 3.0.7**을 기반으로 작성돼 있습니다.

Django가 정상적으로 설치 되었다면, 아래의 구문으로 Django 설치 버전을 확인할 수 있습니다.

<br>

{% highlight django %}

python -m django --version

{% endhighlight %}

**결과**
:    
3.0.7<br>
<br>

Django 3.0 이상의 버전은 **Python 3.6**, **Python 3.7**, **Python 3.8**만 공식적으로 지원합니다.

만약, `Python 3.5` 등의 낮은 버전을 사용한다면 Django 2.2나 Django 2.1 등의 낮은 버전을 사용합니다.
