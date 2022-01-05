---
layout: post
title: "Python Django 강좌 : 제 1강 - 소개 및 설치"
tagline: "Python Django Introduction and Installation"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django MTV, Python Django Model, Python Django Template, Python Django View, Python Django REST Framework,  Python DRF, Representational State Transfer
ref: Python-Django
category: Python
permalink: /posts/Python-Django-1/
comments: true
toc: true
---

## Django란?

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

## Django 설치

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

<br>
<br>

## Django Rest Framework 설치

{% highlight django %}

pip install djangorestframework==3.11.0

{% endhighlight %}

`Django`에는 **REST(Representational State Transfer)** API를 위한 프레임워크가 존재합니다.

이를 `DRF(Django REST Framework)`라고 합니다.

`DRF`는 `pip`를 통하여 설치할 수 있습니다.

<br>
<br>

## Rest API란?

`REST`란 **자원(Resource)**을 정의하고 자원에 대한 **주소(URL)**를 지정하는 방법을 의미합니다.

다음 여섯 가지 조건을 `REST`라고 하며 해당 조건을 지켜 설계된 API를 **Restful API**라고 합니다.

* **Uniform Interface (일관적인 인터페이스)** : HTTP 표준만 따른다면 어떤 언어, 어떤 플랫폼에서도 사용이 가능한 인터페이스 스타일
* **Client-Server (클라이언트-서버)** : 서버는 API를 제공하고, 클라이언트는 사용자 인증에 관련된 일들을 직접 관리
* **Stateless (무상태)** : 요청 간 클라이언트의 컨텍스트(context)가 서버에 저장되지 않음
* **Cacheable (캐시 처리 가능)** : 클라이언트는 응답을 캐싱할 수 있어야 함
* **Layerd System (계층화)** : 클라이언트는 대상 서버에 직접 연결되었는지, 중간 서버를 통해 연결되었는지를 알 수 없음, 중간 서버는 로드 밸런싱 기능이나 공유 캐시 기능을 제공함으로써 시스템 규모 확장성을 향상시킬 수 있음
* **Self-descriptiveness (자체 표현 구조)** : Rest API 메시지만 보고도 쉽게 이해할 수 있는 자체 표현 구조로 설계해야 함

<br>

- `자원(Resource)` : URI
- `행위(Verb)` : HTTP Method (POST, GET, PUT, DELTE)
- `표현(Representations)` : JSON, XML, TEXT, RSS 등 여러 형태로 응답
- `서버(Server)` : 자원을 가지고 있는 쪽
- `클라이언트(Client)` : 자원을 요청하는 쪽

<br>
<br>

## Django CORS 설치

{% highlight django %}

pip install django-cors-headers==3.4.0

{% endhighlight %}

`CORS(Cross-Origin Resource Sharing)`란 교차 출처 리소스 공유라는 의미로 **실행 중인 웹 애플리케이션**이 다른 출처의 선택한 자원에 접근할 수 있는 권한을 부여하도록 브라우저에 알려주는 체제입니다.

즉, 외부에서 서버에 접속할 때 `CORS`가 허용되어있지 않다면 `CORS 오류`가 발생합니다.

이를 방지하기 위해 `Django`용 `CORS` 패키지를 설치합니다.
