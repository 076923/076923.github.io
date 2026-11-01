---
layout: post
title: "Python Django 강좌 : 제 1강 - 소개 및 설치"
tagline: "Python Django Introduction and Installation"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django MTV, Python Django Model, Python Django Template, Python Django View, Python Django REST Framework,  Python DRF, Representational State Transfer, Python Django CORS
ref: Python-Django
category: Python
permalink: /posts/Python-Django-1/
comments: true
toc: true
---

## Django란?

`장고(Django)`는 Python 기반의 오픈 소스 웹 애플리케이션 프레임워크(Web Application Framework)입니다.

장고를 활용해 **UI**, **UX**와 관련된 `프론트엔드(Front-end)` 개발과 **Server**, **DB** 등과 관련된 `백엔드(Back-end)` 개발을 진행할 수 있습니다.

장고의 기본 구성은 `모델(Model)`, `템플릿(Template)`, `뷰(View)`로 **MTV 패턴**을 따릅니다.

`모델`은 데이터에 관한 정보**(저장, 접근, 검증, 작동 등)**를 처리하고 `논리 및 규칙`을 직접 관리합니다.

`템플릿`은 **데이터가 어떻게 표시**되는 지를 정의합니다. 템플릿은 사용자에게 실제로 보여지는 웹 페이지나 문서를 작성합니다.

`뷰`는 **어떤 데이터**를 표시할지 정의하며, 요청을 처리한 뒤 `HTTP 응답(HttpResponse)`을 반환합니다. 응답은 **웹 페이지**, **리디렉션**, **JSON 문서**, **404 오류** 등의 형태가 가능합니다.

즉, 데이터베이스에서 관리할 데이터를 정의(Model)하며, 사용자가 보는 화면(Template)과 애플리케이션의 처리 논리(View)를 정의합니다.

- Tip : 일반적인 `MVC 패턴`과 비교하면 장고의 `뷰`는 MVC의 **컨트롤러(Controller)**, 장고의 `템플릿`은 MVC의 **뷰(View)**에 해당합니다.

장고는 **Instagram, Disqus, Mozilla, Pinterest, Bitbucket** 등에서도 사용하고 있습니다.

<br>
<br>

## 가상 환경 설정

장고는 프로젝트마다 사용하는 버전이 달라지는 경우가 많으므로, `가상 환경(Virtual Environment)`을 만들어 설치하는 것이 좋습니다.

Python에 기본으로 포함된 `venv` 모듈로 가상 환경을 생성할 수 있습니다.

<br>

{% highlight bash %}

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux, macOS
source .venv/bin/activate

{% endhighlight %}

`python -m venv {경로}`로 가상 환경을 생성한 다음, 운영체제에 맞는 명령어로 가상 환경을 활성화합니다.

이후 설치하는 패키지는 모두 활성화된 가상 환경 안에 설치됩니다.

<br>
<br>

## Django 설치

{% highlight bash %}

pip install django==6.1.2

{% endhighlight %}

`Django` 프레임워크는 `pip`를 통하여 설치할 수 있습니다.

본 강좌는 **Django 6.1.2**를 기반으로 작성돼 있습니다.

Django가 정상적으로 설치 되었다면, 아래의 구문으로 Django 설치 버전을 확인할 수 있습니다.

<br>

{% highlight bash %}

python -m django --version

{% endhighlight %}

**결과**
:    
6.1.2<br>
<br>

Django 6.0 이상의 버전은 **Python 3.12**, **Python 3.13**, **Python 3.14**를 공식적으로 지원합니다.

만약, `Python 3.10`이나 `Python 3.11` 등의 낮은 버전을 사용한다면 장기 지원(LTS) 버전인 **Django 5.2**를 사용합니다.

- Tip : 장고는 약 8개월마다 기능 릴리스를 배포하며, `X.2` 버전(예: 4.2, 5.2)이 **장기 지원(LTS, Long-Term Support)** 버전입니다. LTS 버전은 최소 3년간 보안 패치를 제공합니다.

- Tip : 운영 서비스에서는 보안 패치가 포함된 **최신 패치 버전**(예: 6.1.x의 가장 높은 x)을 유지하는 것이 좋습니다.

<br>
<br>

## Django REST Framework 설치

{% highlight bash %}

pip install djangorestframework==3.18.3

{% endhighlight %}

`Django`에는 **REST(Representational State Transfer)** API를 위한 프레임워크가 존재합니다.

이를 `DRF(Django REST Framework)`라고 합니다.

`DRF`는 `pip`를 통하여 설치할 수 있습니다.

`DRF 3.18`은 **Django 5.2**, **Django 6.0**, **Django 6.1**을 지원합니다.

<br>
<br>

## REST API란?

`REST`란 **자원(Resource)**을 정의하고 자원에 대한 **주소(URI)**를 지정하는 방법을 의미합니다.

다음 여섯 가지 조건을 `REST`라고 하며 해당 조건을 지켜 설계된 API를 **RESTful API**라고 합니다.

* **Uniform Interface (일관적인 인터페이스)** : URI로 지정한 자원에 대해 통일되고 한정된 인터페이스(HTTP 메서드 등)로 작업을 수행하는 스타일
* **Client-Server (클라이언트-서버)** : 서버는 자원과 API를 제공하고, 클라이언트는 사용자 인터페이스와 사용자 상태(인증 정보 등)를 관리하여 서로의 역할을 분리
* **Stateless (무상태)** : 요청 간 클라이언트의 컨텍스트(context)가 서버에 저장되지 않으며, 각 요청은 처리에 필요한 정보를 모두 포함
* **Cacheable (캐시 처리 가능)** : 응답은 캐시 가능 여부를 명시해야 하며, 클라이언트는 응답을 캐싱할 수 있어야 함
* **Layered System (계층화)** : 클라이언트는 대상 서버에 직접 연결되었는지, 중간 서버를 통해 연결되었는지를 알 수 없음, 중간 서버는 로드 밸런싱 기능이나 공유 캐시 기능을 제공함으로써 시스템 규모 확장성을 향상시킬 수 있음
* **Code on Demand (주문형 코드, 선택 사항)** : 서버가 클라이언트에서 실행할 코드(JavaScript 등)를 전송해 기능을 확장할 수 있음

- Tip : `Self-descriptiveness (자체 표현 구조)`는 별도의 조건이 아니라, **Uniform Interface**를 구성하는 세부 조건 중 하나입니다. REST API 메시지만 보고도 쉽게 이해할 수 있도록 설계해야 한다는 의미입니다.

<br>

- `자원(Resource)` : URI
- `행위(Verb)` : HTTP Method (POST, GET, PUT, PATCH, DELETE)
- `표현(Representations)` : JSON, XML, TEXT, RSS 등 여러 형태로 응답
- `서버(Server)` : 자원을 가지고 있는 쪽
- `클라이언트(Client)` : 자원을 요청하는 쪽

<br>
<br>

## Django CORS 설치

{% highlight bash %}

pip install django-cors-headers==4.9.0

{% endhighlight %}

`CORS(Cross-Origin Resource Sharing)`란 교차 출처 리소스 공유라는 의미로 **실행 중인 웹 애플리케이션**이 다른 출처의 선택한 자원에 접근할 수 있는 권한을 부여하도록 브라우저에 알려주는 체제입니다.

즉, 브라우저에서 실행되는 다른 출처(Origin)의 웹 페이지가 서버에 요청할 때 `CORS`가 허용되어있지 않다면 `CORS 오류`가 발생합니다.

이를 방지하기 위해 `Django`용 `CORS` 패키지를 설치합니다.

- Tip : `출처(Origin)`는 **프로토콜 + 호스트 + 포트**의 조합입니다. 예를 들어, `http://localhost:3000`과 `http://localhost:8000`은 포트가 다르므로 서로 다른 출처입니다.

- Tip : CORS는 **브라우저**가 적용하는 정책입니다. `requests`나 `Postman`처럼 브라우저가 아닌 클라이언트에서 보내는 요청에는 적용되지 않습니다.

<br>
<br>

## 한 번에 설치하기

{% highlight bash %}

pip install django==6.1.2 djangorestframework==3.18.3 django-cors-headers==4.9.0

{% endhighlight %}

세 패키지를 한 번에 설치할 수도 있습니다.

프로젝트에서 사용하는 패키지는 `pip freeze > requirements.txt`로 저장한 다음, 다른 환경에서 `pip install -r requirements.txt`로 동일하게 설치할 수 있습니다.
