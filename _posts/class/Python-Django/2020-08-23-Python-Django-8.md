---
layout: post
title: "Python Django 강좌 : 제 8강 - URL"
tagline: "Python Django URL"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Django']
keywords: Python, Python Django, Python Django 3, Python Django URL, Python Django urls.py, Python Django Protocol, Python Django Host, Python Django Port, Python Django Resource Path, Python Django Query, Python Django Routing, Python Django django.conf.urls, Python Django ViewSet, Python Django urlpatterns, Python Django path, Python Django include, Python Django Regex
ref: Python
category: posts
permalink: /posts/Python-Django-8/
comments: true
---

## Django URL ##
----------

`URL(Uniform Resource Locators)`은 네트워크 상에서 **자원(Resource)**이 어디에 존재하는지 알려주기 위한 규약입니다.

URL의 기본 구조는 아래의 형태와 의미를 갖습니다.

`https://076923.github.io:8000/python/django?id=1000`

- `https` : 프로토콜(Protocol)
- `076923.github.io` : 호스트(Host)
- `8000` : 포트(Port)
- `python/django` : 리소스 경로(Resource Path)
- `query` : 쿼리(Query)

<br>

**https://076923.github.io:8000/(프로토콜 + 호스트 + 포트)**는 **호스팅**, **현재 IP 주소**, **설정** 등에 의해서 달라집니다.

2강에서 배운 `python manage.py runserver`를 통해 서버를 실행할 때, **http://127.0.0.1:8000/**를 통해 테스트 프로젝트를 확인할 수 있었습니다.

**프로토콜, 호스트, 포트**는 프로젝트 설정이나 호스팅등에 의해 달라지는 것을 알 수 있습니다.

<br>

다음으로 **/python/django?id=1000(리소스 경로 + 쿼리)** 등은 URL 설정에서 정의할 수 있습니다.

장고에서 URL은 **URL 경로와 일치하는 뷰(View)**를 `매핑(Mapping)`하거나 `라우팅(Routing)`하는 역할을 합니다.

즉, 장고에서 URL 설정은 하나의 항목을 연결하는 `퍼머링크(Permalink)`를 생성하거나 `쿼리스트링(Query string)` 등을 정의할 수 있습니다.

<br>

`urls.py` 파일에 URL 경로에 관한 논리를 정의합니다.

`urls.py` 파일은 [장고 프로젝트 이름]으로 생성한 폴더 아래에 포함되어 있습니다.

<br>

```

[현재 프로젝트]/
  ⬇ 📁 [장고 프로젝트 이름]
    🖹 __init__.py
    🖹 asgi.py
    🖹 settings.py
    🖹 urls.py
    🖹 wsgi.py
  > 📁 [장고 앱 이름]
  🖹 manage.py

```

<br>

## urls.py ##
----------

{% highlight python %}

"""daehee URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf.urls import url
from first_app.views import UserViewSet

urlpatterns = [
    url('users/(?P<uuid>[0-9a-f\-]{32,})$', UserViewSet.as_view({'get':'retrieve', 'put':'update', 'delete':'destroy'})),
    url('users', UserViewSet.as_view({'get':'list', 'post':'create'})),
]

{% endhighlight %}

URL(urls.py) 파일을 위와 같이 정의합니다.

어떤 **리소스 경로**나 **쿼리**로 접근했을 때, 연결될 뷰를 설정합니다.

<br>
<br>

## Module ##
----------

{% highlight python %}

from django.conf.urls import url
from first_app.views import UserViewSet

{% endhighlight %}

`from django.conf.urls import url`는 URL 연결과 관련된 모듈입니다.

`from first_app.views import UserViewSet`는 `first_app` 앱의 `views.py`에서 선언한 UserViewSet 클래스입니다.

<br>
<br>

## urlpatterns ##
----------

{% highlight python %}

urlpatterns = [
    url('users/(?P<uuid>[0-9a-f\-]{32,})$', UserViewSet.as_view({'get':'retrieve', 'put':'update', 'delete':'destroy'})),
    url('users', UserViewSet.as_view({'get':'list', 'post':'create'})),
]

{% endhighlight %}

`urlpatterns` 목록 안에 현재 프로젝트에서 사용될 URL 경로를 url 함수를 통해 설정합니다.

`url(경로, ViewSet 클래스)`를 이용하여 **경로(Path)**와 **ViewSet 클래스**를 연결합니다.

현재 ViewSet 클래스는 `create(POST)`, `list(GET)`, `retrieve(GET)`, `update(PUT)`, `destroy(DELETE)`로 다섯 가지의 함수가 선언되어 있습니다.

`retrieve`, `update`, `destroy`는 하나의 대상에 대해 작업을 진행하며, `create`, `list`는 특별한 대상을 상대로 작업이 진행되지 않습니다.

모든 대상은 `users`의 경로이며, 하나의 대상은 `users/<uuid>`의 경로로 볼 수 있습니다.

그러므로, 특별한 대상으로 작업을 진행하는 `url`부터 먼저 선언되어야 합니다.

그 이유는 `if-elif`의 구조로 생각하면 이해하기 쉽습니다.

<br>

{% highlight python %}

if 'users' in path:
    return "모든 대상"
elif 'users/<uuid>' in path:
    return "하나의 대상"

{% endhighlight %}

만약, 위의 구조로 `urlpatterns`가 정의되어 있다면, 하나의 대상으로 작업하는 경로가 인식되지 않아 `retrieve`, `update`, `destroy`는 접근할 수 없습니다.

그러므로, 항상 세부 구조를 탐색하는 경로일수록 **상단에 배치**해 사용합니다.

<br>

`경로(Path)`는 **리소스 경로(Resource Path)**나, **쿼리(Query)**를 설정할 수 있습니다.

모델에서 고유 id는 `UUID`를 설정했으므로, `UUID`를 통해 접근하도록 하겠습니다.

`http://127.0.0.1:8000/users/<UUID>`로 접근하려면 `UUID` 패턴을 인식해야 합니다.

`url` 함수는 정규표현식을 지원하므로, 정규표현식을 활용해 `UUID` 패턴을 검증하도록 하겠습니다.

<br>

`'users/(?P<uuid>[0-9a-f\-]{32,})$'`의 구조로 하나의 대상에 접근할 수 있습니다.

`(?P)`는 해당 영역 내부의 문자는 **정규표현식을 적용**한다는 의미입니다.

`<uuid>`는 정규표현식으로 작성된 url 경로를 `uuid`라는 **변수**명으로 `뷰(View)`에 전달한다는 의미가 됩니다.

`[0-9a-f\-]{32,}`는 간단하게 작성된 **UUID 패턴**입니다.

즉, UUID 정규표현식 패턴을 `uuid`라는 변수로 뷰에 제공한다는 의미가 됩니다.

뷰에서는 `uuid` 변수를 활용해 `http://127.0.0.1:8000/users/<UUID>`로 접근한 `<UUID>` 값을 확인할 수 있습니다.

<br>

`ViewSet 클래스`는 `as_view()`으로 리소스 작업을 `HTTP 메서드`에 바인딩할 수 있습니다.

사전의 구조로 바인딩 할 수 있으며, **{'HTTP 메서드', 'ViewSet 메서드'}**의 구조를 갖습니다.

특별한 경우가 아니라면 HTTP 메서드는 중복해서 사용하지 않습니다.

<br>

* Tip : 더 정확한 UUID 패턴은 `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`으로 사용할 수 있습니다.

<br>
<br>

## 라우팅(Routing) ##
----------

`라우팅(Rouing)`이란 네트워크 상에서 데이터를 전달할 때, 목적지까지의 경로를 체계적으로 결정하는 **경로 선택 과정**을 의미합니다.

현재 장고 프로젝트는 하나의 애플리케이션을 가지고 있으므로, 매우 작은 네트워크로 볼 수 있습니다.

만약, 장고 프로젝트가 매우 커진다면 하나의 `urls.py`에서 관리하기가 어려워질 수 있습니다.

그러므로 **애플리케이션**마다 `urls.py`를 생성해 **프로젝트** `urls.py`에 연결해 관리할 수 있습니다.

앱(first_app)에 `urls.py` 파일을 생성해 아래의 코드처럼 생성합니다.

<br>

{% highlight python %}

## first_app/urls.py
from django.conf.urls import url
from first_app.views import UserViewSet

urlpatterns = [
    url('(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$', UserViewSet.as_view({ 'get':'retrieve', 'put':'update', 'delete':'destroy'})),
    url('', UserViewSet.as_view({ 'get':'list', 'post':'create'})),
]

{% endhighlight %}

`프로젝트 urls.py`와의 차이점은 `경로(Path)`에 `users`가 존재하지 않습니다.

제거된 `users`를 `프로젝트 urls.py`에 설정합니다.

<br>

{% highlight python %}

## daehee/urls.py
from django.urls import path
from django.conf.urls import include

urlpatterns = [
    path('users', include('first_app.urls'))
]

{% endhighlight %}

`프로젝트 urls.py`는 새로운 두 종류의 모듈이 추가됩니다.

`from django.urls import path`는 라우팅할 경로를 설정할 수 있습니다.

`from django.conf.urls import include`는 다른 `urls.py`를 가져와 읽을 수 있는 역할을 합니다.

기본 URL 설정과 동일하게 `urlpatterns` 안에 작성합니다.

`path(경로, 다른 urls.py 경로)`로 설정할 수 있습니다.

`users` 경로를 접근했을 때, `include('first_app.urls')` 경로의 `first_app/urls.py`로 이동해 접근합니다.

즉, `users`로 이동했을 때 URL 경로를 재 탐색하므로, `first_app/urls.py`에는 `users`를 작성하지 않습니다.

