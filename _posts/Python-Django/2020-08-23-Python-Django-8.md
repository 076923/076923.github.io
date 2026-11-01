---
layout: post
title: "Python Django 강좌 : 제 8강 - URL"
tagline: "Python Django URL"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django URL, Python Django urls.py, Python Django Protocol, Python Django Host, Python Django Port, Python Django Resource Path, Python Django Query, Python Django Routing, Python Django path, Python Django re_path, Python Django path converter, Python Django ViewSet, Python Django urlpatterns, Python Django include
ref: Python-Django
category: Python
permalink: /posts/Python-Django-8/
comments: true
toc: true
---

## Django URL

`URL(Uniform Resource Locator)`은 네트워크 상에서 **자원(Resource)**이 어디에 존재하는지 알려주기 위한 규약입니다.

URL의 기본 구조는 아래의 형태와 의미를 갖습니다.

`https://076923.github.io:8000/python/django?id=1000`

- `https` : 프로토콜(Protocol)
- `076923.github.io` : 호스트(Host)
- `8000` : 포트(Port)
- `/python/django` : 리소스 경로(Resource Path)
- `id=1000` : 쿼리(Query)

<br>

**https://076923.github.io:8000(프로토콜 + 호스트 + 포트)**는 **호스팅**, **현재 IP 주소**, **설정** 등에 의해서 달라집니다.

2강에서 배운 `python manage.py runserver`를 통해 서버를 실행할 때, **http://127.0.0.1:8000/**를 통해 테스트 프로젝트를 확인할 수 있었습니다.

**프로토콜, 호스트, 포트**는 프로젝트 설정이나 호스팅 등에 의해 달라지는 것을 알 수 있습니다.

<br>

다음으로 **/python/django(리소스 경로)**는 URL 설정에서 정의할 수 있습니다.

장고에서 URL 설정은 **URL 경로와 일치하는 뷰(View)**를 연결하는 `URL 매핑(Mapping)` 또는 `라우팅(Routing)` 역할을 합니다.

즉, 장고에서 URL 설정으로 하나의 항목을 가리키는 `퍼머링크(Permalink)` 등을 정의할 수 있습니다.

- Tip : 장고의 URL 설정은 **리소스 경로**만 비교합니다. **?id=1000**과 같은 `쿼리스트링(Query string)`은 URL 설정에 작성하지 않고, 뷰에서 `request.GET`(DRF에서는 `request.query_params`)으로 읽습니다.

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
<br>

## urls.py

{% highlight python %}

"""
URL configuration for daehee project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
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

from django.contrib import admin
from django.urls import path
from first_app.views import UserViewSet

urlpatterns = [
    path("admin/", admin.site.urls),
    path("users/<uuid:uuid>", UserViewSet.as_view({"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"})),
    path("users/", UserViewSet.as_view({"get": "list", "post": "create"})),
]

{% endhighlight %}

URL(urls.py) 파일을 위와 같이 정의합니다.

어떤 **리소스 경로**로 접근했을 때, 연결될 뷰를 설정합니다.

<br>

### Module

{% highlight python %}

from django.contrib import admin
from django.urls import path
from first_app.views import UserViewSet

{% endhighlight %}

`from django.contrib import admin`은 장고의 관리자 페이지입니다. 프로젝트 생성 시 기본으로 등록되어 있습니다.

`from django.urls import path`는 URL 경로와 뷰를 연결하는 함수입니다.

`from first_app.views import UserViewSet`은 `first_app` 앱의 `views.py`에서 선언한 UserViewSet 클래스입니다.

- Tip : 예전 버전에서 사용하던 `from django.conf.urls import url`은 Django 4.0에서 **제거**되었습니다. 정규표현식이 필요하다면 `django.urls.re_path`를 사용합니다.

<br>

### urlpatterns

{% highlight python %}

urlpatterns = [
    path("admin/", admin.site.urls),
    path("users/<uuid:uuid>", UserViewSet.as_view({"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"})),
    path("users/", UserViewSet.as_view({"get": "list", "post": "create"})),
]

{% endhighlight %}

`urlpatterns` 목록 안에 현재 프로젝트에서 사용될 URL 경로를 `path` 함수를 통해 설정합니다.

`path(경로, 뷰)`를 이용하여 **경로(Path)**와 **뷰**를 연결합니다.

현재 ViewSet 클래스는 `create(POST)`, `list(GET)`, `retrieve(GET)`, `update(PUT)`, `partial_update(PATCH)`, `destroy(DELETE)`의 동작을 사용할 수 있습니다.

`retrieve`, `update`, `partial_update`, `destroy`는 하나의 대상에 대해 작업을 진행하며, `create`, `list`는 특정 대상이 아닌 전체 목록을 상대로 작업이 진행됩니다.

모든 대상은 `users/`의 경로이며, 하나의 대상은 `users/<uuid>`의 경로로 볼 수 있습니다.

<br>

장고는 요청된 경로를 `urlpatterns`의 **위에서부터 순서대로** 비교하고, **처음으로 일치하는** 패턴의 뷰를 실행합니다.

그 이유는 `if-elif`의 구조로 생각하면 이해하기 쉽습니다.

<br>

{% highlight python %}

if match("users/", path):
    return "모든 대상"
elif match("users/<uuid:uuid>", path):
    return "하나의 대상"

{% endhighlight %}

`path` 함수는 경로 **전체가 일치**하는지 비교하므로, 위의 두 패턴은 순서와 관계없이 올바르게 동작합니다.

하지만 정규표현식을 사용하는 `re_path`에서 `^`, `$` 기호로 시작과 끝을 고정하지 않거나, `<str:name>`처럼 넓은 범위를 받는 패턴을 사용한다면 앞의 패턴이 뒤의 패턴을 가로챌 수 있습니다.

그러므로, 항상 **구체적인 경로일수록 상단에 배치**해 사용하는 것이 안전합니다.

<br>

### 경로 변환기(Path converter)

모델에서 고유 id는 `UUID`를 설정했으므로, `UUID`를 통해 접근하도록 하겠습니다.

`http://127.0.0.1:8000/users/<UUID>`로 접근하려면 `UUID` 패턴을 인식해야 합니다.

`path` 함수는 `<변환기:변수명>` 형태의 **경로 변환기(Path converter)**를 지원합니다.

`<uuid:uuid>`는 경로에서 **UUID 형식의 문자열**만 일치시키고, 그 값을 `uuid.UUID` 객체로 변환해 `uuid`라는 **변수명**으로 `뷰(View)`에 전달한다는 의미입니다.

뷰에서는 `uuid` 변수를 활용해 `http://127.0.0.1:8000/users/<UUID>`로 접근한 `<UUID>` 값을 확인할 수 있습니다.

UUID 형식이 아닌 값으로 접근하면 일치하는 패턴이 없으므로 `404 Not Found`가 반환됩니다.

<br>

| 변환기 | 일치하는 값 |
|:-:|------|
| str | 슬래시(/)를 제외한 비어 있지 않은 문자열 (기본값) |
| int | 0 또는 양의 정수 |
| slug | 영문자, 숫자, 하이픈(-), 밑줄(_)로 구성된 문자열 |
| uuid | 소문자 UUID 형식의 문자열 (예 : 075194d3-6885-417e-a8a8-6c931e272f00) |
| path | 슬래시(/)를 포함한 비어 있지 않은 문자열 |

<br>

### re_path

{% highlight python %}

from django.urls import re_path

urlpatterns = [
    re_path(r"^users/(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", UserViewSet.as_view({"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"})),
    re_path(r"^users/$", UserViewSet.as_view({"get": "list", "post": "create"})),
]

{% endhighlight %}

경로 변환기로 표현하기 어려운 패턴은 `re_path` 함수로 **정규표현식**을 사용할 수 있습니다.

`(?P<uuid>패턴)`은 **이름이 있는 그룹(Named group)**으로, 괄호 안의 정규표현식과 일치한 문자열을 `uuid`라는 **변수**명으로 `뷰(View)`에 전달한다는 의미가 됩니다.

`^`와 `$`는 각각 경로의 **시작**과 **끝**을 의미하며, 경로 전체가 일치하도록 고정합니다.

- Tip : `re_path`로 전달되는 값은 변환되지 않은 **문자열**입니다.

<br>

### as_view

`ViewSet 클래스`는 `as_view()`로 **HTTP 메서드**를 **ViewSet의 동작(action)**에 바인딩할 수 있습니다.

사전의 구조로 바인딩 할 수 있으며, **{'HTTP 메서드': 'ViewSet 메서드'}**의 구조를 갖습니다.

하나의 경로에서 같은 HTTP 메서드는 하나의 동작에만 연결할 수 있습니다.

- Tip : 14강 `Router`에서는 이 바인딩 과정을 자동으로 처리하는 방법을 다룹니다.

<br>
<br>

## 라우팅(Routing)

`라우팅(Routing)`이란 네트워크 상에서 데이터를 전달할 때, 목적지까지의 경로를 체계적으로 결정하는 **경로 선택 과정**을 의미합니다.

현재 장고 프로젝트는 하나의 애플리케이션을 가지고 있으므로, 매우 작은 네트워크로 볼 수 있습니다.

만약, 장고 프로젝트가 매우 커진다면 하나의 `urls.py`에서 관리하기가 어려워질 수 있습니다.

그러므로 **애플리케이션**마다 `urls.py`를 생성해 **프로젝트** `urls.py`에 연결해 관리할 수 있습니다.

앱(first_app)에 `urls.py` 파일을 생성해 아래의 코드처럼 생성합니다.

<br>

{% highlight python %}

## first_app/urls.py
from django.urls import path
from first_app.views import UserViewSet

urlpatterns = [
    path("<uuid:uuid>", UserViewSet.as_view({"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"})),
    path("", UserViewSet.as_view({"get": "list", "post": "create"})),
]

{% endhighlight %}

`프로젝트 urls.py`와의 차이점은 `경로(Path)`에 `users/`가 존재하지 않습니다.

제거된 `users/`를 `프로젝트 urls.py`에 설정합니다.

<br>

{% highlight python %}

## daehee/urls.py
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("users/", include("first_app.urls")),
]

{% endhighlight %}

`프로젝트 urls.py`에는 `include` 함수가 추가됩니다.

`from django.urls import include`는 다른 `urls.py`를 가져와 연결하는 역할을 합니다.

기본 URL 설정과 동일하게 `urlpatterns` 안에 작성합니다.

`path(경로, include(다른 urls.py 모듈 경로))`로 설정할 수 있습니다.

`users/` 경로로 시작하는 요청은 일치한 부분(`users/`)을 잘라낸 다음, **나머지 경로**를 `include("first_app.urls")`의 `first_app/urls.py`에서 이어서 탐색합니다.

즉, `users/`로 이동했을 때 나머지 URL 경로를 다시 탐색하므로, `first_app/urls.py`에는 `users/`를 작성하지 않습니다.

- Tip : `include`를 사용할 때는 프로젝트 경로를 `users/`처럼 **슬래시(/)로 끝나게** 작성해야 합니다. `users`로 작성하면 앱의 `<uuid:uuid>` 패턴과 이어져 `users075194d3-...`처럼 의도하지 않은 경로가 만들어집니다.

- Tip : 장고는 `APPEND_SLASH = True`가 기본값이므로, `GET /users`처럼 마지막 슬래시가 빠진 요청은 `/users/`로 **리디렉션**합니다. 단, `POST` 요청은 리디렉션할 수 없으므로 개발 모드에서 `RuntimeError`가 발생합니다. 요청할 때는 URL을 정확히 작성합니다.
