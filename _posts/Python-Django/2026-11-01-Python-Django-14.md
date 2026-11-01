---
layout: post
title: "Python Django 강좌 : 제 14강 - Router"
tagline: "Python Django REST Framework Router"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Router, Python Django DefaultRouter, Python Django SimpleRouter, Python Django register, Python Django basename, Python Django lookup_field, Python Django lookup_url_kwarg, Python Django lookup_value_converter, Python Django action, Python Django extra actions
ref: Python-Django
category: Python
permalink: /posts/Python-Django-14/
comments: true
toc: true
---

## Django Router

8강에서는 `ViewSet`을 URL에 연결하기 위해 `as_view({"get": "list", "post": "create"})`처럼 **HTTP 메서드와 동작(action)을 직접 바인딩**했습니다.

뷰셋이 많아지면 같은 형태의 URL 패턴을 반복해서 작성해야 하며, 동작을 추가할 때마다 `urls.py`도 함께 수정해야 합니다.

DRF는 이 과정을 자동으로 처리하는 `라우터(Router)`를 제공합니다.

라우터에 뷰셋을 **등록(register)**하면, 뷰셋이 가진 동작을 확인해 **목록 URL**과 **상세 URL**을 자동으로 생성합니다.

<br>

| URL | HTTP 메서드 | 동작(action) |
|:-:|:-:|:-:|
| `{prefix}/` | GET | list |
| `{prefix}/` | POST | create |
| `{prefix}/{lookup}/` | GET | retrieve |
| `{prefix}/{lookup}/` | PUT | update |
| `{prefix}/{lookup}/` | PATCH | partial_update |
| `{prefix}/{lookup}/` | DELETE | destroy |

<br>

`{prefix}`는 등록할 때 지정한 **경로 이름**이며, `{lookup}`은 하나의 대상을 찾기 위한 **식별자**입니다.

뷰셋에 구현되지 않은 동작은 URL에 연결되지 않습니다. 예를 들어, `ReadOnlyModelViewSet`을 등록하면 `list`, `retrieve`만 연결됩니다.

<br>
<br>

## SimpleRouter와 DefaultRouter

DRF는 두 종류의 기본 라우터를 제공합니다.

`SimpleRouter`는 위의 표와 같은 **목록 URL**과 **상세 URL**을 생성합니다.

`DefaultRouter`는 `SimpleRouter`의 기능에 더해, 등록된 모든 API의 목록을 보여주는 **API 루트 뷰(API Root View)**와 `.json`과 같은 **형식 접미사(Format suffix)** URL을 함께 생성합니다.

일반적으로 Browsable API로 API 목록을 확인하기 쉬운 `DefaultRouter`를 사용합니다.

<br>
<br>

## Django Code 구성

### views.py

{% highlight python %}

## first_app/views.py
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from first_app.models import UserModel
from first_app.serializers import UserSerializer


class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer
    lookup_url_kwarg = "uuid"
    lookup_value_converter = "uuid"

    # create, list, retrieve, update, destroy 메서드는 6강과 동일

    @action(detail=False, methods=["get"], url_path="event")
    def event(self, request, *args, **kwargs):
        queryset = self.get_queryset().filter(age__lt=30)
        serializer = UserSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

{% endhighlight %}

6강에서 작성한 `UserViewSet`에 세 가지 속성과 하나의 메서드를 추가합니다.

<br>

`lookup_url_kwarg`는 상세 URL에서 식별자를 전달할 **변수명**을 설정합니다.

기본값은 `lookup_field`의 값인 `pk`입니다. 6강에서 `retrieve(self, request, uuid=None)`처럼 `uuid` 변수명을 사용했으므로, `uuid`로 지정합니다.

`lookup_value_converter`는 라우터가 URL 패턴을 만들 때 사용할 **경로 변환기**를 설정합니다. 8강의 `<uuid:uuid>`와 같이 UUID 형식만 일치시킵니다.

- Tip : `lookup_field`는 **모델에서 대상을 찾을 필드 이름**이며, `lookup_url_kwarg`는 **URL의 변수명**입니다. 기본 구현의 `get_object()` 메서드는 `queryset.get(lookup_field=URL의 lookup_url_kwarg 값)`의 형태로 대상을 찾습니다.

- Tip : `lookup_value_converter`는 라우터를 `use_regex_path=False`로 생성했을 때 사용됩니다. 정규표현식 경로를 사용한다면 `lookup_value_regex`로 패턴을 지정합니다.

<br>

### action

`@action` 데코레이터를 사용하면 기본 동작 외에 **추가 동작(Extra action)**을 만들 수 있습니다.

라우터는 `@action`이 적용된 메서드를 찾아 URL을 자동으로 생성합니다.

<br>

| 매개변수 | 설명 |
|:-:|------|
| detail | `True`라면 상세 URL(`{prefix}/{lookup}/{url_path}/`), `False`라면 목록 URL(`{prefix}/{url_path}/`)에 연결합니다. (필수) |
| methods | 허용할 HTTP 메서드 목록입니다. 기본값은 `["get"]`입니다. |
| url_path | URL 경로 이름을 설정합니다. 기본값은 메서드 이름입니다. |
| url_name | URL 이름(name)을 설정합니다. 기본값은 메서드 이름의 밑줄(_)을 하이픈(-)으로 바꾼 값입니다. |

<br>

예제의 `event` 메서드는 `detail=False`이므로 `users/event/` URL에 연결되며, 나이가 30 미만인 사용자 목록을 반환합니다.

`detail=True`로 작성한다면 메서드에서 `self.get_object()`로 대상 객체를 가져올 수 있습니다.

<br>

{% highlight python %}

@action(detail=True, methods=["post"], url_path="birthday")
def birthday(self, request, *args, **kwargs):
    instance = self.get_object()
    instance.age += 1
    instance.save(update_fields=["age", "updated_date"])
    return Response({"age": instance.age}, status=status.HTTP_200_OK)

{% endhighlight %}

위의 예시는 `POST users/{uuid}/birthday/` 요청이 들어오면 해당 사용자의 나이를 1 증가시킵니다.

- Tip : `save(update_fields=[...])`는 지정한 필드만 `UPDATE` 문에 포함합니다. `auto_now` 필드도 갱신하려면 목록에 함께 작성해야 합니다.

<br>

### urls.py

{% highlight python %}

## daehee/urls.py
from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from first_app.views import UserViewSet
from blog.views import PostViewSet
from blog.views import CommentViewSet

router = DefaultRouter(use_regex_path=False)
router.register("users", UserViewSet)
router.register("post", PostViewSet)
router.register("comment", CommentViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include(router.urls)),
]

{% endhighlight %}

`DefaultRouter`의 인스턴스를 생성한 다음, `register(prefix, viewset)` 메서드로 뷰셋을 등록합니다.

`use_regex_path=False`로 생성하면 정규표현식 대신 `path` 함수의 **경로 변환기** 형식으로 URL을 생성합니다.

등록이 완료되면 `router.urls`에 생성된 URL 패턴 목록이 저장되며, `include(router.urls)`로 프로젝트 URL에 연결합니다.

이제 8강과 11강에서 작성한 `first_app/urls.py`, `blog/urls.py`의 `as_view()` 바인딩은 더 이상 필요하지 않으므로, 프로젝트 `urls.py`에서 `include("first_app.urls")`, `include("blog.urls")`를 제거합니다.

<br>

`register` 메서드는 `basename` 매개변수로 **URL 이름의 접두사**를 지정할 수 있습니다.

`basename`을 생략하면 뷰셋의 `queryset` 속성에서 모델 이름을 소문자로 가져와 사용합니다. 예제에서는 `usermodel`, `post`, `comment`가 됩니다.

- Tip : 뷰셋에 `queryset` 속성이 없고 `get_queryset()` 메서드만 정의했다면, 모델 이름을 알 수 없으므로 반드시 `basename`을 지정해야 합니다.

<br>
<br>

## 생성된 URL 확인

라우터가 생성한 URL은 다음과 같습니다.

<br>

| URL | 이름(name) |
|------|------|
| / | api-root |
| users/ | usermodel-list |
| users/event/ | usermodel-event |
| users/&lt;uuid:uuid&gt;/ | usermodel-detail |
| post/ | post-list |
| post/&lt;str:pk&gt;/ | post-detail |
| comment/ | comment-list |
| comment/&lt;str:pk&gt;/ | comment-detail |

<br>

`DefaultRouter`가 생성한 `api-root`(`http://127.0.0.1:8000/`)에 접속하면 다음과 같이 등록된 API 목록을 확인할 수 있습니다.

<br>

{% highlight json %}

{
    "users": "http://127.0.0.1:8000/users/",
    "post": "http://127.0.0.1:8000/post/",
    "comment": "http://127.0.0.1:8000/comment/"
}

{% endhighlight %}

URL 이름은 `{basename}-list`, `{basename}-detail`, `{basename}-{url_name}`의 형식으로 생성되며, `reverse("usermodel-detail", kwargs={"uuid": ...})`처럼 URL을 역으로 생성할 때 사용합니다.

`lookup_value_converter`를 지정하지 않은 `PostViewSet`, `CommentViewSet`은 기본 변환기인 `str`이 사용됩니다. 정수 ID만 받으려면 `lookup_value_converter = "int"`를 지정합니다.

<br>

라우터는 기본적으로 모든 URL의 끝에 **슬래시(/)**를 추가합니다.

그러므로 상세 URL은 `users/{uuid}`가 아닌 `users/{uuid}/`가 됩니다. 마지막 슬래시가 없는 URL을 사용하려면 `DefaultRouter(trailing_slash=False)`로 생성합니다.

- Tip : 8강에서 `as_view()`로 직접 작성한 상세 URL은 `users/{uuid}`처럼 슬래시가 없었으므로, 라우터로 변경한 뒤에는 클라이언트의 요청 URL도 함께 변경해야 합니다.

<br>
<br>

## 테스트

{% highlight python %}

import requests

base_url = "http://127.0.0.1:8000/users/"

response = requests.post(base_url, json={"email": "s076923@gmail.com", "name": "윤대희", "age": 20})
print(response.status_code)

users = requests.get(base_url).json()
uuid = users[0]["id"]

response = requests.patch(f"{base_url}{uuid}/", json={"age": 25})
print(response.status_code)

response = requests.get(f"{base_url}event/")
print(response.json())

{% endhighlight %}

**결과**
:    
201<br>
200<br>
[{'id': 'ff48338b-91f1-48d4-ad83-f27f05e6a59f', 'event_age': True, 'email': 's076923@gmail.com', 'name': '윤대희', 'age': 25, 'created_date': '2026-11-01T21:39:03.176973+09:00', 'updated_date': '2026-11-01T21:39:03.205544+09:00'}]<br>
<br>

`as_view()` 바인딩을 작성하지 않았지만, `POST`(create), `GET`(list), `PATCH`(partial_update), 추가 동작인 `event`까지 모두 연결된 것을 확인할 수 있습니다.

라우터를 사용하면 URL 설계가 일정한 규칙을 따르게 되므로, 뷰셋이 늘어나더라도 `urls.py`를 간결하게 유지할 수 있습니다.
