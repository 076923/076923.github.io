---
layout: post
title: "Python Django 강좌 : 제 15강 - Authentication & Permission"
tagline: "Python Django REST Framework Authentication and Permission"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Authentication, Python Django Permission, Python Django TokenAuthentication, Python Django SessionAuthentication, Python Django authtoken, Python Django obtain_auth_token, Python Django IsAuthenticated, Python Django IsAuthenticatedOrReadOnly, Python Django AllowAny, Python Django BasePermission, Python Django has_object_permission, Python Django 401, Python Django 403
ref: Python-Django
category: Python
permalink: /posts/Python-Django-15/
comments: true
toc: true
---

## Authentication & Permission

지금까지 작성한 API는 **누구나** 데이터를 조회하고, 작성하고, 수정하고, 삭제할 수 있습니다.

실제 서비스에서는 로그인한 사용자만 게시물을 작성하거나, 작성자 본인만 게시물을 수정할 수 있도록 제한해야 합니다.

DRF는 이를 위해 `인증(Authentication)`과 `권한(Permission)` 기능을 제공합니다.

`인증`은 요청을 보낸 **사용자가 누구인지 확인**하는 과정입니다. 인증에 성공하면 `request.user`에 사용자 객체가 저장되며, 실패하거나 인증 정보가 없다면 `AnonymousUser`가 저장됩니다.

`권한`은 확인된 사용자가 **해당 요청을 수행할 수 있는지 판단**하는 과정입니다.

DRF는 뷰의 메서드를 실행하기 전에 **인증 → 권한 확인** 순서로 검사를 진행합니다.

<br>

| 상태 코드 | 의미 |
|:-:|------|
| 401 Unauthorized | 인증 정보가 없거나 올바르지 않음 (누구인지 알 수 없음) |
| 403 Forbidden | 인증은 되었지만 요청을 수행할 권한이 없음 |

<br>

- Tip : 인증 방식에 따라 인증 정보가 없을 때 `401` 대신 `403`이 반환될 수 있습니다. `401` 응답에는 `WWW-Authenticate` 헤더가 필요한데, `SessionAuthentication`처럼 이 헤더를 제공하지 않는 방식이 **첫 번째 인증 클래스**라면 `403`이 반환됩니다.

<br>
<br>

## 인증 방식

DRF가 기본으로 제공하는 인증 방식은 다음과 같습니다.

<br>

| 인증 클래스 | 설명 |
|:-:|------|
| SessionAuthentication | 장고의 세션(로그인 쿠키)으로 인증합니다. 같은 도메인의 웹 페이지나 Browsable API에서 사용합니다. |
| TokenAuthentication | 사용자마다 발급한 토큰을 `Authorization` 헤더로 전달해 인증합니다. 앱이나 다른 서버에서 사용합니다. |
| BasicAuthentication | 아이디와 비밀번호를 매 요청의 헤더에 담아 인증합니다. 테스트 용도로만 사용합니다. |
| RemoteUserAuthentication | 웹 서버 등 외부에서 인증한 사용자 정보를 사용합니다. |

<br>

이번 강좌에서는 모바일 앱이나 프런트엔드에서 많이 사용하는 `TokenAuthentication`을 사용합니다.

- Tip : 만료 시간이 있는 토큰이나 JWT(JSON Web Token)가 필요하다면 `djangorestframework-simplejwt`와 같은 서드파티 패키지를 사용합니다. DRF의 기본 토큰은 **만료 시간이 없습니다.**

<br>
<br>

## 프로젝트 설정

### settings.py

{% highlight python %}

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'first_app',
    'blog',
]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
}

{% endhighlight %}

토큰을 저장할 테이블이 필요하므로, `INSTALLED_APPS`에 `rest_framework.authtoken`을 추가합니다.

`REST_FRAMEWORK` 설정은 DRF의 **전역 설정**입니다.

`DEFAULT_AUTHENTICATION_CLASSES`는 모든 뷰에 기본으로 적용할 **인증 클래스 목록**입니다. 목록의 순서대로 인증을 시도하며, 처음으로 성공한 방식을 사용합니다.

`DEFAULT_PERMISSION_CLASSES`는 모든 뷰에 기본으로 적용할 **권한 클래스 목록**입니다. 목록의 **모든 권한을 통과**해야 요청이 허용됩니다.

`IsAuthenticatedOrReadOnly`는 인증된 사용자는 모든 요청을, 인증되지 않은 사용자는 **읽기 요청(GET, HEAD, OPTIONS)**만 허용합니다.

- Tip : `DEFAULT_PERMISSION_CLASSES`를 설정하지 않으면 기본값은 `AllowAny`(모두 허용)입니다. 그래서 지금까지는 누구나 모든 요청을 보낼 수 있었습니다.

<br>

{% highlight bash %}

python manage.py migrate

{% endhighlight %}

**결과**
:    
{% highlight text %}

Operations to perform:
  Apply all migrations: admin, auth, authtoken, blog, contenttypes, first_app, sessions
Running migrations:
  Applying authtoken.0001_initial... OK
  Applying authtoken.0002_auto_20160226_1747... OK
  Applying authtoken.0003_tokenproxy... OK
  Applying authtoken.0004_alter_tokenproxy_options... OK

{% endhighlight %}

`authtoken` 앱의 마이그레이션을 적용해 토큰 테이블(`authtoken_token`)을 생성합니다.

<br>

### 권한 클래스

DRF가 기본으로 제공하는 권한 클래스는 다음과 같습니다.

<br>

| 권한 클래스 | 설명 |
|:-:|------|
| AllowAny | 모든 요청을 허용합니다. |
| IsAuthenticated | 인증된 사용자의 요청만 허용합니다. |
| IsAdminUser | `is_staff`가 `True`인 사용자의 요청만 허용합니다. |
| IsAuthenticatedOrReadOnly | 인증된 사용자는 모든 요청을, 인증되지 않은 사용자는 읽기 요청만 허용합니다. |
| DjangoModelPermissions | 장고의 모델 권한(add, change, delete, view)에 따라 허용합니다. |

<br>

권한 클래스는 `&`(AND), `|`(OR), `~`(NOT) 연산자로 조합할 수 있습니다. (예 : `IsAdminUser | IsAuthenticatedOrReadOnly`)

<br>
<br>

## 사용자와 토큰 생성

{% highlight bash %}

python manage.py createsuperuser

{% endhighlight %}

테스트에 사용할 사용자를 생성합니다. 아이디, 이메일, 비밀번호를 순서대로 입력합니다.

생성한 사용자로 `http://127.0.0.1:8000/admin/` 관리자 페이지에 로그인할 수도 있습니다.

<br>

{% highlight bash %}

python manage.py drf_create_token daehee

{% endhighlight %}

**결과**
:    
Generated token 14bc4f86b3d70c249620d8b74fe5e7c6a45746bf for user daehee<br>
<br>

`python manage.py drf_create_token [아이디]`로 해당 사용자의 토큰을 발급할 수 있습니다.

하지만 실제 서비스에서는 사용자가 **아이디와 비밀번호로 토큰을 요청**할 수 있어야 하므로, 토큰 발급 API를 추가합니다.

<br>

### urls.py

{% highlight python %}

## daehee/urls.py
from django.contrib import admin
from django.urls import include, path
from rest_framework.authtoken.views import obtain_auth_token
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
    path("api-token-auth/", obtain_auth_token),
    path("", include(router.urls)),
]

{% endhighlight %}

14강의 `urls.py`에 `obtain_auth_token` 뷰를 추가합니다.

`obtain_auth_token`은 `username`과 `password`를 `POST`로 받아, 올바른 사용자라면 토큰을 반환하는 뷰입니다.

토큰이 없다면 새로 생성하고, 이미 있다면 기존 토큰을 반환합니다.

<br>
<br>

## 작성자 기반 권한

### models.py

{% highlight python %}

## blog/models.py
from django.conf import settings
from django.db import models


class Post(models.Model):
    id = models.BigAutoField(help_text="Post ID", primary_key=True)
    title = models.CharField(help_text="Post title", max_length=100, blank=False, null=False)
    contents = models.TextField(help_text="Post contents", blank=False, null=False)
    author = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="posts", on_delete=models.CASCADE, null=True, blank=True)

{% endhighlight %}

게시물의 작성자를 저장하기 위해, `Post` 모델에 사용자 모델을 참조하는 `author` 외래키를 추가합니다.

`settings.AUTH_USER_MODEL`은 현재 프로젝트에서 사용하는 **사용자 모델**을 가리킵니다. 기본값은 `auth.User`입니다.

이미 작성된 게시물에는 작성자 정보가 없으므로, `null=True`로 선언해 기존 데이터를 유지합니다.

- Tip : 사용자 모델을 `from django.contrib.auth.models import User`로 직접 참조하지 않고 `settings.AUTH_USER_MODEL`을 사용하면, 나중에 사용자 정의 사용자 모델로 변경하더라도 코드를 수정하지 않아도 됩니다.

<br>

{% highlight bash %}

python manage.py makemigrations blog
python manage.py migrate blog

{% endhighlight %}

**결과**
:    
{% highlight text %}

Migrations for 'blog':
  blog\migrations\0002_post_author.py
    + Add field author to post
Operations to perform:
  Apply all migrations: blog
Running migrations:
  Applying blog.0002_post_author... OK

{% endhighlight %}

모델이 변경되었으므로 마이그레이션을 생성하고 적용합니다.

<br>

### serializers.py

{% highlight python %}

## blog/serializers.py
class PostSerializer(serializers.ModelSerializer):
    post = CommentSerializer(many=True, read_only=True)
    author = serializers.ReadOnlyField(source="author.username")

    class Meta:
        model = Post
        fields = ("id", "title", "contents", "author", "post")

{% endhighlight %}

`author` 필드는 클라이언트가 직접 입력하지 않고, **요청을 보낸 사용자**로 자동 저장해야 합니다.

그러므로 `ReadOnlyField`로 선언해 응답에만 포함하고, `source="author.username"`으로 사용자의 **아이디**를 출력합니다.

작성자가 없는 게시물은 `null`로 표시됩니다.

<br>

### permissions.py

{% highlight python %}

## blog/permissions.py
from rest_framework import permissions


class IsAuthorOrReadOnly(permissions.BasePermission):
    message = "작성자만 수정하거나 삭제할 수 있습니다."

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True

        return obj.author == request.user

{% endhighlight %}

작성자 본인만 수정할 수 있는 **사용자 정의 권한 클래스**를 작성합니다.

권한 클래스는 `BasePermission`을 상속받으며, 두 종류의 메서드를 재정의할 수 있습니다.

`has_permission(self, request, view)`는 **모든 요청**에 대해 뷰가 실행되기 전에 호출됩니다.

`has_object_permission(self, request, view, obj)`는 **특정 객체**에 대한 요청에서 호출되며, `obj`에 대상 객체가 전달됩니다.

`SAFE_METHODS`는 데이터를 변경하지 않는 `GET`, `HEAD`, `OPTIONS` 메서드의 목록입니다. 읽기 요청은 모두 허용하고, 그 외의 요청은 `obj.author`가 요청한 사용자와 같을 때만 허용합니다.

`message`는 권한 확인에 실패했을 때 응답에 표시될 메세지입니다.

- Tip : `has_object_permission`은 뷰에서 `self.get_object()`를 호출할 때 실행됩니다. 기본 `retrieve`, `update`, `destroy`는 `get_object()`를 사용하지만, 6강처럼 `Model.objects.get()`으로 직접 객체를 가져오도록 재정의했다면 **객체 권한이 확인되지 않으므로** `self.check_object_permissions(request, obj)`를 직접 호출해야 합니다.

<br>

### views.py

{% highlight python %}

## blog/views.py
from django.db import transaction
from rest_framework import status
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response

from blog.models import Post
from blog.models import Comment
from blog.serializers import PostSerializer
from blog.serializers import CommentSerializer
from blog.permissions import IsAuthorOrReadOnly


class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.select_related("author").prefetch_related("post")
    serializer_class = PostSerializer
    permission_classes = [IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]

    def create(self, request, *args, **kwargs):
        with transaction.atomic():
            post_data = {
                "title": request.data.get("title"),
                "contents": request.data.get("contents"),
            }
            post_serializer = PostSerializer(data=post_data)
            post_serializer.is_valid(raise_exception=True)
            post_result = post_serializer.save(author=request.user)

            comment_data = {
                "post_id": post_result.id,
                "contents": request.data.get("post", {}).get("contents"),
            }
            comment_serializer = CommentSerializer(data=comment_data)
            comment_serializer.is_valid(raise_exception=True)
            comment_serializer.save()

        return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)


class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.select_related("post_id")
    serializer_class = CommentSerializer

{% endhighlight %}

`permission_classes` 속성으로 해당 뷰셋에만 적용할 권한 클래스를 지정합니다. 뷰에 지정한 값은 전역 설정(`DEFAULT_PERMISSION_CLASSES`)을 **대체**합니다.

`IsAuthenticatedOrReadOnly`로 인증되지 않은 사용자의 작성을 막고, `IsAuthorOrReadOnly`로 작성자가 아닌 사용자의 수정과 삭제를 막습니다.

13강의 `create()` 메서드에서 `post_serializer.save(author=request.user)`로 변경합니다.

`save()` 메서드에 키워드 인수를 전달하면, 유효성 검사를 통과한 데이터에 **추가 값**을 더해 저장합니다.

`queryset`에는 `select_related("author")`를 추가해 게시물 목록을 조회할 때 작성자 정보를 `JOIN`으로 함께 가져옵니다.

- Tip : 인증 클래스도 `authentication_classes` 속성으로 뷰마다 다르게 지정할 수 있습니다.

<br>

### 회원 가입 허용

{% highlight python %}

## first_app/views.py
from rest_framework.permissions import AllowAny


class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer
    lookup_url_kwarg = "uuid"
    lookup_value_converter = "uuid"
    permission_classes = [AllowAny]

{% endhighlight %}

전역 설정으로 `IsAuthenticatedOrReadOnly`를 적용했으므로, 이제 `UserViewSet`도 인증되지 않은 사용자는 `POST` 요청을 보낼 수 없습니다.

예제의 `UserViewSet`은 회원 가입처럼 **누구나 작성**할 수 있어야 하므로, `permission_classes = [AllowAny]`로 지정해 전역 설정을 덮어씁니다.

- Tip : `AllowAny`는 수정과 삭제도 모두 허용하므로, 실제 서비스에서는 `create`만 허용하고 나머지 동작은 인증을 요구하도록 `get_permissions()` 메서드를 재정의해 동작(`self.action`)별로 권한을 나눕니다.

<br>
<br>

## 테스트

### 인증 없이 요청

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/post/"
data = {
    "title": "제 15강 - Authentication & Permission",
    "contents": "장고에서 ...",
    "post": {"contents": "로그인 후 작성"}
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.headers.get("WWW-Authenticate"))
print(response.json())

{% endhighlight %}

**결과**
:    
401<br>
Token<br>
{'detail': 'Authentication credentials were not provided.'}<br>
<br>

인증 정보 없이 게시물을 작성하면 `401 Unauthorized`가 반환됩니다.

`TokenAuthentication`이 첫 번째 인증 클래스이므로, `WWW-Authenticate: Token` 헤더로 **토큰 인증이 필요하다**는 사실을 알려줍니다.

읽기 요청인 `GET`은 인증 없이도 정상적으로 조회됩니다.

<br>

### 토큰 발급

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/api-token-auth/"
data = {
    "username": "daehee",
    "password": "비밀번호"
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())

{% endhighlight %}

**결과**
:    
200<br>
{'token': '14bc4f86b3d70c249620d8b74fe5e7c6a45746bf'}<br>
<br>

아이디와 비밀번호로 토큰을 발급받습니다.

비밀번호가 올바르지 않다면 `400 Bad Request`와 `Unable to log in with provided credentials.` 메세지가 반환됩니다.

<br>

### 토큰으로 요청

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/post/"
headers = {
    "Authorization": "Token 14bc4f86b3d70c249620d8b74fe5e7c6a45746bf"
}
data = {
    "title": "제 15강 - Authentication & Permission",
    "contents": "장고에서 ...",
    "post": {"contents": "로그인 후 작성"}
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code)

response = requests.get(url)
print(response.json()[-1])

{% endhighlight %}

**결과**
:    
201<br>
{'id': 6, 'title': '제 15강 - Authentication & Permission', 'contents': '장고에서 ...', 'author': 'daehee', 'post': [{'post_id': {'id': 6, 'title': '제 15강 - Authentication & Permission', 'contents': '장고에서 ...'}, 'contents': '로그인 후 작성'}]}<br>
<br>

`Authorization` 헤더에 `Token {토큰}` 형식으로 토큰을 전달합니다. `Token`과 토큰 값 사이에는 **공백 한 칸**이 있어야 합니다.

인증에 성공하면 게시물이 작성되며, `author` 필드에 요청한 사용자의 아이디가 저장된 것을 확인할 수 있습니다.

- Tip : 포스트맨(Postman)에서는 `Headers` 탭에 `Authorization` 키와 `Token {토큰}` 값을 추가해 요청합니다.

<br>

### 다른 사용자로 수정

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/post/6/"
headers = {
    "Authorization": "Token {다른 사용자의 토큰}"
}

response = requests.patch(url, json={"title": "제목 변경"}, headers=headers)

print(response.status_code)
print(response.json())

{% endhighlight %}

**결과**
:    
403<br>
{'detail': '작성자만 수정하거나 삭제할 수 있습니다.'}<br>
<br>

다른 사용자의 토큰으로 게시물을 수정하면, 인증은 성공했지만 `IsAuthorOrReadOnly` 권한 확인에 실패하므로 `403 Forbidden`과 함께 `message`에 작성한 문구가 반환됩니다.

작성자 본인의 토큰으로 요청한다면 `200 OK`와 함께 정상적으로 수정됩니다.

- Tip : 토큰은 비밀번호와 같은 역할을 하므로, 운영 환경에서는 반드시 `HTTPS`를 사용해 전송해야 합니다.

<br>

### 테스트 코드

{% highlight python %}

## blog/tests.py
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase

from blog.models import Post


class PostPermissionTest(APITestCase):
    def setUp(self):
        self.author = User.objects.create_user("daehee", password="pw-12345678")
        self.other = User.objects.create_user("other", password="pw-12345678")
        self.data = {"title": "제목", "contents": "내용", "post": {"contents": "댓글"}}

    def test_anonymous_cannot_create(self):
        response = self.client.post("/post/", self.data, format="json")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_only_author_can_update(self):
        self.client.force_authenticate(user=self.author)
        self.client.post("/post/", self.data, format="json")
        post = Post.objects.get()

        self.client.force_authenticate(user=self.other)
        response = self.client.patch(f"/post/{post.id}/", {"title": "변경"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        self.client.force_authenticate(user=self.author)
        response = self.client.patch(f"/post/{post.id}/", {"title": "변경"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

{% endhighlight %}

10강의 `APITestCase`로 권한 동작을 테스트할 수 있습니다.

`setUp()` 메서드는 각 테스트 메서드가 실행되기 전에 호출되어 테스트에 필요한 데이터를 준비합니다.

`force_authenticate(user=...)`는 토큰을 발급하지 않고도 지정한 사용자로 **인증된 상태**를 만들어, 권한 로직만 간단하게 확인할 수 있습니다.
