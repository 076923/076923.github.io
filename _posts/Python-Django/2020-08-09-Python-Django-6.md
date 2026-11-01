---
layout: post
title: "Python Django 강좌 : 제 6강 - View"
tagline: "Python Django ViewSet"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django View, Python Django views.py, Python Django ViewSet, Python Django queryset, Python Django serializer_class, Python Django create, Python Django list, Python Django retrieve, Python Django update, Python Django partial_update, Python Django destroy, Python Django CRUD, Python Django POST, Python Django GET, Python Django PUT, Python Django PATCH, Python Django DELETE, Python Django DoesNotExist, Python Django Response
ref: Python-Django
category: Python
permalink: /posts/Python-Django-6/
comments: true
toc: true
---

## Django View

장고에서 뷰는 **어떤 데이터**를 표시할지 정의하며, `HTTP 응답(Response)`을 반환합니다.

`views.py` 파일에 애플리케이션의 처리 논리를 정의합니다.

사용자가 입력한 URL에 따라, `모델(Model)`에서 필요한 데이터를 가져와 뷰에서 가공해 보여주며, `템플릿(Template)`에 전달하는 역할을 합니다.

장고의 뷰 파일(views.py)은 요청에 따른 **처리 논리**를 정의합니다.

즉, 사용자의 `요청(request)`을 받아 모델과 템플릿(또는 직렬화 클래스)을 중개하는 역할을 합니다.

<br>
<br>

## views.py

{% highlight python %}

from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from first_app.models import UserModel
from first_app.serializers import UserSerializer


# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer

    def get_queryset(self):
        return UserModel.objects.all()

    def create(self, request, *args, **kwargs):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid(raise_exception=False):
            serializer.save()
            return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)
        return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def list(self, request, *args, **kwargs):
        serializer = UserSerializer(self.get_queryset(), many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def retrieve(self, request, uuid=None, *args, **kwargs):
        try:
            instance = UserModel.objects.get(id=uuid)
        except UserModel.DoesNotExist:
            return Response({"message": f"존재하지 않는 UUID({uuid})"}, status=status.HTTP_404_NOT_FOUND)

        serializer = UserSerializer(instance)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def update(self, request, uuid=None, *args, **kwargs):
        try:
            instance = UserModel.objects.get(id=uuid)
        except UserModel.DoesNotExist:
            return Response({"message": f"존재하지 않는 UUID({uuid})"}, status=status.HTTP_404_NOT_FOUND)

        partial = kwargs.pop("partial", False)
        serializer = UserSerializer(instance, data=request.data, partial=partial)
        if serializer.is_valid(raise_exception=False):
            serializer.save()
            return Response({"message": "Operate successfully"}, status=status.HTTP_200_OK)
        return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, uuid=None, *args, **kwargs):
        try:
            instance = UserModel.objects.get(id=uuid)
        except UserModel.DoesNotExist:
            return Response({"message": f"존재하지 않는 UUID({uuid})"}, status=status.HTTP_404_NOT_FOUND)

        instance.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

{% endhighlight %}

뷰(views.py) 파일을 위와 같이 정의합니다.

DRF에서는 `ViewSet`이라는 클래스를 제공합니다.

`ViewSet` 클래스는 `get()`이나 `post()` 같은 **HTTP 메서드 이름**의 메서드 대신, `create()`나 `list()` 같은 **동작(action) 이름**의 메서드를 사용합니다.

`ViewSet` 클래스는 `URL` 설정에서 HTTP 메서드와 동작을 연결해 간단하게 사용할 수 있습니다.

<br>

### Module

{% highlight python %}

from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from first_app.models import UserModel
from first_app.serializers import UserSerializer

{% endhighlight %}

`from rest_framework import status`는 **HTTP 상태 코드(status)** 상수를 가져옵니다. `2xx(성공)`, `4xx(클라이언트 오류)` 등을 이름으로 사용할 수 있습니다.

`from rest_framework import viewsets`는 views.py에서 사용할 뷰셋 클래스입니다.

`from rest_framework.response import Response`는 응답에 사용할 객체입니다. 전달한 데이터를 클라이언트가 요청한 형식(JSON, 웹 브라우저용 HTML 등)으로 변환해 반환합니다.

`from first_app.models import UserModel`은 `models.py`에서 선언한 UserModel 모델입니다.

`from first_app.serializers import UserSerializer`는 `serializers.py`에서 선언한 UserSerializer 직렬화 클래스입니다.

**serializers.py는 아직 선언하지 않았으며, 다음 강좌에서 자세히 다룹니다.**

- Tip : `존재하지 않는 객체` 예외는 각 모델의 `UserModel.DoesNotExist`로 처리합니다. `from django.core.exceptions import *`처럼 와일드카드(*)로 모든 이름을 가져오는 방식은 어떤 이름이 들어왔는지 알기 어려우므로 권장하지 않습니다.

<br>

### ModelViewSet

{% highlight python %}

# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer

    def get_queryset(self):
        return UserModel.objects.all()

{% endhighlight %}

`UserViewSet`의 이름으로 뷰셋 클래스를 생성하고, `ModelViewSet`을 상속받아 사용합니다.

`ModelViewSet` 클래스는 여러 `Mixin` 클래스를 조합해 `create()`, `list()`, `retrieve()`, `update()`, `partial_update()`, `destroy()`의 **읽기/쓰기** 기능을 모두 제공합니다.

`ModelViewSet` 클래스를 사용하게 되면, `queryset`(또는 `get_queryset()`)과 `serializer_class`를 지정해야 합니다.

`queryset`은 뷰에서 다룰 데이터의 기본 조회 조건이며, `serializer_class`는 추후에 선언할 `UserSerializer`를 연결합니다.

`get_queryset()` 메서드는 **요청마다** 조회 조건을 계산해야 할 때 재정의합니다. (예 : 로그인한 사용자의 데이터만 조회)

기본 구현은 `queryset` 속성을 반환하며, 현재 코드에서는 필수요소는 아니지만 예제를 위해 사용합니다.

- Tip : `queryset` 속성은 클래스가 정의될 때 한 번만 만들어지지만, 실제 SQL은 요청마다 새로 실행되므로 최신 데이터를 가져옵니다.

<br>
<br>

## create(POST)

{% highlight python %}

def create(self, request, *args, **kwargs):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid(raise_exception=False):
        serializer.save()
        return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)
    return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

`create()` 메서드는 `POST`와 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **쓰기** 기능을 의미합니다.

`serializer`에 **요청 데이터(request.data)**를 입력해 데이터를 **역직렬화** 할 수 있도록 전달합니다.

그 후, `serializer.is_valid(raise_exception=False)`를 통해 입력된 데이터의 **유효성을 검사**합니다.

`raise_exception=False`를 통해, 유효성 검사에 실패해도 예외를 발생시키지 않습니다.

예외를 발생시키지 않는 이유는 `Response`를 통해 **별도의 포맷**으로 오류를 제공하기 위함입니다.

유효성 검사에 통과한다면 `참(True)` 값을 반환하며, 통과하지 못한다면 `거짓(False)` 값을 반환하고 오류 내용을 `serializer.errors`에 저장합니다.

유효성 검사에 통과한 경우에는 `serializer.save()`를 통해 데이터를 데이터베이스 테이블에 저장합니다.

이 후, 각각 `Response()`를 통해 **메세지** 및 `HTTP 상태 코드(status)`를 반환합니다.

- Tip : POST 메서드는 새로운 리소스를 **생성(create)**할 때 사용됩니다.

- Tip : `status.HTTP_201_CREATED`는 요청이 성공적으로 처리되었으며, 자원이 생성되었음을 나타내는 성공 상태 응답 코드입니다.

- Tip : `status.HTTP_400_BAD_REQUEST`는 잘못된 문법이나 제한 조건으로 인하여 서버가 요청을 처리할 수 없음을 의미하는 상태 응답 코드입니다.

- Tip : `raise_exception=True`로 사용하면 유효성 검사 실패 시 `ValidationError`가 발생하고, DRF가 자동으로 `400 Bad Request` 응답을 반환합니다.

<br>
<br>

## list(GET)

{% highlight python %}

def list(self, request, *args, **kwargs):
    serializer = UserSerializer(self.get_queryset(), many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

{% endhighlight %}

`list()` 메서드는 `GET`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **읽기(목록 조회)** 기능을 의미합니다.

`self.get_queryset()`으로 테이블에서 값을 가져와, `UserSerializer`에 값을 제공합니다.

`many=True`일 경우 **여러 개의 객체(쿼리셋)**를 직렬화할 수 있습니다.

이 후, `Response()`를 통해 직렬화된 데이터를 전달합니다.

- Tip : `GET 메서드`는 리소스를 **읽거나(Read)** **검색(Retrieve)**할 때에 사용되는 메서드입니다.

- Tip : `status.HTTP_200_OK`는 요청이 성공했음을 나타내는 성공 응답 상태 코드입니다.

<br>
<br>

## retrieve(GET)

{% highlight python %}

def retrieve(self, request, uuid=None, *args, **kwargs):
    try:
        instance = UserModel.objects.get(id=uuid)
    except UserModel.DoesNotExist:
        return Response({"message": f"존재하지 않는 UUID({uuid})"}, status=status.HTTP_404_NOT_FOUND)

    serializer = UserSerializer(instance)
    return Response(serializer.data, status=status.HTTP_200_OK)

{% endhighlight %}

`retrieve()` 메서드는 `GET`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **읽기(단일 조회)** 기능을 의미합니다.

`uuid` 매개변수는 `urls.py`에서 URL 경로로 전달되는 변수명을 의미합니다. `serializers.py`와 마찬가지로 아직 작성하지 않았습니다.

`UserModel.objects.get(id=uuid)`는 `UserModel`의 `매니저(objects)`에서 **하나의 행(Row)**을 가져오기 위해 `get()` 메서드를 사용합니다.

가져올 키 값을 `id`로 설정하고, 입력받은 `uuid`를 대입합니다.

그럴 경우, `id` 필드에서 `uuid`와 일치하는 행이 `instance` 변수에 저장됩니다.

검색한 `id`의 값이 존재하지 않는 경우 `UserModel.DoesNotExist` 예외가 발생하므로, `except` 구문에서 `404 Not Found`와 오류 메세지를 반환합니다.

성공적으로 값을 불러왔을 때, `serializer.data`와 `200 OK`로 응답합니다.

- Tip : `retrieve()` 메서드의 `uuid` 매개변수는 `urls.py`의 설정에 따라 달라질 수 있습니다.

- Tip : `UserModel.objects.get(id=uuid)`의 `id`는 `models.py`의 `UserModel`에서 선언한 **id = models.UUIDField()**의 `id` 필드를 의미합니다.

- Tip : `get()`은 조건에 맞는 행이 없으면 `DoesNotExist`, **두 개 이상이면** `MultipleObjectsReturned` 예외가 발생합니다.

<br>
<br>

## update(PUT, PATCH)

{% highlight python %}

def update(self, request, uuid=None, *args, **kwargs):
    try:
        instance = UserModel.objects.get(id=uuid)
    except UserModel.DoesNotExist:
        return Response({"message": f"존재하지 않는 UUID({uuid})"}, status=status.HTTP_404_NOT_FOUND)

    partial = kwargs.pop("partial", False)
    serializer = UserSerializer(instance, data=request.data, partial=partial)
    if serializer.is_valid(raise_exception=False):
        serializer.save()
        return Response({"message": "Operate successfully"}, status=status.HTTP_200_OK)
    return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

`update()` 메서드는 `PUT`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **수정** 기능을 의미합니다.

검색 기능과 마찬가지로 특정 uuid를 통해 검색된 **행(Row)**의 값을 수정합니다.

수정할 행의 데이터를 검색 기능처럼 가져오며, 쓰기 기능처럼 요청 받은 값을 전달합니다.

차이점은 새로 생성하는 것이 아니므로, 검색된 **행(instance)**과 요청 받은 **값(request.data)**을 `UserSerializer`에 같이 전달합니다.

이후, 쓰기 기능처럼 **유효성을 검사하고 결과에 맞게 응답합니다.**

<br>

`PUT`은 리소스 **전체**를 교체하므로 모든 필수 필드를 전달해야 하며, `PATCH`는 리소스의 **일부**만 수정합니다.

`ModelViewSet`의 `partial_update()` 메서드(PATCH)는 `partial=True` 인수를 추가해 `update()` 메서드를 호출합니다.

그러므로, `update()` 메서드는 `partial` 인수를 받아 `UserSerializer(..., partial=partial)`로 전달해야 `PATCH` 요청도 정상적으로 처리할 수 있습니다.

- Tip : `update(self, request, uuid=None)`처럼 추가 인수를 받지 않도록 작성하면, `PATCH` 요청 시 `partial` 인수를 받을 수 없어 `TypeError`가 발생합니다. 재정의하는 메서드에는 `*args, **kwargs`를 함께 선언합니다.

<br>
<br>

## destroy(DELETE)

{% highlight python %}

def destroy(self, request, uuid=None, *args, **kwargs):
    try:
        instance = UserModel.objects.get(id=uuid)
    except UserModel.DoesNotExist:
        return Response({"message": f"존재하지 않는 UUID({uuid})"}, status=status.HTTP_404_NOT_FOUND)

    instance.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)

{% endhighlight %}

`destroy()` 메서드는 `DELETE`와 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **삭제** 기능을 의미합니다.

동일하게 삭제할 `id`를 검색하고, `delete()` 메서드로 행을 지웁니다.

삭제 처리가 완료됐다면, 응답 상태를 반환합니다.

- Tip : `destroy()` 메서드의 `uuid` 매개변수는 `urls.py`의 설정에 따라 달라질 수 있습니다.

- Tip : `status.HTTP_204_NO_CONTENT`는 요청이 성공했으며, 응답 본문(Body)에 보낼 내용이 없음을 의미하는 상태 응답 코드입니다.

<br>
<br>

## 상속하지 않고 사용하기

{% highlight python %}

class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer
    lookup_url_kwarg = "uuid"

{% endhighlight %}

현재 클래스에서 선언한 `create()`, `list()` 메서드 등은 `Mixin` 클래스에 이미 구현된 메서드를 **재정의(override)**한 것입니다.

그러므로, 선언하지 않아도 기본적으로 구현된 메서드들을 사용할 수 있습니다.

예를 들어, `list()` 메서드를 지우더라도 `ListModelMixin`의 `list()` 메서드가 적용됩니다.

`Mixin` 클래스는 `queryset` 또는 `get_queryset()`과 `serializer_class`를 불러와 사용하므로, 반드시 선언해야 합니다.

기본 구현은 URL의 `pk` 변수로 대상을 찾으므로, URL의 변수명을 `uuid`로 사용한다면 `lookup_url_kwarg = "uuid"`를 지정합니다.

- Tip : 기본 구현은 대상이 없을 때 자동으로 `404 Not Found`를 반환하고, 유효성 검사에 실패하면 `400 Bad Request`와 오류 내용을 반환합니다.

- Tip : 응답 형식을 바꾸는 등 **특별한 처리가 필요한 메서드만 재정의**하는 것이 일반적입니다.
