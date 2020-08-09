---
layout: post
title: "Python Django 강좌 : 제 6강 - View"
tagline: "Python Django ViewSet"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Django']
keywords: Python, Python Django, Python Django 3, Python Django View, Python Django views.py, Python Django ViewSet, Python Django queryset, Python Django serializer_class, Python Django create, Python Django list, Python Django retrieve, Python Django update, Python Django destroy, Python Django CRUD, Python Django POST, Python Django GET, Python Django PUT, Python Django DELTE, Python Django exceptions, Python Django Response
ref: Python
category: posts
permalink: /posts/Python-Django-6/
comments: true
---

## Django View ##
----------

장고에서 뷰는 **어떤 데이터**를 표시할지 정의하며, `HTTP 응답 상태 코드(response)`를 반환합니다.

`views.py` 파일에 애플리케이션의 처리 논리를 정의합니다.

사용자가 입력한 URL에 따라, `모델(Model)`에서 필요한 데이터를 가져와 뷰에서 가공해 보여주며, `템플릿(Template)`에 전달하는 역할을 합니다.

장고의 뷰 파일(views.py)은 요청에 따른 **처리 논리**를 정의합니다.

즉, 사용자가 요청하는 `값(request)`을 받아 모델과 템플릿을 중개하는 역할을 합니다.

<br>

## views.py ##
----------

{% highlight python %}

from django.core.exceptions import *
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
        else:
            return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def list(self, request):
        # queryset = UserModel.objects.all()
        # serializer = UserSerializer(queryset, many=True)
        serializer = UserSerializer(self.get_queryset(), many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def retrieve(self, request, uuid=None):
        try:
            objects = UserModel.objects.get(id=uuid)
            serializer = UserSerializer(objects)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ObjectDoesNotExist:
            return Response({"message": "존재하지 않는 UUID({})".format(uuid)}, status=status.HTTP_404_NOT_FOUND)

    def update(self, request, uuid=None):
        objects = UserModel.objects.get(id=uuid)
        serializer = UserSerializer(objects, data=request.data)
        if serializer.is_valid(raise_exception=False):
            serializer.save()
            return Response({"message": "Operate successfully"}, status=status.HTTP_200_OK)
        else:
            return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, uuid=None):
        objects = UserModel.objects.get(id=uuid)
        objects.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

{% endhighlight %}

뷰(views.py) 파일을 위와 같이 정의합니다.

DRF에서는 `ViewSet`이라는 추상 클래스를 제공합니다.

`ViewSet` 클래스는 `get()`이나 `post()` 같은 메서드를 제공하지는 않지만, `create()`나 `list()` 같은 메서드를 지원합니다.

`ViewSet` 클래스는 `URL` 설정을 통해서 간단하게 연결할 수 있습니다.

<br>
<br>

## Module ##
----------

{% highlight python %}

from django.core.exceptions import *
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from first_app.models import UserModel
from first_app.serializers import UserSerializer

{% endhighlight %}

`from django.core.exceptions import *`은 장고에서 사용하는 예외 사항을 가져옵니다. `와일드카드(*)`를 사용해서 모든 예외 사항을 등록합니다.

`from rest_framework import status`은 **HTTP 상태 코드(status)**를 등록합니다. `1xx(조건부 응답)`이나 `2xx(성공)` 등을 반환할 수 있습니다.

`from rest_framework import viewsets`은 views.py에서 사용할 뷰 클래스입니다.

`from rest_framework.response import Response`는 응답에 사용할 **TemplateResponse** 형식의 객체입니다. 이를 통해 클라이언트에게 제공할 콘텐츠 형태로 변환합니다.

`from first_app.models import UserModel`은 `models.py`에서 선언한 UserModel 모델입니다.

`from first_app.serializers import UserSerializer`은 `serializers.py`에서 선언한 UserSerializer 직렬화 클래스입니다.

**serializers.py는 아직 선언하지 않았으며, 다음 강좌에서 자세히 다룹니다.**

<br>
<br>

## ModelViewSet ##
----------

{% highlight python %}

# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer
    
    def get_queryset(self):
        return UserModel.objects.all()

{% endhighlight %}

`UserViewSet`의 이름으로 뷰셋 클래스를 생성하고, `ModelViewSet`을 상속받아 사용합니다.

`ModelViewSet` 클래스는 `mixin` 클래스를 사용하며, `create()`, `update()`, `list()` 등의 **읽기/쓰기** 기능을 모두 제공합니다.

`ModelViewSet` 클래스를 사용하게 되면, `queryset` 및 `serializer_class`를 사용해야 합니다.

`queryset`은 테이블의 모든 데이터를 가져오게 하며, `serializer_class`는 추후에 선언할 `UserSerializer`를 연결합니다.

`get_queryset` 메서드는 `queryset`을 선언하지 않았을 때 사용합니다.

현재 코드에서는 필수요소는 아니지만, 예제를 위해 사용합니다.

`get_queryset`을 선언했을 때, 동일하게 테이블의 모든 데이터를 가져오게합니다.

<br>
<br>

## create(POST) ##
----------

{% highlight python %}

def create(self, request, *args, **kwargs):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid(raise_exception=False):
        serializer.save()
        return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)
    else:
        return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

`create()` 메서드는 `POST`와 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **쓰기** 기능을 의미합니다.

`serializer`에 **요청 데이터(request.data)**를 입력해 데이터를 **직렬화** 시킬 수 있도록 전달합니다.

그 후, `serializer.is_valid(raise_exception=False)`를 통해 직렬화된 데이터의 **유효성을 검사**합니다.

`raise_exception=False`을 통해, 오류를 발생시키지 않습니다.

오류를 발생시키지 않는 이유는 `Response`를 통해 **별도의 포맷**으로 제공하기 위함입니다.

유효성 검사에 통과한다면 `참(True)` 값을 반환하며, 통과하지 못한다면 `거짓(False)` 값을 반환합니다.

유효성 검사에 통과한 경우에는 `serializer.save()`을 통해 데이터를 데이터베이스 테이블에 저장합니다.

이 후, 각각 `Response()`메서드를 통해 **메세지** 및 `HTTP 상태 코드(status)`를 반환합니다.

<br>

* Tip : POST 메서드는 새로운 리소스를 **생성(create)**할 때 사용됩니다.
* Tip : `status.HTTP_201_CREATED`는 요청이 성공적으로 처리되었으며, 자원이 생성되었음을 나타내는 성공 상태 응답 코드입니다.
* Tip : `HTTP_400_BAD_REQUEST`는 잘못된 문법이나 제한 조건으로 인하여 서버가 요청을 이해할 수 없음을 의미하는 상태 응답 코드입니다.


## list(GET) ##
----------

{% highlight python %}

def list(self, request):
    # queryset = UserModel.objects.all()
    # serializer = UserSerializer(queryset, many=True)
    serializer = UserSerializer(self.get_queryset(), many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

{% endhighlight %}

`list()` 메서드는 `GET`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **읽기** 기능을 의미합니다.

주석 처리된 `queryset`과 `serializer`로도 데이터베이스의 테이블에서 값을 읽을 수도 있습니다.

`queryset`에서 테이블에서 값을 가져와, `UserSerializer`에 값을 제공합니다.

`many=True`일 경우 **여러 개의 데이터를 입력**할 수 있습니다.

이 후, `Response()`메서드를 통해 직렬화된 데이터를 전달합니다.

<br>

* Tip : `GET 메서드`는 **읽거나(Read)** **검색(Retrieve)**할 때에 사용되는 메서드입니다.
* Tip : `status.HTTP_200_OK`는 요청이 성공했음을 나타내는 성공 응답 상태 코드입니다.

<br>
<br>

## retrieve(GET) ##
----------

{% highlight python %}

def retrieve(self, request, uuid=None):
    try:
        objects = UserModel.objects.get(id=uuid)
        serializer = UserSerializer(objects)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except ObjectDoesNotExist:
        return Response({"message": "존재하지 않는 UUID({})".format(uuid)}, status=status.HTTP_404_NOT_FOUND)


{% endhighlight %}

`retrieve()` 메서드는 `GET`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **읽기(검색)** 기능을 의미합니다.

`uuid` 매개변수는 `urls.py`에서 입력된 변수명을 의미합니다. `serializers.py`와 마찬가지로 아직 작성하지 않았습니다.

`try` 구문 안의 코드로도 읽기 기능을 작성할 수 있지만, 올바르지 않은 값을 검색했을 때 예외 처리를 위해 `except`를 추가합니다.

`UserModel.objects.get(id=uuid)`는 `UserModel`의 `객체(objects)`에서 **하나의 열(Row)**을 가져오기 위해 `get()` 메서드를 사용합니다.

가져올 키 값을 `id`로 설정하고, 입력받은 `uuid`를 대입합니다.

그럴 경우, `id` 필드에서 `uuid`와 일치하는 열이 `objects` 변수에 저장됩니다.

성공적으로 값을 불러왔을 때, `serializer.data`와 `200 OK`로 응답합니다.

검색한 `id`의 값이 존재하지 않는 경우 `ObjectDoesNotExist` 오류가 발생하므로, `except` 구문에서 오류 메세지를 반환합니다.

<br>

* Tip : `retrieve()` 메서드의 `uuid` 매개변수는 `urls.py`의 설정에 따라 달라질 수 있습니다.
* Tip : `UserModel.objects.get(id=uuid)`의 `id`는 `models.py`의 `UserModel`에서 선언한 **id = models.UUIDField()**의 `id` 필드(변수)를 의미합니다.

<br>
<br>

## update(PUT) ##
----------

{% highlight python %}

def update(self, request, uuid=None):
    objects = UserModel.objects.get(id=uuid)
    serializer = UserSerializer(objects, data=request.data)
    if serializer.is_valid(raise_exception=False):
        serializer.save()
        return Response({"message": "Operate successfully"}, status=status.HTTP_200_OK)
    else:
        return Response({"message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

`update()` 메서드는 `PUT`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블에 **수정** 기능을 의미합니다.

검색 기능과 마찬가지로 특정 uuid를 통해 검색된 **열(Rows)**의 값을 수정합니다.

수정할 열의 데이터를 검색 기능처럼 가져오며, 쓰기 기능처럼 요청 받은 값을 전달합니다.

차이점은 새로 생성하는 것이 아니므로, 검색된 **열(objects)**과 요청 받은 **값(request.data)**을 `UserSerializer에` 같이 전달합니다.

이후, 쓰기 기능처럼 **유효성을 검사하고 결과에 맞게 응답합니다.**

<br>

* Tip : `update()` 메서드의 `uuid` 매개변수는 `urls.py`의 설정에 따라 달라질 수 있습니다.

<br>
<br>

## destroy(DELETE) ##
----------

{% highlight python %}

def destroy(self, request, uuid=None):
    objects = UserModel.objects.get(id=uuid)
    objects.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)

{% endhighlight %}

`destroy()` 메서드는 `DELETE`과 관련된 메서드로 볼 수 있습니다.

데이터베이스 테이블의 **삭제** 기능을 의미합니다.

동일하게 삭제할 `id`를 검색하고, `delete()` 메서드로 열을 지웁니다.

삭제 처리가 완료됐다면, 응답 상태를 반환합니다.

<br>

* Tip : `destroy()` 메서드의 `uuid` 매개변수는 `urls.py`의 설정에 따라 달라질 수 있습니다.
* Tip : `status.HTTP_204_NO_CONTENT`는 요청이 성공했으나 클라이언트가 현재 페이지에서 벗어나지 않아도 된다는 것을 의미하는 상태 응답 코드입니다.

<br>
<br>

## 상속하지 않고 사용하기 ##
----------

{% highlight python %}

# def list(self, request):
#     # queryset = UserModel.objects.all()
#     # serializer = UserSerializer(queryset, many=True)
#     serializer = UserSerializer(self.get_queryset(), many=True)
#     return Response(serializer.data, status=status.HTTP_200_OK)

{% endhighlight %}

현재 클래스에서 선언된 `create()`, `list()` 메서드 등은 `mixins` 클래스에서 상속된 메서드입니다.

그러므로, 선언하지 않아도 기본적으로 선언된 메서드들을 사용할 수 있습니다.

만약, `list()` 메서드를 일괄 주석처리하더라도 `ListModelMixin`가 적용됩니다.

`ListModelMixin` 클래스는 UserViewSet 아래에 선언한 `queryset` 또는 `get_queryset`을 불러와 사용합니다.

그러므로, `queryset` 또는 `get_queryset`을 선언해야합니다.