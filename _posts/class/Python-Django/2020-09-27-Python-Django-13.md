---
layout: post
title: "Python Django 강좌 : 제 13강 - Transaction"
tagline: "Python Django Transaction"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Django']
keywords: Python, Python Django, Python Django 3, Python Django Transaction
ref: Python-Django
category: posts
permalink: /posts/Python-Django-13/
comments: true
---

## Django Transaction ##
----------

`트랜잭션(Transaction)`이란 **데이터베이스의 상태를 변환**시키려는 작업의 단위를 의미합니다.

트랜잭션의 목적으로는 **데이터베이스 완전성(integrity) 유지**에 있습니다.

또한, 트랜잭션은 아래와 같은 네 가지의 성질을 갖고 있습니다.

- `Atomicity(원자성)`
    - 트랜잭션 연산은 데이터베이스에 모두 반영되거나 모두 반영되지 않아야 합니다.
    - 트랜잭션 내의 모든 명령은 완벽히 수행되어야 하며, 하나라도 오류가 발생한다면 모두 트랜잭션 전부가 취소되어야 합니다.

- `Consistency(일관성)`
    - 트랜잭션이 실행을 성공적으로 완료한다면 일관성 있는 데이터베이스 상태로 변환합니다.
    - 트랜잭션 수행 전과 수행 후의 상태가 같아야 합니다.

- `Isolation(독립성)`
  - 둘 이상의 트랜잭션이 동시에 실행되는 경우, 어느 하나의 트랜잭션 실행 중에 다른 트랜잭션의 연산이 끼어들 수 없습니다.
  - 수행 중인 트랜잭션은 완전히 종료될 때까지 다른 트랜잭션에서 수행 결과를 참조할 수 없습니다.

- `Durablility(지속성)`
  - 성공적으로 완료된 트랜잭션의 결과는 시스템이 고장나더라도 영구적으로 반영되어야 합니다.


즉, 트랜잭션 내의 작업은 모든 작업이 정상적으로 완료되면 데이터베이스에 **반영(commit)**하거나, 일부 작업이 실패한다면 실행 전으로 **되돌려야(rollback)**합니다.

앞선 12강의 외래키 예제로 설명한다면, `post` 테이블과 `comment` 테이블에 한 번에 데이터를 입력했습니다.

여기서, `post` 테이블에는 정상적으로 값이 등록됬지만, `comment` 테이블에 데이터를 저장할 때 오류가 발생했다면 `post` 테이블에만 값이 저장됩니다.

이는 의도한 바가 아니기 때문에 되돌리는 작업이 필요합니다.

여기서 오류가 발생했다고 `post` 테이블의 데이터를 삭제하는 것이 아닌, `롤백(rollback)`을 실행합니다.

이를 위해 사용하는 것이 트랜잭션입니다.

<br>

![1]({{ site.images }}/assets/images/Python/django/ch13/1.png)

앞선 예제를 변경하지 않고, 위와 같이 `post` 필드 내부의 값이 문제가 있다면 오류를 반환하지만 데이터베이스에서는 `Post` 데이터와 관련된 데이터는 저장됩니다.

`post` URL로 이동해서 결과를 확인해본다면, 아래와 같이 결과가 저장된 것을 확인할 수 있습니다.

<br>

![2]({{ site.images }}/assets/images/Python/django/ch13/2.png)

오류가 발생했지만, `post` 테이블에는 값이 저장된 것을 확인할 수 있습니다.

앞선 코드를 통해 확인해본다면 왜 이런 현상이 발생했는지 알 수 있습니다.

### views.py

{% highlight python %}

def create(self, request, *args, **kwargs):
    post_data = {
        "title": request.data["title"],
        "contents": request.data["contents"],
    }
    post_serializer = PostSerializer(data=post_data)
    if post_serializer.is_valid(raise_exception=True):
        post_result = post_serializer.save()

        comment_data = {
            "post_id": post_result.id,
            "contents": request.data["post"]["contents"],
        }
        comment_serializer = CommentSerializer(data=comment_data)
        if comment_serializer.is_valid(raise_exception=True):
            comment_serializer.save()
            return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)

    return Response({"message": "Error"}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

코드 상에서 `request.data`를 파싱하여 `PostSerializer`에 대해서만 먼저 유효성을 검사합니다.

유효하다면, `post_serializer.save()`을 통해 데이터베이스에 값을 저장합니다.

이후, `comment_data`에서 `contents` 필드에 `request.data["post"]["contents"]`를 가져오지만, 요청 데이터는 `contents` 필드 대신, `error`로 전혀 다른 필드가 작성되어 있습니다.

이 때, 오류가 발생하여 코드가 중단됩니다.

별도의 예외처리를 통해 오류 구문 없이 결과를 출력할 수는 있지만, 이미 `post` 테이블에 저장된 값은 되돌릴 수 없습니다.

이 때, 트랜잭션을 추가해 해결할 수 있습니다.

<br>
<br>

## Django Transaction 구현 ##
----------


### views.py

{% highlight python %}

from django.db import transaction

from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response

from blog.models import Post
from blog.models import Comment
from blog.serializers import PostSerializer
from blog.serializers import CommentSerializer


class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

    @transaction.atomic(using='default')
    def create(self, request, *args, **kwargs):
        try:
            with transaction.atomic():
                post_data = {
                    "title": request.data["title"],
                    "contents": request.data["contents"],
                }
                post_serializer = PostSerializer(data=post_data)
                if post_serializer.is_valid(raise_exception=True):
                    post_result = post_serializer.save()

                    comment_data = {
                        "post_id": post_result.id,
                        "contents": request.data["post"]["contents"],
                    }
                    comment_serializer = CommentSerializer(data=comment_data)
                    if comment_serializer.is_valid(raise_exception=True):
                        comment_serializer.save()
                        return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)
        except:
            pass

        return Response({"message": "Error"}, status=status.HTTP_400_BAD_REQUEST)


class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

{% endhighlight %}

먼저, `from django.db import transaction` 모듈을 추가해 트랜잭션 기능을 불러옵니다.

트랜잭션을 적용할 메서드 위에 트랜잭션 `데코레이터(Decorator)`를 추가합니다.

예제와 같이 `def create()`위에 `@transaction.atomic(using='default')` 구문을 추가합니다.

다음으로 `try-except`를 통해 어떤 오류가 발생하더라도 마지막 `Response`으로 전달되도록 구성합니다.

트랜잭션이 적용되야할 구문에는 `with transaction.atomic():`의 구문을 추가해 트랜잭션이 적용될 블록을 구성합니다.

여기서 주의해야할 점은 크게 두 가지가 있습니다.

### try-except 사용

트랜잭션 코드는 `with` 구문안에서 오류가 발생했을 때, `롤백(rollback)`을 적용합니다.

그러므로, `with` 구문안에 `try-except`를 적용한다면 정상적인 과정으로 인식하여 롤백되지 않습니다.

### raise_exception 사용

`try-except`와 마찬가지로, `serializer.is_valid`의 **raise_exception** 매개변수의 인수는 `True`로 사용해야합니다.

유효성을 검사할 때 `raise_exception`이 `False`라면 오류를 발생시키지 않아, 정상적인 과정으로 인식하여 롤백되지 않습니다.

<br>

간단하게 설명하자면, `with transaction.atomic():` 구문은 예외가 발생했을 때, 롤백하는 과정만 가지고 있습니다.

롤백하는 과정만 가지고 있기 때문에, `try-except`와는 다른 역할입니다.

즉, `request.data["post"]["contents"]`를 불러올 때 `contents` 필드가 없어서 발생하는 오류는 `try-except`를 통해 별도로 잡아주어야 합니다.

그렇기 때문에 가장 상단에 `try-except`를 추가하여, `return Response({"message": "Error"}, status=status.HTTP_400_BAD_REQUEST)`으로 이동합니다.

<br>
<br>

![3]({{ site.images }}/assets/images/Python/django/ch13/3.png)

![4]({{ site.images }}/assets/images/Python/django/ch13/4.png)

트랜잭션을 적용한 다음, `POST` 요청을 해보도록 하겠습니다.

`try-except`을 통하여 오류가 발생했을 때, 정상적인 오류 메세지가 전달되며, 데이터베이스에도 값이 저장되지 않는 것을 확인할 수 있습니다.

복합적으로 데이터베이스에 값을 저장한다면, 트랜잭션 기능을 적용하거나 별도의 함수를 생성하여 오류를 처리해야 합니다.

만약, 트랜잭션이나 별도의 함수를 구현하지 않는다면 **데이터베이스에 저장된 값을 신뢰할 수 없는 상태**가 됩니다.