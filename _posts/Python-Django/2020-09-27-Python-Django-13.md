---
layout: post
title: "Python Django 강좌 : 제 13강 - Transaction"
tagline: "Python Django Transaction"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Transaction, Python Django transaction.atomic, Python Django ATOMIC_REQUESTS, Python Django rollback, Python Django ACID
ref: Python-Django
category: Python
permalink: /posts/Python-Django-13/
comments: true
toc: true
---

## Django Transaction

`트랜잭션(Transaction)`이란 **데이터베이스의 상태를 변환**시키려는 하나의 논리적인 작업 단위를 의미합니다.

트랜잭션의 목적은 **데이터베이스 무결성(integrity) 유지**에 있습니다.

또한, 트랜잭션은 아래와 같은 네 가지의 성질을 갖고 있으며, 이를 `ACID`라고 합니다.

- `Atomicity(원자성)`
    - 트랜잭션 연산은 데이터베이스에 모두 반영되거나 모두 반영되지 않아야 합니다.
    - 트랜잭션 내의 모든 명령은 완벽히 수행되어야 하며, 하나라도 오류가 발생한다면 트랜잭션 전부가 취소되어야 합니다.

- `Consistency(일관성)`
    - 트랜잭션이 실행을 성공적으로 완료한다면 데이터베이스는 항상 일관성 있는 상태를 유지해야 합니다.
    - 트랜잭션 수행 전과 수행 후 모두 데이터베이스의 제약 조건(외래키, 고유 키 등)과 규칙을 만족해야 합니다.

- `Isolation(독립성, 격리성)`
    - 둘 이상의 트랜잭션이 동시에 실행되는 경우, 어느 하나의 트랜잭션 실행 중에 다른 트랜잭션의 연산이 끼어들 수 없습니다.
    - 수행 중인 트랜잭션은 완전히 종료될 때까지 다른 트랜잭션에서 수행 결과를 참조할 수 없습니다.
    - 실제 격리 수준은 데이터베이스 설정(Read Committed, Repeatable Read 등)에 따라 달라집니다.

- `Durability(지속성)`
    - 성공적으로 완료된 트랜잭션의 결과는 시스템이 고장나더라도 영구적으로 반영되어야 합니다.

<br>

즉, 트랜잭션 내의 작업은 모든 작업이 정상적으로 완료되면 데이터베이스에 **반영(commit)**하거나, 일부 작업이 실패한다면 실행 전으로 **되돌려야(rollback)**합니다.

앞선 12강의 외래키 예제로 설명한다면, `post` 테이블과 `comment` 테이블에 한 번에 데이터를 입력했습니다.

여기서, `post` 테이블에는 정상적으로 값이 등록됐지만, `comment` 테이블에 데이터를 저장할 때 오류가 발생했다면 `post` 테이블에만 값이 저장됩니다.

이는 의도한 바가 아니기 때문에 되돌리는 작업이 필요합니다.

여기서 오류가 발생했다고 `post` 테이블의 데이터를 직접 삭제하는 것이 아닌, `롤백(rollback)`을 실행합니다.

이를 위해 사용하는 것이 트랜잭션입니다.

<br>

{% highlight json %}

{
    "title" : "제 4강 - Django 애플리케이션 생성",
    "contents" : "장고에서 ...",
    "post" : {
        "error" : "트랜잭션 오류 테스트"
    }
}

{% endhighlight %}

**결과**
:    
400 Bad Request<br>
{"contents": ["This field may not be null."]}<br>
<br>

앞선 예제를 변경하지 않고, 위와 같이 `post` 필드 내부에 `contents` 대신 `error`라는 잘못된 키를 전달해 보겠습니다.

댓글의 유효성 검사에 실패해 오류를 반환하지만, 데이터베이스에는 `Post` 데이터가 저장됩니다.

`post/` URL로 이동해서 결과를 확인해본다면, 아래와 같이 결과가 저장된 것을 확인할 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-13/2.webp" class="lazyload" width="100%" height="100%"/>

오류가 발생했지만, `post` 테이블에는 값이 저장된 것을 확인할 수 있습니다.

앞선 코드를 통해 확인해본다면 왜 이런 현상이 발생했는지 알 수 있습니다.

<br>

### views.py

{% highlight python %}

def create(self, request, *args, **kwargs):
    post_data = {
        "title": request.data.get("title"),
        "contents": request.data.get("contents"),
    }
    post_serializer = PostSerializer(data=post_data)
    post_serializer.is_valid(raise_exception=True)
    post_result = post_serializer.save()

    comment_data = {
        "post_id": post_result.id,
        "contents": request.data.get("post", {}).get("contents"),
    }
    comment_serializer = CommentSerializer(data=comment_data)
    comment_serializer.is_valid(raise_exception=True)
    comment_serializer.save()

    return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)

{% endhighlight %}

코드 상에서 `request.data`를 나누어 `PostSerializer`에 대해서만 먼저 유효성을 검사합니다.

유효하다면, `post_serializer.save()`를 통해 데이터베이스에 값을 저장합니다.

장고는 기본적으로 **자동 커밋(autocommit)** 모드로 동작하므로, `save()`가 실행되는 순간 쿼리가 바로 데이터베이스에 반영됩니다.

이후, `comment_data`에서 `contents` 필드에 `request.data.get("post", {}).get("contents")`를 가져오지만, 요청 데이터는 `contents` 필드 대신, `error`로 전혀 다른 필드가 작성되어 있습니다.

그러므로 `contents`의 값은 `None`이 되고, `comment_serializer.is_valid(raise_exception=True)`에서 `ValidationError`가 발생하여 코드가 중단됩니다.

하지만, 이미 `post` 테이블에 저장된 값은 되돌릴 수 없습니다.

이때, 트랜잭션을 추가해 해결할 수 있습니다.

<br>
<br>

## Django Transaction 구현

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
    queryset = Post.objects.prefetch_related("post")
    serializer_class = PostSerializer

    def create(self, request, *args, **kwargs):
        with transaction.atomic():
            post_data = {
                "title": request.data.get("title"),
                "contents": request.data.get("contents"),
            }
            post_serializer = PostSerializer(data=post_data)
            post_serializer.is_valid(raise_exception=True)
            post_result = post_serializer.save()

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

먼저, `from django.db import transaction` 모듈을 추가해 트랜잭션 기능을 불러옵니다.

트랜잭션이 적용되어야 할 구문을 `with transaction.atomic():` 블록 안에 작성합니다.

`with` 블록에 진입하면 트랜잭션이 시작되며, 블록이 **정상적으로 끝나면 커밋(commit)**하고, 블록 안에서 **예외가 발생해 블록을 빠져나가면 롤백(rollback)**합니다.

예제에서 댓글의 유효성 검사에 실패하면 `ValidationError`가 발생해 `with` 블록을 빠져나가므로, 앞서 저장한 `Post`도 함께 롤백됩니다.

이후 `ValidationError`는 DRF의 예외 처리기가 받아 **400 Bad Request**와 오류 내용을 반환합니다.

<br>

{% highlight python %}

@transaction.atomic
def create(self, request, *args, **kwargs):
    ...

{% endhighlight %}

메서드 전체를 트랜잭션으로 묶는다면 `@transaction.atomic` `데코레이터(Decorator)`를 사용할 수도 있습니다.

`with` 구문과 데코레이터는 같은 역할을 하므로, **둘 중 하나만** 사용합니다.

- Tip : 여러 데이터베이스를 사용하는 경우 `transaction.atomic(using="default")`처럼 트랜잭션을 적용할 데이터베이스의 별칭을 지정할 수 있습니다.

<br>

여기서 주의해야할 점은 크게 두 가지가 있습니다.

<br>

### try-except 사용

`transaction.atomic()`은 **블록 밖으로 전달되는 예외**를 감지해 롤백합니다.

그러므로, `with` 블록 **안에서** `try-except`로 예외를 잡아버린다면 정상적인 과정으로 인식하여 롤백되지 않습니다.

{% highlight python %}

# 잘못된 예시 : 예외를 블록 안에서 처리하므로 롤백되지 않음
with transaction.atomic():
    try:
        post_serializer.save()
        comment_serializer.save()
    except Exception:
        pass

# 올바른 예시 : 블록 밖에서 예외를 처리하므로 롤백된 후 except가 실행됨
try:
    with transaction.atomic():
        post_serializer.save()
        comment_serializer.save()
except IntegrityError:
    return Response({"message": "Error"}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

별도의 오류 응답이 필요하다면, 예제처럼 `try-except`를 `with` 블록 **바깥**에 작성합니다.

또한, `except:`나 `except Exception:`처럼 모든 예외를 잡으면 실제 버그까지 숨겨지므로, 처리할 예외만 지정하는 것이 좋습니다.

<br>

### raise_exception 사용

`try-except`와 마찬가지로, `serializer.is_valid`의 **raise_exception** 매개변수의 인수는 `True`로 사용해야합니다.

유효성을 검사할 때 `raise_exception`이 `False`라면 예외 없이 `False`만 반환하므로, 이후 코드가 그대로 진행되거나 정상적으로 함수가 종료되어 롤백되지 않습니다.

`raise_exception=False`를 사용한다면 실패 시 직접 `transaction.set_rollback(True)`를 호출하거나 예외를 발생시켜야 합니다.

<br>

간단하게 설명하자면, `with transaction.atomic():` 구문은 **예외가 발생했을 때 롤백하는 역할**만 가지고 있으며, 예외를 처리해 응답을 만드는 역할은 하지 않습니다.

예외를 응답으로 바꾸는 것은 DRF의 예외 처리기나 블록 바깥의 `try-except`가 담당합니다.

<br>

{% highlight json %}

{
    "title" : "제 5강 - Model",
    "contents" : "장고에서 ...",
    "post" : {
        "error" : "트랜잭션 오류 테스트"
    }
}

{% endhighlight %}

**결과**
:    
400 Bad Request<br>
{"contents": ["This field may not be null."]}<br>
<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-13/4.webp" class="lazyload" width="100%" height="100%"/>

트랜잭션을 적용한 다음, 같은 형태로 `POST` 요청을 해보도록 하겠습니다.

오류가 발생했을 때 오류 메세지가 전달되며, `post/` URL을 확인해보면 **제 5강 게시물**이 데이터베이스에 저장되지 않은 것을 확인할 수 있습니다.

복합적으로 데이터베이스에 값을 저장한다면, 트랜잭션 기능을 적용하거나 별도의 함수를 생성하여 오류를 처리해야 합니다.

만약, 트랜잭션이나 별도의 함수를 구현하지 않는다면 **데이터베이스에 저장된 값을 신뢰할 수 없는 상태**가 됩니다.

<br>
<br>

## ATOMIC_REQUESTS

{% highlight python %}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'ATOMIC_REQUESTS': True,
    }
}

{% endhighlight %}

`settings.py`의 데이터베이스 설정에 `ATOMIC_REQUESTS`를 `True`로 지정하면, **모든 요청(뷰)**이 하나의 트랜잭션으로 실행됩니다.

뷰에서 예외가 발생하면 해당 요청에서 실행된 쿼리가 모두 롤백됩니다.

DRF는 예외를 응답으로 변환할 때 `ATOMIC_REQUESTS`가 켜져 있다면 자동으로 롤백을 표시하므로, `ValidationError` 등으로 오류 응답을 반환해도 롤백됩니다.

- Tip : `ATOMIC_REQUESTS`는 간편하지만, 조회만 하는 요청까지 모두 트랜잭션으로 감싸므로 트래픽이 많은 서비스에서는 성능에 영향을 줄 수 있습니다. 필요한 부분에만 `transaction.atomic()`을 사용하는 방법이 더 일반적입니다.

- Tip : 커밋이 완료된 뒤에 이메일 발송, 캐시 갱신 등을 실행해야 한다면 `transaction.on_commit(함수)`를 사용합니다. 롤백되면 등록한 함수는 실행되지 않습니다.
