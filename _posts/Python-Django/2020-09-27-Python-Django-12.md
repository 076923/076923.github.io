---
layout: post
title: "Python Django 강좌 : 제 12강 - Foreign Key (2)"
tagline: "Python Django Foreign Key"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Foreign Key, Python Django Foreign Key insert, Python Django multiple insert, Python Django Nested Serializer create
ref: Python-Django
category: Python
permalink: /posts/Python-Django-12/
comments: true
toc: true
---

## Django Foreign Key

앞선 외래키 사용법에서는 두 개의 **URL(post, comment)**을 생성해 각각 요청하여 데이터를 입력하였습니다.

`post` 기능과 `comment` 기능은 별도의 기능이기 때문에 분리하였지만, 게시물에 종속된 **별도의 기능(태그, 키워드, 정보)**을 다른 테이블로 분리했다면 테이블이 다르더라도 함께 작성되는 것이 더 효율적이고 관리하기 편할 수도 있습니다.

이렇듯 하나의 요청으로 **부모 테이블**과 **자식 테이블**이 함께 작성되어야 하는 경우에는 `뷰(views.py)`의 코드를 수정해 적용할 수 있습니다.

<br>
<br>

## 요청 데이터

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-12/1.webp" class="lazyload" width="100%" height="100%"/>

**게시물(post) 데이터**와 **댓글(comment) 데이터**를 함께 받아서 `post` 테이블과 `comment` 테이블에 한 번에 입력해보도록 하겠습니다.

Django REST framework에서 지원되는 **HTML 입력 폼**으로는 중첩된 데이터를 입력할 수 없으므로, `포스트맨(Postman)`이나 `코드`를 통해 **JSON 형식**으로 데이터를 전달합니다.

Browsable API를 사용한다면 하단 입력 폼의 `Raw data` 탭에서 `application/json` 형식으로 작성할 수 있습니다.

<br>

{% highlight json %}

{
    "title" : "제 3강 - Django 프로젝트 설정",
    "contents" : "장고(Django)의 ...",
    "post" : {
        "contents" : "댓글까지 함께 작성."
    }
}

{% endhighlight %}

위와 같은 형태로 데이터를 `post/` URL에 전달합니다.

`post` 키 안에 작성할 댓글의 내용을 전달합니다.

<br>
<br>

## Django Code 구성

### views.py

{% highlight python %}

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

`create()` 메서드를 재정의해 요청 데이터를 각각의 테이블에 맞는 데이터로 나누고, 나눈 데이터마다 역직렬화합니다.

`request.data`는 앞선 **요청 데이터**와 동일한 형태를 갖습니다.

이 데이터를 나누어 `PostSerializer`와 `CommentSerializer`에 맞게끔 변형합니다.

<br>

{% highlight python %}

post_data = {
    "title": request.data.get("title"),
    "contents": request.data.get("contents"),
}
post_serializer = PostSerializer(data=post_data)
post_serializer.is_valid(raise_exception=True)
post_result = post_serializer.save()

{% endhighlight %}

`Comment` 테이블은 `Post` 테이블의 `id`를 외래키로 사용하기 때문에, 먼저 `Post` 테이블에 값을 저장해야 합니다.

그러므로, `PostSerializer`의 형식과 동일한 형태의 `post_data`를 생성합니다.

`request.data["title"]` 대신 `request.data.get("title")`을 사용하면, 값이 없을 때 `KeyError`가 발생하지 않고 `None`이 전달되어 **유효성 검사에서 오류를 반환**합니다.

`post_data`로 `PostSerializer`의 인스턴스인 `post_serializer`를 생성합니다.

여기서, `self.get_serializer(data=post_data)`를 사용해도 무방하지만 직관적으로 이해하기 위해 `PostSerializer`를 사용하도록 하겠습니다.

`is_valid(raise_exception=True)` 메서드로 유효성을 확인합니다. 유효하지 않다면 `ValidationError`가 발생하며, DRF가 자동으로 **400 Bad Request**와 오류 내용을 반환합니다.

유효하다면 `post_serializer.save()`를 통해 데이터베이스에 값을 저장합니다.

`post_result`에는 저장된 `Post` **모델 인스턴스**가 반환됩니다.

<br>

{% highlight python %}

comment_data = {
    "post_id": post_result.id,
    "contents": request.data.get("post", {}).get("contents"),
}
comment_serializer = CommentSerializer(data=comment_data)
comment_serializer.is_valid(raise_exception=True)
comment_serializer.save()

{% endhighlight %}

앞선 방식과 동일한 방식으로 한 번 더 반복합니다.

단, `comment` 테이블은 외래키인 `post_id` 값을 필요로 합니다.

그러므로, `post` 테이블에 값을 저장하면서 생성된 `id` 값을 가져와 `post_id` 필드의 값으로 사용합니다.

`request.data.get("post", {})`는 `post` 키가 없을 때 빈 사전을 사용해, 다음 `.get("contents")`에서 오류가 발생하지 않도록 합니다.

<br>

{% highlight python %}

return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)

{% endhighlight %}

두 테이블에 정상적으로 저장된다면, `Response`를 통해 성공 결과를 반환합니다.

유효성 검사에 실패한 경우에는 `raise_exception=True`로 인해 이 줄까지 도달하지 않고 **400 Bad Request**가 반환됩니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-12/2.webp" class="lazyload" width="100%" height="100%"/>

`post/` URL로 이동하여 결과를 조회한다면 정상적으로 두 테이블에 저장된 것을 확인할 수 있습니다.

<br>
<br>

## 주의사항

위 코드는 `Post`를 먼저 저장한 다음 `Comment`를 저장합니다.

만약, 댓글 데이터에 문제가 있어 `CommentSerializer`의 유효성 검사에 실패한다면 **400 Bad Request**가 반환되지만, 이미 저장된 `Post`는 데이터베이스에 그대로 남게 됩니다.

즉, 요청은 실패했는데 데이터의 일부만 저장되는 문제가 발생합니다.

이 문제는 다음 강좌의 `트랜잭션(Transaction)`으로 해결할 수 있습니다.

- Tip : 중첩된 데이터를 저장하는 로직을 뷰 대신 직렬화 클래스의 `create()` 메서드에 작성하는 **쓰기 가능한 중첩 직렬화(Writable Nested Serializer)** 방식도 사용할 수 있습니다. 이 경우 뷰는 기본 구현을 그대로 사용하고, 저장 로직을 직렬화 클래스에서 관리할 수 있습니다.
