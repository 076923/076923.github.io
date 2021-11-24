---
layout: post
title: "Python Django 강좌 : 제 12강 - Foreign Key (2)"
tagline: "Python Django Foreign Key"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django Foreign Key, Python Django Foreign Key insert, Python Django multiple insert
ref: Python-Django
category: Python
permalink: /posts/Python-Django-12/
comments: true
toc: true
---

## Django Foreign Key

앞선 외래키 사용법에서는 두 개의 **URL(post, comment)**를 생성해 각각 요청하여 데이터를 입력하였습니다.

`post` 기능과 `comment` 기능은 별도의 기능이기 때문에 분리하였지만, 게시물에 종속된 **별도의 기능(태그, 키워드, 정보)**을 추가해 다른 테이블로 분리했다면 테이블이 다르더라도 함께 작성되는 것이 더 효율적이고 관리하기 편할 수도 있습니다.

이렇듯 하나의 요청으로 **부모 테이블**과 **자식 테이블**이 함께 작성되어야 하는 경우에는 `뷰(views.py)`의 코드를 수정해 적용할 수 있습니다.

<br>
<br>

## 요청 데이터

![1]({{ site.images }}/assets/posts/Python/Django/lecture-12/1.png)

**게시물(post) 데이터**와 댓**글(comment) 데이터**를 함께 받아서 `post` 테이블과 `comment` 테이블에 한 번에 입력해보도록 하겠습니다.

Django Rest framework에서 지원되는 **HTML 입력 폼**으로는 지원되지 않는 형태이므로, `포스트맨(Postman)`이나 `코드`를 통해 데이터를 전달합니다.


{% highlight python %}

{
    "title" : "제 3강 - Django 프로젝트 설정",
    "contents" : "장고(Django)의 ...",
    "post" : {
        "contents" : "댓글까지 함께 작성."
    }
}

{% endhighlight %}

위와 같은 형태로 데이터를 `post` URL에 전달합니다.

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
    queryset = Post.objects.all()
    serializer_class = PostSerializer

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


class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

{% endhighlight %}

`create()` 메서드를 추가해 각각의 테이블에 맞는 데이터로 변경시키고, 변경시킨 데이터마다 직렬화합니다.

`request.data`는 앞선 **포스트맨(Postman) 데이터**와 동일한 형태를 갖습니다.

이 데이터를 파싱(parsing)하여 `PostSerializer`와 `CommentSerializer`에 맞게끔 변형합니다.

<br>

{% highlight python %}

post_data = {
    "title": request.data["title"],
    "contents": request.data["contents"],
}
post_serializer = PostSerializer(data=post_data)
if post_serializer.is_valid(raise_exception=True):
    post_result = post_serializer.save()

{% endhighlight %}

`Comment` 테이블은 `Post` 테이블의 `id`에 영향을 받기 때문에, 먼저 `Post` 테이블에 값을 저장해야 합니다.

그러므로, `PostSerializer`의 형식과 동일한 형태의 `post_data`를 생성합니다.

`PostSerializer`의 결과를 저장한 `post_serializer`를 생성합니다.

여기서, `self.get_serializer`을 사용해도 무방하지만 직관적으로 이해하기 위해 `PostSerializer`를 사용하도록 하겠습니다.

`*.is_valid()` 메서드로 유효성을 확인합니다. 유효하다면 `post_serializer.save()`를 통해 데이터베이스에 값을 저장합니다.

`post_result`는 저장된 결과를 반환합니다.

<br>

{% highlight python %}

comment_data = {
    "post_id": post_result.id,
    "contents": request.data["post"]["contents"],
}
comment_serializer = CommentSerializer(data=comment_data)
if comment_serializer.is_valid(raise_exception=True):
    comment_serializer.save()
    return Response({"message": "Operate successfully"}, status=status.HTTP_201_CREATED)

{% endhighlight %}

앞선 방식과 동일한 방식으로 한 번 더 반복합니다.

단, `comment` 테이블은 외래키인 `post_id` 값을 필요로 합니다.

그러므로, `post` 테이블에 값을 저장하면서 발생한 `id` 값을 가져와 `post_id` 필드의 값으로 사용합니다.

정상적으로 저장된다면, `Response`을 통해 성공 결과를 반환합니다.

<br>

{% highlight python %}

return Response({"message": "Error"}, status=status.HTTP_400_BAD_REQUEST)

{% endhighlight %}

요청 결과가 실패했을 때에도 `Response`을 통해 실패 결과를 반환합니다.

<br>

![2]({{ site.images }}/assets/posts/Python/Django/lecture-12/2.png)

`post` URL로 이동하여 결과를 조회한다면 정상적으로 두 테이블에 저장된 것을 확인할 수 있습니다.
