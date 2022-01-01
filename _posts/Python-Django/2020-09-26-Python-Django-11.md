---
layout: post
title: "Python Django 강좌 : 제 11강 - Foreign Key (1)"
tagline: "Python Django Foreign Key"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django Foreign Key, Python Django related_name, Python Django on_delete, Python Django db_column, Python Django CASCADE, Python Django PROTECT, Python Django SET_NULL, Python Django SET, Python Django DO_NOTHING, Python Django to_representation, Python Django self.fields
ref: Python-Django
category: Python
permalink: /posts/Python-Django-11/
comments: true
toc: true
---

## Django Foreign Key

`외래키(Foreign Key)`란 테이블의 필드 중에서 다른 테이블의 행과 **식별할 수 있는 키**를 의미합니다.

일반적으로 외래키가 포함된 테이블을 **자식 테이블**이라 하며, 외래키 값을 갖고 있는 테이블은 **부모 테이블**이라 합니다.

즉, 외래키란 **테이블과 테이블을 연결**하기 위해 사용되는 키입니다.

만약, 외래키를 사용하지 않고 **게시물**과 **댓글**의 내용을 저장할 기능을 구현한다면, 다음과 같은 테이블로 생성해야 합니다.

<br>
<br>

## Post 테이블

|  id | 제목                          | 내용               | 댓글1         | 댓글2       | 댓글3 | ... | 댓글N |
|:---:|-------------------------------|--------------------|---------------|-------------|-------|-----|-------|
|  1  | 제 1강 - Django 소개 및 설치  | 장고(Django)는 ... | 안녕하세요... | 감사합니다. | null  | ... | null  |
|  2  | 제 2강 - Django 프로젝트 생성 | 장고(Django)를 ... | 질문이 있...  | null        | null  | ... | null  |
| ... | ...                           | ...                | ...           | ...         | ...   | ... | ...   |

<br>

위와 같은 형태는 하나의 테이블에 너무 많은 `열(column)`이 추가되어, 매우 효율적이지 못한 구조가 됩니다.

또한, 열을 N개까지 추가했더라 하더라도 너무 많은 열과 데이터로 인해 조회하는데에 비교적 오랜 시간이 소요됩니다.

위와 같은 구조를 두 개의 테이블로 나눈다면 다음과 같이 변경할 수 있습니다.

<br>

### Post 테이블

|  id | 제목                          | 내용               | 
|:---:|-------------------------------|--------------------|
|  1  | 제 1강 - Django 소개 및 설치  | 장고(Django)는 ... |
|  2  | 제 2강 - Django 프로젝트 생성 | 장고(Django)를 ... |
| ... | ...                           | ...                |

<br>

### Comment 테이블

|  id | post_id | 내용               |
|:---:|---|--------------------|
|  1  | 1 | 안녕하세요... |
|  2  | 1 | 감사합니다. |
|  3  | 2 | 질문이 있... |
| ... | ... | ... |


`Comment` 테이블을 별도로 생성한 다음, 해당 내용이 어느 `Post` 테이블의 `id`에서 사용됬는지 표기한다면 간단한 구조로 생성할 수 있습니다.

불필요한 열이 생성되지 않아, 효율적인 테이블을 구성할 수 있습니다.

<br>
<br>

## Django Code 구성

### models.py

{% highlight python %}

from django.db import models

# Create your models here.
class Post(models.Model):
    id = models.BigAutoField(help_text="Post ID", primary_key=True)
    title = models.CharField(help_text="Post title", max_length=100, blank=False, null=False)
    contents = models.TextField(help_text="post contents", blank=False, null=False)


class Comment(models.Model):
    id = models.BigAutoField(help_text="Comment ID", primary_key=True)
    post_id = models.ForeignKey("Post", related_name="post", on_delete=models.CASCADE, db_column="post_id")
    contents = models.TextField(help_text="Comment contents", blank=False, null=False)

{% endhighlight %}

게시물의 제목과 내용을 저장할 `Post` 테이블을 생성합니다.

`Post` 테이블의 `식별자(id)`, `제목(title)`, `내용(contents)` 필드에 관한 정의를 작성합니다.

<br>

게시물에 작성될 댓글의 내용을 저장할 `Comment` 테이블을 생성합니다.

`Comment` 테이블의 `식별자(id)`, `외래키(post_id)`, `내용(contents)` 필드에 관한 정의를 작성합니다.

<br>

외래키를 작성할 때 필수적으로 포함되어야할 매개변수는 **참조할 테이블**, **개체 관계에 사용할 이름**, **개체 삭제시 수행할 동작** 등 입니다.

`참조할 테이블`은 외래키에서 어떤 테이블을 참조할지 의미합니다. 예제에서는 `Post` 테이블입니다.

`개체 관계에 사용할 이름(related_name)`은 추상 모델에서 관계를 정의할 때 사용될 이름을 의미합니다. 예제에서는 `post`입니다.

`개체 삭제시 수행할 동작(on_delete)`은 외래키(ForeignKey)가 바라보는 테이블의 값이 삭제될 때 수행할 방법을 지정합니다.

즉, 게시물이 삭제될 때 댓글은 어떻게 처리할지를 정의합니다.

`데이터베이스 상의 필드 이름(db_column)`은 테이블에 정의될 이름을 의미합니다.

만약, `db_column` 매개변수를 사용하지 않는다면, 데이터베이스 필드에 작성될 필드명은 `post_id_id`가 됩니다.

`post_id_id`는 의도한 필드명이 아니므로, `db_column` 매개변수의 인수에 `post_id`를 사용합니다.

<br>

#### on_delete

| on_delete | 의미 |
|:---------:|:----:|
| models.CASCADE | 외래키를 포함하는 행도 함께 삭제 |
| models.PROTECT | 해당 요소가 함께 삭제되지 않도록 오류 발생 (ProtectedError) |
| models.SET_NULL | 외래키 값을 NULL 값으로 변경 (null=True일 때 사용 가능)|
| models.SET(func)  | 외래키 값을 func 행동 수행 (func는 함수나 메서드 등을 의미) |
| models.DO_NOTHING | 아무 행동을 하지 않음 |

<br>

### views.py

{% highlight python %}

from rest_framework import viewsets

from blog.models import Post
from blog.models import Comment
from blog.serializers import PostSerializer
from blog.serializers import CommentSerializer


class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer


class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

{% endhighlight %}

뷰는 별도의 알고리즘을 추가하지 않고 기본 형태를 사용합니다.

게시물 작성과 댓글 작성은 별도의 기능이므로, 두 개의 뷰셋을 생성합니다.

<br>

### serializers.py

{% highlight python %}

from rest_framework import serializers
from blog.models import Post
from blog.models import Comment


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ("post_id", "contents")


class PostSerializer(serializers.ModelSerializer):
    post = CommentSerializer(many=True, read_only=True)

    class Meta:
        model = Post
        fields = ("id", "title", "contents", "post")

{% endhighlight %}

`CommentSerializer`와 `PostSerializer`를 선언합니다.

여기서, `PostSerializer`는 `CommentSerializer`를 불러올 예정이므로, `CommentSerializer`, `PostSerializer` 순으로 선언합니다.

<br>

댓글은 참조한 **게시물의 ID**와 **내용**을 확인할 예정이므로, `fields`에 `post_id`와 `contents`를 선언합니다.

<br>

게시물은 **게시물의 ID** **제목**, **내용**, **댓글 내용**을 확인할 예정입니다.

여기서, 댓글은 `Comment` 테이블에 작성됩니다. 그러므로, `CommentSerializer`를 통해 직렬화를 해야합니다.

새로운 필드인 `post`를 선언하고 해당 필드는 `CommentSerializer`를 통하도록 합니다.

여러 개의 댓글이 작성될 수 있으므로, `many` 매개변수는 `True`로 사용하고, 게시물에서 댓글을 수정하지 않으므로, `read_only` 매개변수도 `True`로 사용합니다.

여기서, 새로운 필드인 `post` 변수명은 `models.py`에서 작성한 `개체 관계에 사용할 이름(related_name)`으로 작성해야 합니다.

만약, 전혀 다른 변수명으로 사용할 경우 정상적으로 동작하지 않습니다.

<br>

### urls.py

{% highlight python %}

from django.conf.urls import url
from blog.views import PostViewSet
from blog.views import CommentViewSet


urlpatterns = [
    url('post', PostViewSet.as_view({'get':'list', 'post':'create'})),
    url('comment', CommentViewSet.as_view({'get':'list', 'post':'create'})),
]

{% endhighlight %}

`URL`은 `post`와 `comment`를 추가합니다.

간단한 **조회(get, list)**와 **작성(post, create)**만 수행하도록 하겠습니다.

<br>
<br>

## Django Runserver

### Post

![1]({{ site.images }}/assets/posts/Python/Django/lecture-11/1.webp){: width="100%" height="100%"}

`post` URL로 이동해, 게시물의 **제목**과 **내용**을 입력합니다.

<br>

![2]({{ site.images }}/assets/posts/Python/Django/lecture-11/2.webp){: width="100%" height="100%"}

두 게시물이 작성되면 위와 같이 표시되는 것을 알 수 있습니다.

`post` 필드는 현재 작성된 댓글이 없으므로 `[]`의 형태로 표시됩니다.

<br>

### Comment

![3]({{ site.images }}/assets/posts/Python/Django/lecture-11/3.webp){: width="100%" height="100%"}

`comment` URL로 이동해, 특정 게시물에 댓글의 **내용**을 입력합니다.

여기서, 어떤 게시물에 작성할지 `post_id`를 선택하게 됩니다.

`post_id`는 외래키이므로, `Post` 테이블에 존재하는 값만 입력할 수 있습니다.

<br>

![4]({{ site.images }}/assets/posts/Python/Django/lecture-11/4.webp){: width="100%" height="100%"}

세 개의 댓글이 작성되면 위와 같이 표시되는 것을 알 수 있습니다.

<br>

### Post

![5]({{ site.images }}/assets/posts/Python/Django/lecture-11/5.webp){: width="100%" height="100%"}

이제 다시, `post` URL로 이동하면 게시물마다 어떤 댓글이 달렸는지 확인할 수 있습니다.

<br>
<br>

## 자식 테이블에서 부모 테이블 참조하기

현재 `comment` URL에서는 댓글이 어떤 부모 테이블의 행을 참조했는지 확인하기 어렵습니다.

만약, 댓글에서 게시물의 제목과 내용 등을 확인해야하는 일이 발생한다면, 아래와 같이 코드를 추가해 확인할 수 있습니다.

<br>

### serializers.py

{% highlight python %}

from rest_framework import serializers
from blog.models import Post
from blog.models import Comment


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ("post_id", "contents")

    def to_representation(self, instance):
        self.fields['post_id'] =  PostRepresentationSerializer(read_only=True)
        return super(CommentSerializer, self).to_representation(instance)


class PostSerializer(serializers.ModelSerializer):
    post = CommentSerializer(many=True, read_only=True)

    class Meta:
        model = Post
        fields = ("id", "title", "contents", "post")


class PostRepresentationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ("id", "title", "contents")

{% endhighlight %}

댓글에서 게시물의 내용을 확인하기 위해서 `to_representation` 메서드를 추가합니다.

`to_representation` 메서드는 `Object instance` 형식을 `사전(Dictionary)` 형태로 변경시킵니다.

현재 필드들(self.fields) 중에서 `post_id`의 필드를 다시 직렬화해 부모의 테이블에서 가져오게 합니다.

이때, `PostRepresentationSerializer` 클래스를 새로 생성해야합니다.

`PostRepresentationSerializer`는 `PostSerializer` 클래스와 형태가 비슷하나, `post` 필드를 사용하지 않습니다.

`PostSerializer` 클래스로 직렬화 한다면, 다시 `CommentSerializer`를 부르게 되고, **재귀(Recursion)**에 빠지게 됩니다.

그러므로, `PostRepresentationSerializer`을 선언해 `post` 필드를 제거한 직렬화 클래스를 사용합니다.

<br>

![6]({{ site.images }}/assets/posts/Python/Django/lecture-11/6.webp){: width="100%" height="100%"}

다시, `comment` URL로 이동해 결과를 확인한다면 위와 같은 형태로 표시됩니다.

`post_id` 필드의 값을 `사전(Dictionary)` 형식으로 변경시켜, 해당 값이 부모 테이블의 행으로 출력됩니다.
