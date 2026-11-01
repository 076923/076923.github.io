---
layout: post
title: "Python Django 강좌 : 제 11강 - Foreign Key (1)"
tagline: "Python Django Foreign Key"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Foreign Key, Python Django related_name, Python Django on_delete, Python Django db_column, Python Django CASCADE, Python Django PROTECT, Python Django RESTRICT, Python Django SET_NULL, Python Django SET, Python Django DO_NOTHING, Python Django to_representation, Python Django Nested Serializer, Python Django prefetch_related, Python Django select_related
ref: Python-Django
category: Python
permalink: /posts/Python-Django-11/
comments: true
toc: true
---

## Django Foreign Key

`외래키(Foreign Key)`란 테이블의 필드 중에서 **다른 테이블의 행을 식별할 수 있는 키**를 의미합니다.

일반적으로 외래키가 포함된 테이블을 **자식 테이블**이라 하며, 외래키가 참조하는(가리키는) 테이블을 **부모 테이블**이라 합니다.

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

또한, 댓글의 최대 개수를 미리 정해야 하며, 대부분의 칸이 `null`로 비어 있어 저장 공간이 낭비되고 조회와 관리가 어려워집니다.

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

<br>

`Comment` 테이블을 별도로 생성한 다음, 해당 댓글이 어느 `Post` 행(`id`)에 작성됐는지 `post_id`로 표기한다면 간단한 구조로 생성할 수 있습니다.

불필요한 열이 생성되지 않고 댓글의 개수에도 제한이 없어, 효율적인 테이블을 구성할 수 있습니다.

이처럼 하나의 게시물에 여러 개의 댓글이 연결되는 관계를 `일대다(1:N)` 관계라고 합니다.

<br>
<br>

## Django Code 구성

이번 예제는 `blog`라는 새로운 앱을 생성해 진행합니다.

`python manage.py startapp blog`로 앱을 생성한 다음, `INSTALLED_APPS`에 `'blog'`를 등록합니다.

<br>

### models.py

{% highlight python %}

from django.db import models


# Create your models here.
class Post(models.Model):
    id = models.BigAutoField(help_text="Post ID", primary_key=True)
    title = models.CharField(help_text="Post title", max_length=100, blank=False, null=False)
    contents = models.TextField(help_text="Post contents", blank=False, null=False)


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

`ForeignKey`를 작성할 때 필수적으로 포함되어야할 매개변수는 **참조할 모델**과 **개체 삭제시 수행할 동작(on_delete)**입니다.

`참조할 모델`은 외래키에서 어떤 테이블을 참조할지 의미합니다. 예제에서는 `Post` 모델입니다. 모델 클래스나 `"Post"`와 같은 문자열로 지정할 수 있습니다.

`개체 삭제시 수행할 동작(on_delete)`은 외래키(ForeignKey)가 바라보는 부모 테이블의 행이 삭제될 때 수행할 방법을 지정합니다.

즉, 게시물이 삭제될 때 댓글은 어떻게 처리할지를 정의합니다.

<br>

`역참조 이름(related_name)`은 **부모 모델에서 자식 모델로 거꾸로 접근할 때** 사용할 이름을 의미합니다. 예제에서는 `post`입니다.

예를 들어, 게시물 객체에서 `post.post.all()`로 해당 게시물에 작성된 모든 댓글을 가져올 수 있습니다.

`related_name`을 지정하지 않으면 `소문자 모델 이름_set`(예 : `comment_set`)이 기본 이름으로 사용됩니다.

<br>

`데이터베이스 상의 필드 이름(db_column)`은 테이블에 정의될 컬럼 이름을 의미합니다.

장고는 `ForeignKey` 필드의 컬럼 이름에 자동으로 `_id`를 붙이므로, `db_column` 매개변수를 사용하지 않는다면 데이터베이스에 작성될 컬럼명은 `post_id_id`가 됩니다.

`post_id_id`는 의도한 컬럼명이 아니므로, `db_column` 매개변수의 인수에 `post_id`를 사용합니다.

- Tip : 장고의 관례는 외래키 필드 이름을 `post`처럼 **참조하는 객체의 이름**으로 짓고, 역참조 이름은 `comments`처럼 **자식 객체의 복수형**으로 짓는 것입니다. 이 경우 `post = models.ForeignKey(Post, related_name="comments", on_delete=models.CASCADE)`로 선언하면 `db_column` 없이도 컬럼명이 `post_id`가 되고, `post.comments.all()`처럼 읽기 쉬운 코드가 됩니다. 이번 강좌는 이미지와 결과를 맞추기 위해 `post_id`, `post` 이름을 그대로 사용합니다.

<br>

#### on_delete

| on_delete | 의미 |
|:---------:|:----:|
| models.CASCADE | 부모 행이 삭제되면 외래키를 포함하는 자식 행도 함께 삭제 |
| models.PROTECT | 자식 행이 존재하면 부모 행이 삭제되지 않도록 오류 발생 (ProtectedError) |
| models.RESTRICT | PROTECT와 비슷하지만, 같은 삭제 작업에서 CASCADE로 함께 삭제되는 경우에는 삭제를 허용 (RestrictedError) |
| models.SET_NULL | 외래키 값을 NULL 값으로 변경 (null=True일 때 사용 가능) |
| models.SET_DEFAULT | 외래키 값을 필드의 기본값으로 변경 (default 설정 필요) |
| models.SET(value) | 외래키 값을 지정한 값 또는 함수(callable)가 반환한 값으로 변경 |
| models.DO_NOTHING | 장고에서 아무 행동을 하지 않음 (데이터베이스의 제약 조건에 따라 무결성 오류가 발생할 수 있음) |

- Tip : Django 6.1에서는 `models.DB_CASCADE`, `models.DB_SET_NULL`처럼 장고가 아닌 **데이터베이스의 제약 조건**으로 삭제 동작을 처리하는 옵션도 사용할 수 있습니다.

<br>

### views.py

{% highlight python %}

from rest_framework import viewsets

from blog.models import Post
from blog.models import Comment
from blog.serializers import PostSerializer
from blog.serializers import CommentSerializer


class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.prefetch_related("post")
    serializer_class = PostSerializer


class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.select_related("post_id")
    serializer_class = CommentSerializer

{% endhighlight %}

뷰는 별도의 알고리즘을 추가하지 않고 기본 형태를 사용합니다.

게시물 작성과 댓글 작성은 별도의 기능이므로, 두 개의 뷰셋을 생성합니다.

<br>

`prefetch_related("post")`와 `select_related("post_id")`는 연결된 테이블의 데이터를 **미리 함께 조회**하는 메서드입니다.

게시물 목록을 직렬화할 때 게시물마다 댓글을 조회하면, 게시물이 100개일 경우 `1 + 100`번의 쿼리가 실행됩니다. 이를 `N+1 문제`라고 합니다.

`prefetch_related`는 **역참조나 다대다 관계**를 별도의 쿼리 한 번으로 모아서 가져오며, `select_related`는 **정참조(ForeignKey, OneToOne) 관계**를 `JOIN`으로 한 번에 가져옵니다.

- Tip : `objects.all()`을 사용해도 결과는 같지만, 데이터가 많아질수록 쿼리 수가 크게 늘어납니다.

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

여기서, `PostSerializer`는 클래스를 정의할 때 `CommentSerializer`를 사용하므로, `CommentSerializer`, `PostSerializer` 순으로 선언합니다.

<br>

댓글은 참조한 **게시물의 ID**와 **내용**을 확인할 예정이므로, `fields`에 `post_id`와 `contents`를 선언합니다.

`post_id`는 `ForeignKey`이므로, 7강의 필드 매핑과 같이 `PrimaryKeyRelatedField`로 매핑되어 **게시물의 ID 값**으로 표현됩니다.

<br>

게시물은 **게시물의 ID**, **제목**, **내용**, **댓글 내용**을 확인할 예정입니다.

여기서, 댓글은 `Comment` 테이블에 작성됩니다. 그러므로, `CommentSerializer`를 통해 직렬화를 해야합니다.

새로운 필드인 `post`를 선언하고 해당 필드는 `CommentSerializer`를 통하도록 합니다. 이를 **중첩 직렬화(Nested Serializer)**라고 합니다.

여러 개의 댓글이 작성될 수 있으므로, `many` 매개변수는 `True`로 사용하고, 게시물에서 댓글을 수정하지 않으므로, `read_only` 매개변수도 `True`로 사용합니다.

여기서, 새로운 필드인 `post` 변수명은 `models.py`에서 작성한 `역참조 이름(related_name)`과 같아야 합니다.

직렬화 필드는 기본적으로 **필드 이름과 같은 속성**을 모델에서 찾기 때문입니다.

- Tip : 다른 이름을 사용하려면 `comments = CommentSerializer(source="post", many=True, read_only=True)`처럼 `source` 매개변수로 실제 속성 이름을 지정합니다.

<br>

### urls.py

{% highlight python %}

## blog/urls.py
from django.urls import path
from blog.views import PostViewSet
from blog.views import CommentViewSet


urlpatterns = [
    path("post/", PostViewSet.as_view({"get": "list", "post": "create"})),
    path("comment/", CommentViewSet.as_view({"get": "list", "post": "create"})),
]

{% endhighlight %}

`URL`은 `post/`와 `comment/`를 추가합니다.

간단한 **조회(get, list)**와 **작성(post, create)**만 수행하도록 하겠습니다.

8강과 같이 `프로젝트 urls.py`에 `path("", include("blog.urls"))`를 추가해 연결합니다.

<br>

### 마이그레이션

{% highlight bash %}

python manage.py makemigrations blog
python manage.py migrate blog

{% endhighlight %}

새로운 모델을 작성했으므로, 9강과 같이 마이그레이션을 생성하고 적용합니다.

<br>
<br>

## Django Runserver

### Post

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-11/1.webp" class="lazyload" width="100%" height="100%"/>

`post/` URL로 이동해, 게시물의 **제목**과 **내용**을 입력합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-11/2.webp" class="lazyload" width="100%" height="100%"/>

두 게시물이 작성되면 위와 같이 표시되는 것을 알 수 있습니다.

`post` 필드는 현재 작성된 댓글이 없으므로 `[]`의 형태로 표시됩니다.

<br>

### Comment

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-11/3.webp" class="lazyload" width="100%" height="100%"/>

`comment/` URL로 이동해, 특정 게시물에 댓글의 **내용**을 입력합니다.

여기서, 어떤 게시물에 작성할지 `post_id`를 선택하게 됩니다.

`post_id`는 외래키이므로, `Post` 테이블에 존재하는 값만 입력할 수 있습니다.

존재하지 않는 값을 입력하면 `Invalid pk "3" - object does not exist.`와 같은 유효성 오류가 반환됩니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-11/4.webp" class="lazyload" width="100%" height="100%"/>

세 개의 댓글이 작성되면 위와 같이 표시되는 것을 알 수 있습니다.

<br>

### Post

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-11/5.webp" class="lazyload" width="100%" height="100%"/>

이제 다시, `post/` URL로 이동하면 게시물마다 어떤 댓글이 달렸는지 확인할 수 있습니다.

<br>
<br>

## 자식 테이블에서 부모 테이블 참조하기

현재 `comment/` URL에서는 댓글이 참조한 게시물의 **ID**만 표시되므로, 어떤 게시물인지 확인하기 어렵습니다.

만약, 댓글에서 게시물의 제목과 내용 등을 확인해야하는 일이 발생한다면, 아래와 같이 코드를 추가해 확인할 수 있습니다.

<br>

### serializers.py

{% highlight python %}

from rest_framework import serializers
from blog.models import Post
from blog.models import Comment


class PostRepresentationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ("id", "title", "contents")


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ("post_id", "contents")

    def to_representation(self, instance):
        self.fields["post_id"] = PostRepresentationSerializer(read_only=True)
        return super().to_representation(instance)


class PostSerializer(serializers.ModelSerializer):
    post = CommentSerializer(many=True, read_only=True)

    class Meta:
        model = Post
        fields = ("id", "title", "contents", "post")

{% endhighlight %}

댓글에서 게시물의 내용을 확인하기 위해서 `to_representation` 메서드를 재정의합니다.

`to_representation` 메서드는 **모델 인스턴스를 응답에 사용할 사전(Dictionary) 형태로 변환**하는 메서드입니다.

즉, 데이터를 **출력할 때만** 호출되며, 데이터를 입력받는 과정(역직렬화)에는 영향을 주지 않습니다.

출력하기 직전에 현재 필드들(self.fields) 중에서 `post_id` 필드를 `PostRepresentationSerializer`로 교체해, 게시물의 ID 대신 게시물의 내용을 출력하게 합니다.

입력할 때는 기존과 같이 `post_id`에 **게시물의 ID**를 전달하면 됩니다.

<br>

이때, `PostRepresentationSerializer` 클래스를 새로 생성해야합니다.

`PostRepresentationSerializer`는 `PostSerializer` 클래스와 형태가 비슷하나, `post` 필드를 사용하지 않습니다.

`PostSerializer` 클래스로 직렬화 한다면, 다시 `CommentSerializer`를 부르게 되고, 서로를 계속 호출하는 **순환 참조(Circular reference)**에 빠지게 됩니다.

그러므로, `PostRepresentationSerializer`를 선언해 `post` 필드를 제거한 직렬화 클래스를 사용합니다.

- Tip : `super(CommentSerializer, self)`는 Python 2 방식의 표기입니다. Python 3에서는 `super()`로 간단하게 작성합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-11/6.webp" class="lazyload" width="100%" height="100%"/>

다시, `comment/` URL로 이동해 결과를 확인한다면 위와 같은 형태로 표시됩니다.

`post_id` 필드의 값이 `사전(Dictionary)` 형식으로 변경되어, 해당 값이 부모 테이블의 행으로 출력됩니다.

`to_representation`은 `PostSerializer` 안에서 댓글을 출력할 때도 적용되므로, `post/` URL의 댓글에도 게시물 정보가 함께 표시됩니다.
