---
layout: post
title: "Python Django 강좌 : 제 7강 - Serializers"
tagline: "Python Django serializers"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Django']
keywords: Python, Python Django, Python Django 3, Python Django Serializers, Python Django serializers.py, Python Django SerializerMethodField, Python Django validate, Python Django ValidationError, Python Django ModelSerializer, Python Django serializers.Field
ref: Python
category: posts
permalink: /posts/Python-Django-7/
comments: true
---

## Django Serializers ##
----------

장고에서 직렬화는 `쿼리셋(querysets)`이나 `모델 인스턴스(model instances)`와 같은 복잡한 구조의 데이터를 **JSON, XML** 등의 형태로 변환하는 역할을 합니다.

즉, Python 환경에 **적합한 구조로 재구성할 수 있는 포맷으로 변환**하는 과정을 의미합니다.

직렬화를 비롯해 `역직렬화(deserialization)`도 지원하며, 직렬화와 역직렬화를 지원하므로 데이터 유효성 검사도 함께 진행됩니다.

데이터를 접근하거나, 인스턴스를 저장하기 전에 항상 유효성을 검사해야하며, 데이터의 구조나 값이 **유효하지 않으면 오류**를 반환합니다.

`serializers.py` 파일에 직렬화에 관한 논리를 정의합니다.

`serializers.py`는 기본 앱 구성에 포함되지 않으므로, 별도로 생성해야 합니다.

직렬화 파일은 아래의 디렉토리 구조로 파일을 생성합니다.

<br>

```

[현재 프로젝트]/
  > 📁 [장고 프로젝트 이름]
  ⬇ 📁 [장고 앱 이름]
    ⬇ 📁 migrations
      🖹 __init__.py
    🖹 __init__.py
    🖹 admin.py
    🖹 apps.py
    🖹 models.py
    🖹 tests.py
    🖹 view.py
    🖹 serializers.py
  🖹 manage.py

```

<br>

## serializers.py ##
----------

{% highlight python %}

from rest_framework import serializers
from first_app.models import UserModel

# Create your serializers here.
class UserSerializer(serializers.ModelSerializer):
    event_age = serializers.SerializerMethodField(help_text="Custom Field")

    def get_event_age(self, instance):
        return True if instance.age < 30 else False
        
    class Meta:
        model = UserModel
        fields = "__all__"

    def validate_email(self, instance):

        if "admin" in instance:
            raise serializers.ValidationError(detail="사용할 수 없는 메일 계정입니다.")

        return instance

    def validate_name(self, instance):

        if len(instance) < 2:
            raise serializers.ValidationError(detail="이름이 올바르지 않습니다.")

        return instance

    def validate_age(self, instance):

        if instance < 19:
            raise serializers.ValidationError(detail="회원 가입이 불가능한 나이입니다.")

        return instance


    def validate_nationality(self, instance):
        return instance

{% endhighlight %}

직렬화(serializers.py) 파일을 위와 같이 정의합니다.

DRF에서는 `Serializer`는 **모델 인스턴스(model instances)**나 파이썬 **내장 함수(primitives)**를 마샬링합니다.

마샬링 프로세스는 파서(parsers)와 렌더러(renderers)에 의해 처리됩니다.

`ModelSerializer` 클래스는 기본 필드를 자동으로 채울 수 있으며, 유효성 검사 및 `create()` 메서드와 `update()` 구현이 제공됩니다.

<br>
<br>

## Module ##
----------

{% highlight python %}

from rest_framework import serializers
from first_app.models import UserModel

{% endhighlight %}

`from rest_framework import serializers`는 직렬화와 관련된 정의를 가져옵니다.

`from first_app.models import UserModel`는 `models.py`에서 선언한 UserModel 모델입니다.

<br>
<br>

## ModelSerializer ##
----------

{% highlight python %}

# Create your serializers here.
class UserSerializer(serializers.ModelSerializer):
    event_age = serializers.SerializerMethodField(help_text="Custom Field")

    def get_event_age(self, instance):
        return True if instance.age < 30 else False

{% endhighlight %}

`UserSerializer`의 이름으로 직렬화 클래스를 생성하고, `ModelSerializer`을 상속받아 사용합니다.

`ModelSerializer` 클래스는 `Serializer` 클래스를 사용하며, `create()`, `update()` 등의 기능을 제공합니다.

모델에서 정의한 필드에 대한 값을 가져와 사용하며, `SerializerMethodField`를 통해 임의의 필드를 사용할 수 있습니다.

모델에서 정의한 필드가 아니라면 사용할 수 없지만, `SerializerMethodField`를 사용하면 모델의 필드 값 등을 변형해 사용할 수 있습니다.

`SerializerMethodField`를 선언하면 해당 필드를 조회할 때 실행할 함수를 생성해야 합니다.

`def get_<필드명>`의 형태로 함수를 생성할 수 있습니다.

데이터가 조회될 때, `get_<필드명>` 함수가 실행됩니다.

예제의 함수는 데이터베이스의 `age` 필드가 `30` 미만인 경우에는 `True`를 반환하며, `30` 이상인 경우에는 `False`를 반환합니다.

## Meta ##
----------

{% highlight python %}

class Meta:
    model = UserModel
    fields = "__all__"

{% endhighlight %}

`Meta` 클래스는 어떤 **모델**을 사용할지 정의하며, 해당 모델에서 어떤 **필드**를 사용할지 정의합니다.

`fields`의 값을 `__all__`로 사용하는 경우, 모델의 모든 필드를 사용합니다.

만약, 특정 필드만 사용한다면 `fields = ("email", "name", "age", )` 등의 형태로 사용하려는 필드만 적용할 수 있습니다.

<br>
<br>

## validate ##
----------

{% highlight python %}

def validate_email(self, instance):

    if "admin" in instance:
        raise serializers.ValidationError(detail="사용할 수 없는 메일 계정입니다.")

    return instance

def validate_name(self, instance):

    if len(instance) < 2:
        raise serializers.ValidationError(detail="이름이 올바르지 않습니다.")

    return instance

def validate_age(self, instance):

    if instance < 19:
        raise serializers.ValidationError(detail="회원 가입이 불가능한 나이입니다.")

    return instance

{% endhighlight %}

`validate_<필드명>`을 통해 특정 필드에 입력된 값에 대해 별도의 유효성 검사를 진행할 수 있습니다.

`raise serializers.ValidationError(detail="오류 내용")`을 통해 유효성 오류를 발생시킬 수 있습니다.

`email` 필드에 **admin**이라는 문자열이 포함되어 있다면, 유효성 오류를 발생시킵니다.

`name` 필드의 글자수가 **두 글자 미만**이라면, 유효성 오류를 발생시킵니다.

`age` 필드가 **20 미만**이라면, 유효성 오류를 발생시킵니다.


<br>

### 필드 매핑 ###
----------

#### ID 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.AutoField | serializers.IntegerField |
| models.BigAutoField | serializers.IntegerField |
| models.UUIDField | serializers.UUIDField |

#### 문자열 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.CharField | serializers.CharField |
| models.TextField | serializers.CharField |
| models.URLField | serializers.URLField |
| models.EmailField | serializers.EmailField |

#### 데이터 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.BinaryField | serializers.Field |
| models.DecimalField | serializers.DecimalField |
| models.IntegerField | serializers.IntegerField |
| models.PositiveIntegerField | serializers.IntegerField |
| models.FloatField | serializers.FloatField |
| models.BooleanField | serializers.BooleanField |
| models.NullBooleanField | serializers.NullBooleanField |

#### 날짜 및 시간 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.DateField | serializers.DateField |
| models.TimeField | serializers.TimeField |
| models.DateTimeField | serializers.DateTimeField |

#### 기타 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.ImageField | serializers.ImageField |
| models.FileField | serializers.FileField |
| models.FilePathField | serializers.FilePathField |

#### 관계 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| OneToOneField | Serializer Class |
| ForeignKey | Serializer Class |
| ManyToManyField | Serializer Class |


<br>

직렬화 필드는 모델 필드와 매핑됩니다.

예를 들어, `models.TextField`는 `serializers.CharField`로 매핑됩니다.

직렬화 필드에는 `TextField` 필드가 없으므로, 위의 표에서 맞는 매핑을 찾아서 작성해야 합니다.

일반적으로 대부분이 모델 필드와 비슷하거나 동일한 형태의 구조를 갖고 있습니다.

`관계 분야` 필드는 필드 안에 다른 필드들이 존재하므로, `Serializer` 클래스를 생성해서 내부에서 또 유효성 검사를 진행해야합니다.

그러므로, 별도의 클래스를 생성해서 매핑합니다.

직렬화 필드도 모델 필드처럼 옵션값이 존재합니다. 옵션값은 다음과 같습니다.

<br>

### 필드 옵션 ###
----------

| 옵션 | 설명 | 기본값 |
|:-:|------|:------:|
| default | 필드의 기본값을 설정합니다. | - |
| label | HTML Form 등에 표시될 문자열을 설정합니다. |
| help_text | 도움말 텍스트를 설정합니다. | - |
| read_only | 읽기 전용 필드로 설정합니다. | False |
| write_only | 쓰기 전용 필드로 설정합니다. | False |
| required | 역직렬화 여부를 설정합니다. | True |
| allow_null | Null 값을 허용합니다. | False |
| vaildators | 유효성 검사를 적용할 함수를 등록합니다. | - |
| error_messages | 에러 메세지를 설정합니다. | - |

<br>

### 메타 옵션 ###
----------

메타(Meta) 클래스는 **직렬화 필드**에서 사용할 수 있는 설정을 적용합니다.

<br>

| 옵션 | 설명 |
|:-:|------|
| fields | 직렬화에 포함할 필드를 설정합니다. |
| exclude | 직렬화에 제외할 필드를 설정합니다. |
| read_only_fields | 읽기 전용 필드를 설정합니다. |
| extra_kwargs | 추가 옵션을 설정합니다. |
| depth | 외래키 표현 제한 단계를 설정합니다. |
