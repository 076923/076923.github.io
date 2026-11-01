---
layout: post
title: "Python Django 강좌 : 제 7강 - Serializers"
tagline: "Python Django serializers"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Serializers, Python Django serializers.py, Python Django SerializerMethodField, Python Django validate, Python Django ValidationError, Python Django ModelSerializer, Python Django serializers.Field
ref: Python-Django
category: Python
permalink: /posts/Python-Django-7/
comments: true
toc: true
---

## Django Serializers

DRF에서 `직렬화(Serialization)`는 `쿼리셋(querysets)`이나 `모델 인스턴스(model instances)`와 같은 복잡한 구조의 데이터를 **Python 기본 자료형(dict, list, str 등)**으로 변환하는 역할을 합니다.

변환된 데이터는 이후 `렌더러(Renderer)`를 통해 **JSON, XML** 등의 형태로 쉽게 출력할 수 있습니다.

반대로, 클라이언트가 보낸 JSON 등의 데이터를 검증한 다음 모델 인스턴스 등으로 변환하는 과정을 `역직렬화(deserialization)`라고 합니다.

역직렬화를 할 때는 항상 **데이터 유효성 검사**가 함께 진행되며, 데이터의 구조나 값이 **유효하지 않으면 오류**를 반환합니다.

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
    🖹 serializers.py
    🖹 tests.py
    🖹 views.py
  🖹 manage.py

```

<br>
<br>

## serializers.py

{% highlight python %}

from rest_framework import serializers
from first_app.models import UserModel


# Create your serializers here.
class UserSerializer(serializers.ModelSerializer):
    event_age = serializers.SerializerMethodField(help_text="Custom Field")

    class Meta:
        model = UserModel
        fields = "__all__"

    def get_event_age(self, instance):
        return instance.age < 30

    def validate_email(self, value):
        if "admin" in value:
            raise serializers.ValidationError(detail="사용할 수 없는 메일 계정입니다.")
        return value

    def validate_name(self, value):
        if len(value) < 2:
            raise serializers.ValidationError(detail="이름이 올바르지 않습니다.")
        return value

    def validate_age(self, value):
        if value < 19:
            raise serializers.ValidationError(detail="회원 가입이 불가능한 나이입니다.")
        return value

{% endhighlight %}

직렬화(serializers.py) 파일을 위와 같이 정의합니다.

DRF의 `Serializer`는 **모델 인스턴스(model instances)**나 쿼리셋을 **Python 기본 자료형(primitives)**으로 변환하고, 그 반대 과정도 수행합니다.

Python 기본 자료형과 JSON 등의 실제 형식 사이의 변환은 `파서(parsers)`와 `렌더러(renderers)`가 처리합니다.

`ModelSerializer` 클래스는 모델을 보고 필드를 자동으로 생성하며, 모델에 정의된 유효성 검사와 `create()`, `update()` 메서드의 기본 구현을 제공합니다.

<br>

### Module

{% highlight python %}

from rest_framework import serializers
from first_app.models import UserModel

{% endhighlight %}

`from rest_framework import serializers`는 직렬화와 관련된 클래스를 가져옵니다.

`from first_app.models import UserModel`은 `models.py`에서 선언한 UserModel 모델입니다.

<br>

### ModelSerializer

{% highlight python %}

# Create your serializers here.
class UserSerializer(serializers.ModelSerializer):
    event_age = serializers.SerializerMethodField(help_text="Custom Field")

    def get_event_age(self, instance):
        return instance.age < 30

{% endhighlight %}

`UserSerializer`의 이름으로 직렬화 클래스를 생성하고, `ModelSerializer`를 상속받아 사용합니다.

`ModelSerializer` 클래스는 `Serializer` 클래스를 상속하며, `create()`, `update()` 등의 기능을 제공합니다.

모델에서 정의한 필드에 대한 값을 가져와 사용하며, `SerializerMethodField`를 통해 모델에 없는 임의의 필드를 추가할 수 있습니다.

`SerializerMethodField`를 사용하면 모델의 필드 값 등을 변형해 새로운 값을 만들 수 있습니다.

`SerializerMethodField`를 선언하면 해당 필드를 조회할 때 실행할 메서드를 생성해야 합니다.

`def get_<필드명>(self, instance)`의 형태로 메서드를 생성할 수 있습니다.

데이터가 조회될 때, `get_<필드명>` 메서드가 실행되며 `instance`에는 직렬화 중인 모델 인스턴스가 전달됩니다.

예제의 메서드는 데이터베이스의 `age` 필드가 `30` 미만인 경우에는 `True`를 반환하며, `30` 이상인 경우에는 `False`를 반환합니다.

- Tip : `SerializerMethodField`는 **읽기 전용** 필드입니다. 응답에는 포함되지만, 요청 데이터로 값을 입력받지 않습니다.

<br>

### Meta

{% highlight python %}

class Meta:
    model = UserModel
    fields = "__all__"

{% endhighlight %}

`Meta` 클래스는 어떤 **모델**을 사용할지 정의하며, 해당 모델에서 어떤 **필드**를 사용할지 정의합니다.

`fields`의 값을 `"__all__"`로 사용하는 경우, 모델의 모든 필드를 사용합니다.

만약, 특정 필드만 사용한다면 `fields = ("email", "name", "age")` 등의 형태로 사용하려는 필드만 적용할 수 있습니다.

- Tip : `"__all__"`은 모델에 필드가 추가되면 자동으로 응답에도 노출되므로, 비밀번호처럼 민감한 필드가 있는 모델에서는 사용할 필드를 명시하는 것이 안전합니다.

<br>

### validate

{% highlight python %}

def validate_email(self, value):
    if "admin" in value:
        raise serializers.ValidationError(detail="사용할 수 없는 메일 계정입니다.")
    return value

def validate_name(self, value):
    if len(value) < 2:
        raise serializers.ValidationError(detail="이름이 올바르지 않습니다.")
    return value

def validate_age(self, value):
    if value < 19:
        raise serializers.ValidationError(detail="회원 가입이 불가능한 나이입니다.")
    return value

{% endhighlight %}

`validate_<필드명>`을 통해 특정 필드에 입력된 값에 대해 별도의 유효성 검사를 진행할 수 있습니다.

`value`에는 기본 유효성 검사를 통과한 해당 필드의 **입력값**이 전달되며, 검사가 끝나면 값을 반환해야 합니다.

`raise serializers.ValidationError(detail="오류 내용")`을 통해 유효성 오류를 발생시킬 수 있습니다.

`email` 필드에 **admin**이라는 문자열이 포함되어 있다면, 유효성 오류를 발생시킵니다.

`name` 필드의 글자수가 **두 글자 미만**이라면, 유효성 오류를 발생시킵니다.

`age` 필드가 **19 미만**이라면, 유효성 오류를 발생시킵니다.

- Tip : 메서드에서 값을 반환하지 않으면 해당 필드의 값이 `None`으로 저장되므로, 반드시 `return value`를 작성합니다.

<br>

{% highlight python %}

def validate(self, attrs):
    if attrs.get("name") == attrs.get("email"):
        raise serializers.ValidationError(detail="이름과 이메일이 같을 수 없습니다.")
    return attrs

{% endhighlight %}

여러 필드를 함께 비교해야 한다면, `validate(self, attrs)` 메서드를 정의합니다.

`attrs`는 필드별 유효성 검사를 통과한 값들의 **사전(Dictionary)**이며, 모든 `validate_<필드명>` 메서드가 실행된 뒤에 호출됩니다.

<br>
<br>

## 필드 매핑

### ID 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.AutoField | serializers.IntegerField |
| models.BigAutoField | serializers.BigIntegerField |
| models.UUIDField | serializers.UUIDField |

<br>

### 문자열 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.CharField | serializers.CharField |
| models.TextField | serializers.CharField |
| models.SlugField | serializers.SlugField |
| models.URLField | serializers.URLField |
| models.EmailField | serializers.EmailField |

<br>

### 데이터 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.BinaryField | serializers.ModelField |
| models.DecimalField | serializers.DecimalField |
| models.IntegerField | serializers.IntegerField |
| models.BigIntegerField | serializers.BigIntegerField |
| models.PositiveIntegerField | serializers.IntegerField |
| models.FloatField | serializers.FloatField |
| models.BooleanField | serializers.BooleanField |
| models.JSONField | serializers.JSONField |

<br>

### 날짜 및 시간 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.DateField | serializers.DateField |
| models.TimeField | serializers.TimeField |
| models.DateTimeField | serializers.DateTimeField |
| models.DurationField | serializers.DurationField |

<br>

### 기타 분야

| 모델 필드 | 매핑 필드 |
|:-:|:-:|
| models.ImageField | serializers.ImageField |
| models.FileField | serializers.FileField |
| models.FilePathField | serializers.FilePathField |

<br>

### 관계 분야

| 모델 필드 | 기본 매핑 필드 |
|:-:|:-:|
| models.OneToOneField | serializers.PrimaryKeyRelatedField |
| models.ForeignKey | serializers.PrimaryKeyRelatedField |
| models.ManyToManyField | serializers.PrimaryKeyRelatedField(many=True) |

<br>

`ModelSerializer`는 모델 필드를 위의 표와 같이 직렬화 필드로 매핑합니다.

예를 들어, `models.TextField`는 `serializers.CharField`로 매핑됩니다.

직렬화 필드에는 `TextField` 필드가 없으므로, 직렬화 필드를 직접 선언할 때는 위의 표에서 맞는 매핑을 찾아서 작성해야 합니다.

일반적으로 대부분이 모델 필드와 비슷하거나 동일한 형태의 구조를 갖고 있습니다.

`관계 분야` 필드는 기본적으로 연결된 객체의 **기본 키(Primary Key) 값**으로 표현됩니다.

연결된 객체의 내용 전체를 표현하려면 해당 모델의 `Serializer` 클래스를 생성해 필드로 지정하는 **중첩 직렬화(Nested Serializer)**를 사용합니다.

- Tip : 중첩 직렬화는 11강 `Foreign Key`에서 자세히 다룹니다.

직렬화 필드도 모델 필드처럼 옵션값이 존재합니다. 옵션값은 다음과 같습니다.

<br>
<br>

## 필드 옵션

| 옵션 | 설명 | 기본값 |
|:-:|------|:------:|
| default | 입력값이 없을 때 사용할 기본값을 설정합니다. | - |
| label | HTML Form 등에 표시될 문자열을 설정합니다. | - |
| help_text | 도움말 텍스트를 설정합니다. | - |
| read_only | 읽기 전용 필드로 설정합니다. (응답에만 포함) | False |
| write_only | 쓰기 전용 필드로 설정합니다. (요청에서만 사용) | False |
| required | 역직렬화 시 필수 입력 여부를 설정합니다. | True |
| allow_null | None 값을 허용합니다. | False |
| source | 값을 가져올 모델의 속성 이름을 설정합니다. | 필드 이름 |
| validators | 유효성 검사를 적용할 함수 목록을 등록합니다. | - |
| error_messages | 오류 메세지를 설정합니다. | - |

<br>
<br>

## 메타 옵션

메타(Meta) 클래스는 **직렬화 클래스**에서 사용할 수 있는 설정을 적용합니다.

<br>

| 옵션 | 설명 |
|:-:|------|
| model | 직렬화할 모델을 설정합니다. |
| fields | 직렬화에 포함할 필드를 설정합니다. |
| exclude | 직렬화에 제외할 필드를 설정합니다. |
| read_only_fields | 읽기 전용 필드를 설정합니다. |
| extra_kwargs | 자동으로 생성되는 필드에 추가 옵션을 설정합니다. |
| depth | 관계 필드를 중첩 표현할 깊이를 설정합니다. |

<br>

{% highlight python %}

class Meta:
    model = UserModel
    fields = ("id", "email", "name", "age")
    read_only_fields = ("id",)
    extra_kwargs = {
        "email": {"help_text": "로그인에 사용할 이메일"},
    }

{% endhighlight %}

`fields`와 `exclude`는 **둘 중 하나만** 사용해야 합니다.

`extra_kwargs`를 사용하면 필드를 다시 선언하지 않고도, 자동으로 생성되는 필드의 옵션만 변경할 수 있습니다.
