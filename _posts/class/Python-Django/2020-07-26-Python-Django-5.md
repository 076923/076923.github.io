---
layout: post
title: "Python Django 강좌 : 제 5강 - Model"
tagline: "Python Django Model"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Django']
keywords: Python, Python Django, Python Django 3, Python Django Model, Python Django model.py, Python Django Field, Python Django Field Options
ref: Python
category: posts
permalink: /posts/Python-Django-5/
comments: true
---

## Django Model ##
----------

장고에서 모델은 **데이터베이스의 구조(layout)**를 의미합니다.

`models.py` 파일에 하나 이상의 모델 클래스를 정의해 데이터베이스의 테이블을 정의할 수 있습니다

장고는 `ORM(Object-relational mapping)`을 사용해, 객체와 관계의 설정을 손쉽게 진행할 수 있습니다.

`ORM`은 서로 다른 **관계형 데이터베이스 관리 시스템(RDBMSs)**에서 필드를 스스로 매핑해 간단하게 데이터베이스를 구성할 수 있습니다.

즉, 복잡한 SQL문을 사용하지 않으며, 재사용 및 유지보수의 편리성이 증가합니다.

장고의 모델 파일(models.py)은 필드의 **인스턴스 이름 및 자료형 등을 정의합니다.**

<br>

## models.py ##
----------

{% highlight python %}

import uuid
from django.db import models

# Create your models here.
class UserModel(models.Model):
    id = models.UUIDField(help_text="Unique key", primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(help_text="User E-mail", blank=False, null=False)
    name = models.CharField(help_text="User Full Name", max_length=255, blank=False, null=False)
    age = models.PositiveIntegerField(help_text="User Age", blank=False, null=False)
    created_date = models.DateTimeField(help_text="Created Date time", auto_now_add=True)
    updated_date = models.DateTimeField(help_text="Updated Date time", auto_now=True)

    class Meta:
        verbose_name = "유저 정보"
        verbose_name_plural = "유저 정보"
        ordering = ["name", "age"]

{% endhighlight %}

모델(models.py) 파일을 위와 같이 정의합니다.

`UserModel` 클래스를 생성하며, `models.Model`을 상속합니다.

`UserModel` 클래스는 데이터베이스의 테이블 이름을 의미하며, 테이블에는 `first_app_usermodel`로 정의됩니다.

`id`는 데이터의 고유한 이름을 의미합니다. `색인(index)`값을 사용할수도 있지만 `UUID(universally unique identifier)`로 사용하도록 하겠습니다.

`email`은 유저의 이메일, `name`은 유저의 이름, `age`는 유저의 나이, `create_date`는 생성 날짜, `update_date`는 변경 날짜를 의미합니다.

각각의 필드는 `models.필드 타입`으로 정의할 수 있습니다.

필드 타입의 정의가 완료된 후, 각각의 필드에 옵션을 추가해 기본적인 필드 구성을 완료합니다.

마지막으로, `메타(Meta)`를 설정해 해당 데이터베이스의 테이블의 기본 메타데이터 정보를 설정합니다.

<br>

### 필드 타입 ###
----------

#### ID 분야

| 필드 타입 | 설명 |
|:-:|------|
| AutoField | 기본 키 필드이며, 자동적으로 증가하는 필드입니다. 주로 ID에 할당합니다. |
| BigAutoField | 1 ~ 9223372036854775807까지 1씩 자동으로 증가하는 필드입니다. |
| UUIDField | UUID 전용 필드이며, UUID 데이터 유형만 저장할 수 있습니다. |

#### 문자열 분야

| 필드 타입 | 설명 |
|:-:|------|
| CharField | 적은 문자열을 저장하는 문자열 필드입니다. |
| TextField | 많은 문자열을 저장하는 문자열 필드입니다. |
| URLField | URL 데이터를 저장하는 필드입니다. |
| EmailField | E-mail 데이터를 저장하는 필드입니다. |

#### 데이터 분야

| 필드 타입 | 설명 |
|:-:|------|
| BinaryField | 이진 데이터를 저장하는 필드입니다. |
| DecimalField | Decimal 데이터를 저장하는 필드입니다. |
| IntegerField | Interger 데이터를 저장하는 필드입니다. |
| PositiveIntegerField | 양수의 Interger 데이터를 저장하는 필드입니다. |
| FloatField | Float 데이터를 저장하는 필드입니다. |
| BooleanField | 참/거짓 데이터를 저장하는 필드입니다. |
| NullBooleanField | Null값이 가능한 참/거짓을 저장하는 필드입니다. |

#### 날짜 및 시간 분야

| 필드 타입 | 설명 |
|:-:|------|
| DateField | 날짜 데이터를 저장하는 필드입니다. |
| TimeField | 시간 데이터를 저장하는 필드입니다. |
| DateTimeField | 날짜와 시간 데이터를 저장하는 필드입니다. |

#### 기타 분야

| 필드 타입 | 설명 |
|:-:|------|
| ImageField | 이미지 데이터를 저장하는 필드입니다. |
| FileField | 파일 업로드 데이터를 저장하는 필드입니다. |
| FilePathField | 파일 경로 데이터를 저장하는 필드입니다. |

#### 관계 분야

| 필드 타입 | 설명 |
|:-:|------|
| OneToOneField | 일대일 관계를 저장하는 필드입니다. |
| ForeignKey | 일대다 관계를 저장하는 필드입니다. |
| ManyToManyField | 다대다 관계를 저장하는 필드입니다. |

<br>

각각의 필드는 기본적으로 규칙을 검사합니다.

예를 들어, `EmailField`는 `EmailValidator`가 기본적으로 적용되어 입력값에 `@` 등이 포함되어있는지 등을 확인합니다.

필드들은 기본적인 유효성 검사를 진행해 효율적인 모델을 구성할 수 있습니다.

또한, 각각의 필드는 필수 인수를 요구하기도 하는데, `CharField`는 `max_length` 인수를 필수값으로 요구합니다.

`max_length`는 문자열 필드의 최대 길이를 설정합니다.

기본적인 필드의 옵션값은 다음과 같습니다.

<br>

### 필드 옵션 ###
----------

| 옵션 | 설명 | 기본값 |
|:-:|------|:------:|
| default | 필드의 기본값을 설정합니다. | - |
| help_text | 도움말 텍스트를 설정합니다. | - |
| null | Null 값 허용 유/무를 설정합니다. | False |
| blank | 비어있는 값 허용 유/무를 설정합니다. | False |
| unique | 고유 키 유/무를 설정합니다. | False |
| primary_key | 기본 키 유/무를 설정합니다. (null=False, unique=True와 동일) | False |
| editable | 필드 수정 유/무를 설정합니다. | False |
| max_length | 필드의 최대 길이를 설정합니다. | - |
| auto_now | 개체가 저장될 때마다 값을 설정합니다. | False |
| auto_now_add | 개체가 처음 저장될 때 값을 설정합니다. | False |
| on_delete | 개체가 제거될 때의 동작을 설정합니다. | - |
| db_column | 데이터베이스의 컬럼의 이름을 설정합니다. | - |

<br>

### 메타 옵션 ###
----------

메타(Meta) 클래스는 모델 내부에서 사용할 수 있는 설정을 적용합니다.

정렬 순서나 관리 설정 등을 변경할 수 있습니다.

메타 클래스의 옵션은 다음과 같습니다.

<br>

| 옵션 | 설명 | 기본값 |
|:-:|------|:------:|
| abstract | 추상 클래스 유/무를 설정합니다. | False |
| db_table | 모델에 사용할 데이터베이스 테이블 이름을 설정합니다. | - |
| managed | 데이터베이스의 생성, 수정, 삭제 등의 권한을 설정합니다. | True |
| ordering | 객체를 가져올 때의 정렬 순서를 설정합니다. | - |
| verbose_name | 사람이 읽기 쉬운 객체의 이름을 설정합니다. (단수형으로 작성) | - |
| verbose_name_plural | 사람이 읽기 쉬운 객체의 이름을 설정합니다. (복수형으로 작성) | - |

<br>

여기서 가장 중요한 옵션을 고르자면, `managed`와 `ordering`으로 간주할 수 있습니다.

`managed` 옵션이 `True`일 경우 장고가 데이터베이스의 테이블을 마이그레이션 명령어를 통해 관리합니다.

만약, `managed` 옵션이 `False`일 경우 장고에서 데이터베이스의 테이블을 관리하지 않게 되어, 마이그레이션을 진행하지 않아도 됩니다.

즉, 미리 설계된 데이터베이스의 설정 및 제한 조건을 따라갑니다.

`ordering` 옵션은 데이터베이스의 객체 목록을 가져올 때 정렬 순서를 의미합니다.

예시의 `["name", "age"]`은 `name`과 `age`를 위주로 각각 오름차순으로 정렬합니다.

만약, `name`은 내림차순, `age`는 오름차순으로 정렬할 경우, `-`을 붙여 `["-name", "age"]`로 사용합니다.