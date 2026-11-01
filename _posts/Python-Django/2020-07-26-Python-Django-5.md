---
layout: post
title: "Python Django 강좌 : 제 5강 - Model"
tagline: "Python Django Model"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Model, Python Django models.py, Python Django Field, Python Django Field Options, Python Django Meta, Python Django JSONField
ref: Python-Django
category: Python
permalink: /posts/Python-Django-5/
comments: true
toc: true
---

## Django Model

장고에서 모델은 **데이터베이스의 구조(layout)**를 의미합니다.

`models.py` 파일에 하나 이상의 모델 클래스를 정의해 데이터베이스의 테이블을 정의할 수 있습니다.

장고는 `ORM(Object-Relational Mapping)`을 사용해, 객체와 관계의 설정을 손쉽게 진행할 수 있습니다.

`ORM`은 Python의 **클래스와 객체**를 **관계형 데이터베이스(RDBMS)의 테이블과 행**에 대응시켜, 데이터베이스 종류에 맞는 SQL을 자동으로 생성합니다.

즉, 복잡한 SQL문을 직접 작성하지 않아도 되며, 데이터베이스를 변경하더라도 코드를 거의 수정하지 않아 재사용 및 유지보수의 편리성이 증가합니다.

장고의 모델 파일(models.py)은 필드의 **이름 및 자료형, 제약 조건 등을 정의합니다.**

<br>
<br>

## models.py

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

`UserModel` 클래스는 데이터베이스의 테이블을 의미하며, 테이블의 이름은 `앱 이름_소문자 클래스 이름` 규칙에 따라 `first_app_usermodel`로 정의됩니다.

`id`는 데이터를 구분하는 고유한 값인 `기본 키(Primary Key)`를 의미합니다. 자동으로 증가하는 정수를 사용할 수도 있지만 `UUID(Universally Unique Identifier)`를 사용하도록 하겠습니다.

`email`은 유저의 이메일, `name`은 유저의 이름, `age`는 유저의 나이, `created_date`는 생성 날짜, `updated_date`는 변경 날짜를 의미합니다.

각각의 필드는 `models.필드 타입`으로 정의할 수 있습니다.

필드 타입의 정의가 완료된 후, 각각의 필드에 옵션을 추가해 기본적인 필드 구성을 완료합니다.

마지막으로, `메타(Meta)`를 설정해 해당 데이터베이스의 테이블의 기본 메타데이터 정보를 설정합니다.

- Tip : 기본 키를 직접 선언하지 않으면 장고가 `id` 필드를 자동으로 추가합니다. Django 6.0부터 자동으로 추가되는 기본 키의 기본 타입은 `BigAutoField`이며, `DEFAULT_AUTO_FIELD` 설정으로 변경할 수 있습니다.

- Tip : `default=uuid.uuid4`처럼 **함수 자체**를 전달해야 합니다. `uuid.uuid4()`처럼 호출 결과를 전달하면 모든 행이 같은 값을 갖게 됩니다.

<br>
<br>

## 필드 타입

### ID 분야

| 필드 타입 | 설명 |
|:-:|------|
| AutoField | 1부터 1씩 자동으로 증가하는 정수 필드입니다. (32비트) |
| BigAutoField | 1 ~ 9223372036854775807까지 1씩 자동으로 증가하는 정수 필드입니다. (64비트) |
| UUIDField | UUID 전용 필드이며, UUID 데이터 유형만 저장할 수 있습니다. |

<br>

### 문자열 분야

| 필드 타입 | 설명 |
|:-:|------|
| CharField | 길이 제한이 있는 문자열을 저장하는 문자열 필드입니다. |
| TextField | 길이 제한이 없는 긴 문자열을 저장하는 문자열 필드입니다. |
| SlugField | 문자, 숫자, 밑줄, 하이픈으로 구성된 짧은 문자열(URL 등)을 저장하는 필드입니다. |
| URLField | URL 데이터를 저장하는 필드입니다. |
| EmailField | E-mail 데이터를 저장하는 필드입니다. |

<br>

### 데이터 분야

| 필드 타입 | 설명 |
|:-:|------|
| BinaryField | 이진 데이터를 저장하는 필드입니다. |
| DecimalField | 고정 소수점(Decimal) 데이터를 저장하는 필드입니다. |
| IntegerField | 정수(Integer) 데이터를 저장하는 필드입니다. |
| BigIntegerField | 64비트 정수 데이터를 저장하는 필드입니다. |
| PositiveIntegerField | 0 또는 양수의 정수 데이터를 저장하는 필드입니다. |
| FloatField | 부동 소수점(Float) 데이터를 저장하는 필드입니다. |
| BooleanField | 참/거짓 데이터를 저장하는 필드입니다. `null=True`로 Null 값을 허용할 수 있습니다. |
| JSONField | JSON 형식의 데이터를 저장하는 필드입니다. |

- Tip : 예전 버전의 `NullBooleanField`는 Django 4.0에서 제거되었습니다. 대신 `BooleanField(null=True)`를 사용합니다.

<br>

### 날짜 및 시간 분야

| 필드 타입 | 설명 |
|:-:|------|
| DateField | 날짜 데이터를 저장하는 필드입니다. |
| TimeField | 시간 데이터를 저장하는 필드입니다. |
| DateTimeField | 날짜와 시간 데이터를 저장하는 필드입니다. |
| DurationField | 기간(timedelta) 데이터를 저장하는 필드입니다. |

<br>

### 기타 분야

| 필드 타입 | 설명 |
|:-:|------|
| ImageField | 업로드한 이미지 파일의 경로를 저장하는 필드입니다. (Pillow 패키지 필요) |
| FileField | 업로드한 파일의 경로를 저장하는 필드입니다. |
| FilePathField | 서버의 특정 디렉터리에 있는 파일 경로를 선택해 저장하는 필드입니다. |
| GeneratedField | 다른 필드의 값으로 데이터베이스가 계산하는 필드입니다. |

<br>

### 관계 분야

| 필드 타입 | 설명 |
|:-:|------|
| OneToOneField | 일대일 관계를 저장하는 필드입니다. |
| ForeignKey | 다대일 관계를 저장하는 필드입니다. |
| ManyToManyField | 다대다 관계를 저장하는 필드입니다. |

<br>

각각의 필드는 기본적으로 규칙을 검사합니다.

예를 들어, `EmailField`는 `EmailValidator`가 기본적으로 적용되어 입력값에 `@` 등이 포함되어있는지 등을 확인합니다.

필드들은 기본적인 유효성 검사를 진행해 효율적인 모델을 구성할 수 있습니다.

또한, 각각의 필드는 필수 인수를 요구하기도 하는데, `CharField`는 `max_length` 인수를 필수값으로 요구했습니다.

`max_length`는 문자열 필드의 최대 길이를 설정합니다.

- Tip : `PostgreSQL`처럼 길이 제한 없는 문자열을 지원하는 데이터베이스에서는 `CharField`의 `max_length`를 생략할 수 있습니다. 데이터베이스 호환성을 위해 지정하는 것을 권장합니다.

- Tip : `FileField`, `ImageField`는 파일 자체가 아닌 **파일의 경로**를 데이터베이스에 저장하며, 실제 파일은 `MEDIA_ROOT` 등의 저장소에 저장됩니다.

기본적인 필드의 옵션값은 다음과 같습니다.

<br>
<br>

## 필드 옵션

| 옵션 | 설명 | 기본값 |
|:-:|------|:------:|
| default | 필드의 기본값을 설정합니다. | - |
| db_default | 데이터베이스 수준의 기본값을 설정합니다. | - |
| help_text | 도움말 텍스트를 설정합니다. | - |
| verbose_name | 사람이 읽기 쉬운 필드의 이름을 설정합니다. | - |
| null | 데이터베이스에 NULL 값 저장 허용 유/무를 설정합니다. | False |
| blank | 유효성 검사에서 비어있는 값 허용 유/무를 설정합니다. | False |
| choices | 선택 가능한 값의 목록을 설정합니다. | - |
| unique | 고유 값 유/무를 설정합니다. | False |
| db_index | 데이터베이스 인덱스 생성 유/무를 설정합니다. | False |
| primary_key | 기본 키 유/무를 설정합니다. (null=False, unique=True를 의미) | False |
| editable | 관리자 페이지나 폼, 직렬화 클래스에서 필드 수정 유/무를 설정합니다. | True |
| max_length | 필드의 최대 길이를 설정합니다. | - |
| auto_now | 개체가 저장될 때마다 현재 시간으로 값을 설정합니다. | False |
| auto_now_add | 개체가 처음 생성될 때 현재 시간으로 값을 설정합니다. | False |
| on_delete | 참조하는 개체가 삭제될 때의 동작을 설정합니다. (관계 필드 전용, 필수) | - |
| db_column | 데이터베이스의 컬럼의 이름을 설정합니다. | - |

<br>

`null`은 **데이터베이스**에 관한 옵션이며, `blank`는 **유효성 검사**에 관한 옵션입니다.

문자열 필드(`CharField`, `TextField`)는 빈 값을 `NULL` 대신 **빈 문자열('')**로 저장하는 것이 장고의 관례이므로, 일반적으로 `null=True`를 사용하지 않습니다.

- Tip : `editable=False`인 필드나 `auto_now`, `auto_now_add`가 적용된 필드는 관리자 페이지와 `ModelSerializer`에서 **읽기 전용**으로 처리됩니다.

<br>
<br>

## 메타 옵션

메타(Meta) 클래스는 모델 내부에서 사용할 수 있는 설정을 적용합니다.

정렬 순서나 관리 설정 등을 변경할 수 있습니다.

메타 클래스의 옵션은 다음과 같습니다.

<br>

| 옵션 | 설명 | 기본값 |
|:-:|------|:------:|
| abstract | 추상 클래스 유/무를 설정합니다. | False |
| db_table | 모델에 사용할 데이터베이스 테이블의 이름을 설정합니다. | - |
| managed | 마이그레이션으로 테이블의 생성, 수정, 삭제를 관리할지 설정합니다. | True |
| ordering | 객체를 가져올 때의 기본 정렬 순서를 설정합니다. | - |
| verbose_name | 사람이 읽기 쉬운 객체의 이름을 설정합니다. (단수형으로 작성) | - |
| verbose_name_plural | 사람이 읽기 쉬운 객체의 이름을 설정합니다. (복수형으로 작성) | - |
| indexes | 테이블에 생성할 인덱스 목록을 설정합니다. | - |
| constraints | 테이블에 적용할 제약 조건(UniqueConstraint, CheckConstraint 등) 목록을 설정합니다. | - |

<br>

여기서 가장 중요한 옵션을 고르자면, `managed`와 `ordering`으로 간주할 수 있습니다.

`managed` 옵션이 `True`일 경우 장고가 데이터베이스의 테이블을 마이그레이션 명령어를 통해 관리합니다.

만약, `managed` 옵션이 `False`일 경우 장고에서 데이터베이스의 테이블을 생성하거나 수정하지 않습니다.

즉, 이미 존재하는 테이블을 사용하는 경우에 활용하며, 미리 설계된 데이터베이스의 설정 및 제한 조건을 따라갑니다.

`ordering` 옵션은 데이터베이스의 객체 목록을 가져올 때 정렬 순서를 의미합니다.

예시의 `["name", "age"]`는 `name`을 기준으로 오름차순 정렬하고, `name`이 같다면 `age`를 기준으로 오름차순 정렬합니다.

만약, `name`은 내림차순, `age`는 오름차순으로 정렬할 경우, `-`을 붙여 `["-name", "age"]`로 사용합니다.

- Tip : `ordering`은 모든 조회에 정렬(ORDER BY)을 추가하므로, 데이터가 많은 테이블에서는 성능에 영향을 줄 수 있습니다.

<br>
<br>

## 모델 메서드

{% highlight python %}

class UserModel(models.Model):
    ...

    def __str__(self):
        return f"{self.name} ({self.email})"

{% endhighlight %}

`__str__` 메서드를 정의하면, 관리자 페이지나 셸(shell)에서 객체가 `UserModel object (...)` 대신 사람이 읽기 쉬운 문자열로 표시됩니다.

모델에는 필드 외에도 이처럼 **객체 단위의 로직**을 메서드로 추가할 수 있습니다.
