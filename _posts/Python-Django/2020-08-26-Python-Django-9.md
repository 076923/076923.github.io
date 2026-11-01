---
layout: post
title: "Python Django 강좌 : 제 9강 - Migration"
tagline: "Python Django Migration"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Migration, Python Django Schema, Python Django python manage.py migrate, Python Django python manage.py showmigrations, Python Django python manage.py makemigrations, Python Django sqlmigrate, Python Django no such column
ref: Python-Django
category: Python
permalink: /posts/Python-Django-9/
comments: true
toc: true
---

## Django Migration

`마이그레이션(Migration)`이란 데이터베이스의 `스키마(Schema)`를 관리하기 위한 방법입니다.

사전적인 의미로는 현재 사용하고 있는 운영 환경을 다른 운영 환경으로 변환하는 작업을 지칭합니다.

장고에서는 모델의 생성, 변경, 삭제 등 **모델의 변경 사항을 데이터베이스 스키마에 반영**하는 과정을 의미합니다.

현재 **모델(models.py)**은 정의만 되어있을 뿐, 데이터베이스에 테이블을 생성하지 않았습니다.

마이그레이션을 통해 데이터베이스에 테이블을 생성하고 **모델의 생성, 변경, 삭제 등에 따라 작업 내역을 관리하고 데이터베이스를 최신화할 수 있습니다.**

4강에서 기본 마이그레이션을 진행했다면, `애플리케이션 마이그레이션`으로 이동해 진행합니다.

- Tip : 스키마(Schema)란 데이터베이스에서 자료의 구조, 자료 간의 관계 등을 기술한 것을 의미합니다.

<br>
<br>

## 기본 마이그레이션

{% highlight bash %}

python manage.py migrate

{% endhighlight %}

**결과**
:    

{% highlight text %}

Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying auth.0010_alter_group_name_max_length... OK
  Applying auth.0011_update_proxy_permissions... OK
  Applying auth.0012_alter_user_first_name_max_length... OK
  Applying sessions.0001_initial... OK

{% endhighlight %}

`python manage.py migrate`로 현재 프로젝트의 마이그레이션을 진행합니다.

마이그레이션 진행시, 장고 기본 앱에서 사용하는 테이블과 마이그레이션 기록용 테이블이 생성됩니다.

**sqlite3** 데이터베이스의 경우 생성되는 테이블은 다음과 같습니다.

`auth_group`, `auth_group_permissions`, `auth_permission`, `auth_user`, `auth_user_groups`, `auth_user_user_permissions`, `django_admin_log`, `django_content_type`, `django_migrations`, `django_session`의 10개 테이블과 SQLite의 자동 증가 값을 관리하는 `sqlite_sequence` 테이블이 생성됩니다.

- Tip : `django_migrations` 테이블에는 **어떤 마이그레이션이 적용되었는지** 기록됩니다. 장고는 이 테이블을 보고 적용할 마이그레이션을 결정합니다.

<br>
<br>

## 마이그레이션 상태 확인

{% highlight bash %}

python manage.py showmigrations

{% endhighlight %}

**결과**
:    

{% highlight text %}

admin
 [X] 0001_initial
 [X] 0002_logentry_remove_auto_add
 [X] 0003_logentry_add_action_flag_choices
auth
 [X] 0001_initial
 [X] 0002_alter_permission_name_max_length
 [X] 0003_alter_user_email_max_length
 [X] 0004_alter_user_username_opts
 [X] 0005_alter_user_last_login_null
 [X] 0006_require_contenttypes_0002
 [X] 0007_alter_validators_add_error_messages
 [X] 0008_alter_user_username_max_length
 [X] 0009_alter_user_last_name_max_length
 [X] 0010_alter_group_name_max_length
 [X] 0011_update_proxy_permissions
 [X] 0012_alter_user_first_name_max_length
contenttypes
 [X] 0001_initial
 [X] 0002_remove_content_type_name
first_app
 (no migrations)
sessions
 [X] 0001_initial

{% endhighlight %}

`python manage.py showmigrations` 명령어로 현재 마이그레이션 상태를 확인할 수 있습니다.

기본 앱의 마이그레이션은 모두 적용되었지만, `first_app`의 `모델(models.py)`에 대한 마이그레이션은 아직 생성되지 않은 것을 확인할 수 있습니다.

이제, 애플리케이션(앱)에서 생성한 모델에 대해 마이그레이션을 적용해보도록 하겠습니다.

<br>
<br>

## 애플리케이션 마이그레이션

{% highlight bash %}

python manage.py makemigrations first_app

{% endhighlight %}

**결과**
:    
{% highlight text %}

Migrations for 'first_app':
  first_app\migrations\0001_initial.py
    + Create model UserModel

{% endhighlight %}

`python manage.py makemigrations [앱 이름]`으로 모델에서 생성되거나 변경된 사항을 감지하여 **마이그레이션 파일**로 생성합니다.

단순하게 마이그레이션을 진행할 구조를 파일로 생성하는 것이므로, 데이터베이스에 적용되지는 않습니다.

`[앱 이름]`을 생략하면 `INSTALLED_APPS`에 등록된 모든 앱의 변경 사항을 감지합니다.

다시 `python manage.py showmigrations` 명령어를 통해 마이그레이션 상태를 확인할 경우, 다음과 같이 표시됩니다.

- Tip : 마이그레이션 파일은 데이터베이스 종류와 관계없는 **Python 코드**로 작성되며, 적용할 때 데이터베이스 종류에 맞는 SQL로 변환됩니다.

<br>

{% highlight text %}

admin
 [X] 0001_initial
 [X] 0002_logentry_remove_auto_add
 [X] 0003_logentry_add_action_flag_choices
auth
 [X] 0001_initial
 [X] 0002_alter_permission_name_max_length
 [X] 0003_alter_user_email_max_length
 [X] 0004_alter_user_username_opts
 [X] 0005_alter_user_last_login_null
 [X] 0006_require_contenttypes_0002
 [X] 0007_alter_validators_add_error_messages
 [X] 0008_alter_user_username_max_length
 [X] 0009_alter_user_last_name_max_length
 [X] 0010_alter_group_name_max_length
 [X] 0011_update_proxy_permissions
 [X] 0012_alter_user_first_name_max_length
contenttypes
 [X] 0001_initial
 [X] 0002_remove_content_type_name
first_app
 [ ] 0001_initial
sessions
 [X] 0001_initial

{% endhighlight %}

`first_app`에서 **(no migrations)**으로 표시되던 항목이 `[ ] 0001_initial`로 표시되는 것을 확인할 수 있습니다.

<br>

{% highlight bash %}

python manage.py sqlmigrate first_app 0001

{% endhighlight %}

`python manage.py sqlmigrate [앱 이름] [마이그레이션 번호]`로 마이그레이션이 실행할 **SQL 문**을 미리 확인할 수 있습니다.

실제로 데이터베이스에 적용하지는 않으므로, 어떤 테이블과 컬럼이 생성되는지 검토할 때 유용합니다.

이제 이 변경사항에 대해 마이그레이션을 진행해보도록 하겠습니다.

<br>

{% highlight bash %}

python manage.py migrate first_app

{% endhighlight %}

**결과**
:    
{% highlight text %}

Operations to perform:
  Apply all migrations: first_app
Running migrations:
  Applying first_app.0001_initial... OK

{% endhighlight %}

결과에서 확인할 수 있듯이, `first_app` 앱에 대한 마이그레이션이 적용되었습니다.

다시 `python manage.py showmigrations first_app` 명령어를 통해 마이그레이션 상태를 확인할 경우, 다음과 같이 표시됩니다.

<br>

{% highlight text %}

first_app
 [X] 0001_initial

{% endhighlight %}

`first_app`의 **0001_initial**이 적용된 것을 확인할 수 있습니다.

- Tip : `showmigrations` 뒤에 앱 이름을 작성하면 해당 앱의 마이그레이션만 표시됩니다.

`python manage.py migrate first_app` 명령어는 현재 적용되지 않은 마이그레이션을 적용하는 역할을 합니다.

마이그레이션이 정상적으로 적용될 경우, `앱 이름_소문자 클래스 이름`의 형태로 테이블이 생성됩니다.

예제를 기준으로 테이블의 이름을 확인한다면, `first_app_usermodel` 테이블이 생성됩니다.

정상적으로 마이그레이션이 완료되었다면, 프로젝트를 실행할 수 있습니다.

<br>
<br>

## 마이그레이션시 주의사항

마이그레이션을 진행할 때, `모델(models.py)`에서 하나라도 변경이 발생했다면 마이그레이션을 다시 진행해야 합니다.

모델 수정이 발생할 경우, 다음과 같은 절차로 마이그레이션을 적용할 수 있습니다.

<br>

{% highlight bash %}

python manage.py makemigrations [앱 이름]
python manage.py migrate [앱 이름]

{% endhighlight %}

특정 앱에 대해 마이그레이션 파일을 생성 후, 변경사항을 적용합니다.

기본값이 없는 **필수 필드(null=False)**를 기존 데이터가 있는 테이블에 추가하면, `makemigrations` 실행 시 다음과 같은 선택지가 표시됩니다.

**1) 기존 행에 채울 기본값을 지금 입력하거나, 2) 명령을 종료하고 모델에 `default`를 추가할 수 있습니다.**

단, 임의의 값으로 채울 때 올바르지 않은 값을 채운다면 데이터의 의미가 손상되는 `치명적인 오류`가 발생할 수 있습니다.

- Tip : 필드 이름을 변경하면 장고가 `Was usermodel.name renamed to usermodel.full_name?`처럼 **이름 변경 여부를 질문**합니다. `y`를 입력하지 않으면 기존 필드를 삭제하고 새 필드를 생성하므로 데이터가 사라집니다.

마이그레이션이 정상적으로 적용되었다면, 다음과 같은 파일 구조를 갖습니다.

<br>

```

[현재 프로젝트]/
  > 📁 [장고 프로젝트 이름]
  ⬇ 📁 [장고 앱 이름]
    > 📁 __pycache__
    ⬇ 📁 migrations
      > 📁 __pycache__
      🖹 __init__.py
      🖹 0001_initial.py
    🖹 __init__.py
    🖹 admin.py
    🖹 apps.py
    🖹 models.py
    🖹 serializers.py
    🖹 tests.py
    🖹 urls.py
    🖹 views.py
  🖹 db.sqlite3
  🖹 manage.py

```

<br>

마이그레이션 파일은 모델의 변경 내역을 누적하는 역할을 하므로, **소스 코드와 함께 Git 등의 버전 관리 시스템에 커밋**합니다.

여러 명이 각자 모델을 변경해 같은 번호의 마이그레이션 파일이 생기면 **마이그레이션 충돌**이 발생합니다. 이 경우 `python manage.py makemigrations --merge`로 병합 마이그레이션을 생성해 해결할 수 있습니다.

`help_text`, `verbose_name`처럼 데이터베이스 스키마에 영향을 주지 않는 변경도 마이그레이션 파일이 생성되며, 모델과 마이그레이션 내역을 일치시키기 위해 함께 생성하는 것을 권장합니다.

**이미 적용된 마이그레이션 파일은 임의로 수정하거나 제거하면 안됩니다.**

- Tip : `python manage.py makemigrations --check --dry-run` 명령어로 모델 변경 후 마이그레이션 파일을 만들지 않은 상태인지 확인할 수 있습니다. CI 등에서 활용합니다.

만약, **마이그레이션을 취소하거나 돌아가야하는 상황**이라면 다음과 같이 적용할 수 있습니다.

<br>

{% highlight bash %}

python manage.py migrate [앱 이름] 0001_initial

{% endhighlight %}

위의 명령어를 실행할 경우, `0001_initial`이 적용된 상태로 이동합니다.

현재 마이그레이션이 적용된 상태가 `0001_initial` 이전이라면, **정방향(forward)으로 마이그레이션이 진행됩니다.**

만약, 현재 마이그레이션이 적용된 상태가 `0001_initial` 이후라면, 순차적으로 지정된 마이그레이션까지 **역방향(backward)으로 마이그레이션이 진행됩니다.**

`마이그레이션을 초기화` 해야하는 경우에는 다음과 같이 실행할 수 있습니다.

<br>

{% highlight bash %}

python manage.py migrate [앱 이름] zero

{% endhighlight %}

현재 앱에 적용된 모든 마이그레이션을 **역방향으로 되돌립니다.** 해당 앱의 테이블이 삭제되며, 마이그레이션 파일은 그대로 남습니다.

마이그레이션은 각 파일에 작성된 **의존성(dependencies) 순서에 의해 진행됩니다.**

만약, `no such column`이나 `no such table` 오류가 발생한다면 모델의 변경 사항이 데이터베이스에 반영되지 않았다는 의미이므로, 마이그레이션을 생성하고 적용했는지 확인합니다.

<br>
<br>

## 데이터베이스 완전 초기화

개발 중인 프로젝트에서 데이터베이스를 삭제하고 완전하게 처음의 상태로 돌아가기 위해서는 다음과 같은 파일을 제거하면 처음 상태로 돌아갈 수 있습니다.

```

[현재 프로젝트]/
  ⬇ 📁 [장고 앱 이름]
    ⬇ 📁 migrations
      🖹 0001_initial.py
  🖹 db.sqlite3

```

위 구조에서 `[장고 앱 이름]/migrations/0001_initial.py`, `db.sqlite3`를 삭제합니다.

**마이그레이션 내역(0001_initial.py 등의 번호가 붙은 파일), 데이터베이스(db.sqlite3)**를 삭제한다면 초기 상태로 돌아갈 수 있습니다.

이때, `migrations` 폴더의 `__init__.py`는 **삭제하면 안 됩니다.** `__init__.py`가 없으면 장고가 `migrations` 폴더를 인식하지 못합니다.

위와 같은 파일을 제거할 경우, `기본 마이그레이션`부터 다시 진행해야 합니다.

- Tip : 이 방법은 **모든 데이터가 삭제**되므로, 개발 환경에서만 사용합니다. 운영 중인 데이터베이스는 마이그레이션 파일을 삭제하지 않고 새로운 마이그레이션으로 변경합니다.

- Tip : `__pycache__` 폴더는 Python이 생성하는 캐시 파일이며, 삭제하지 않아도 초기화에 영향을 주지 않습니다.
