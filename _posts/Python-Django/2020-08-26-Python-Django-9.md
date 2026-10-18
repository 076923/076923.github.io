---
layout: post
title: "Python Django 강좌 : 제 9강 - Migration"
tagline: "Python Django Migration"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django Migration, Python Django Schema, Python Django python manage.py migrate, Python Django python manage.py showmigrations, Python Django python manage.py makemigrations, Python Django no such column, Python Django 
ref: Python-Django
category: Python
permalink: /posts/Python-Django-9/
comments: true
toc: true
---

## Django Migration

`마이그레이션(Migration)`이란 데이터베이스의 `스키마(Schema)`를 관리하기 위한 방법입니다.

사전적인 의미로는 현재 사용하고 있는 운영 환경을 다른 운영 환경으로 변환하는 작업을 지칭합니다.

데이터베이스에서는 스키마를 비롯해 테이블, 필드 등의 변경이 발생했을 때 지정된 **데이터베이스에 적용**하는 과정을 의미합니다. 

현재 **모델(model.py)**은 정의만 되어있을 뿐, 데이터베이스를 생성하고 적용하지 않았습니다.

마이그레이션을 통해 데이터베이스를 생성하고 **모델의 생성, 변경, 삭제 등에 따라 작업 내역을 관리하고 데이터베이스를 최신화할 수 있습니다.**

4강에서 기본 마이그레이션을 진행했다면, `애플리케이션 마이그레이션`으로 이동해 진행합니다.

- Tip : 스키마(Schema)란 데이터베이스에서 자료의 구조, 자료 간의 관계 등을 기술한 것을 의미합니다.

<br>
<br>

## 기본 마이그레이션

{% highlight python %}

python manage.py migrate

{% endhighlight %}

**결과**
:    

{% highlight python %}

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
  Applying sessions.0001_initial... OK

{% endhighlight %}

`python manage.py migrate`으로 현재 프로젝트의 마이그레이션을 진행합니다.

마이그레이션 진행시, 장고 프로젝트에서 사용하는 11개의 기본 테이블이 생성됩니다.

**sqlite3** 데이터베이스의 경우 생성되는 테이블은 다음과 같습니다.

`auth_group`, `auth_group_permissions`, `auth_permission`, `auth_user`, `auth_user_groups`, `auth_user_user_permissions`, `django_admin_log`, `django_content_type`, `django_migrations`, `django_session`, `sqlite_sequence`의 기본 테이블이 생성됩니다.

<br>
<br>

## 마이그레이션 상태 확인

{% highlight python %}

python manage.py showmigrations

{% endhighlight %}

**결과**
:    

{% highlight python %}

admin
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
contenttypes
 [X] 0001_initial
 [X] 0002_remove_content_type_name
first_app
 (no migrations)
sessions
 [X] 0001_initial

{% endhighlight %}

`python manage.py showmigrations` 명령어로 현재 마이그레이션 상태를 확인할 수 있습니다.

기본적인 마이그레이션으로 11개의 테이블이 생성되었지만, `모델(model.py)`에서 생성한 테이블을 생성되지 않은 것을 확인할 수 있습니다.

이제, 애플리케이션(앱)에서 생성한 모델에 대해 마이그레이션을 적용해보도록 하겠습니다.

<br>
<br>

## 애플리케이션 마이그레이션

{% highlight python %}

python manage.py makemigrations first_app

{% endhighlight %}

**결과**
:    
{% highlight python %}

Migrations for 'first_app':
  first_app\migrations\0001_initial.py
    - Create model UserModel

{% endhighlight %}

`python manage.py makemigrations [앱 이름]`으로 모델에서 생성한 사항이나, 변경 사항된 사항을 감지하여 파일로 생성합니다.

단순하게 마이그레이션을 진행할 구조를 생성하는 것이므로, 적용은 되지는 않습니다.

다시 `python manage.py showmigrations` 명령어를 통해 마이그레이션 상태를 확인할 경우, 다음과 같이 표시됩니다.

- Tip : 데이터베이스 종류에 따라 다른 SQL이 생성됩니다.

<br>

{% highlight python %}

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
contenttypes
 [X] 0001_initial
 [X] 0002_remove_content_type_name
first_app
 [ ] 0001_initial
sessions
 [X] 0001_initial

{% endhighlight %}

`first_app`에서 **(no migrations)**으로 표시되던 항목이 `[ ] 0001_initial`로 표시되는 것을 확인할 수 있습니다.

이 변경사항에 대해 마이그레이션을 진행해보도록 하겠습니다.

<br>

{% highlight python %}

python manage.py migrate first_app

{% endhighlight %}

**결과**
:    
{% highlight python %}

Operations to perform:
  Apply all migrations: first_app
Running migrations:
  Applying first_app.0001_initial... OK

{% endhighlight %}

결과에서 확인할 수 있듯이, `first_app` 앱에 대한 마이그레이션이 적용되었습니다.

다시 `python manage.py showmigrations` 명령어를 통해 마이그레이션 상태를 확인할 경우, 다음과 같이 표시됩니다.

<br>

{% highlight python %}

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
contenttypes
 [X] 0001_initial
 [X] 0002_remove_content_type_name
first_app
 [X] 0001_initial
sessions
 [X] 0001_initial

{% endhighlight %}

`first_app`의 **0001_initial**이 적용된 것을 확인할 수 있습니다.

`python manage.py migrate first_app` 명령어는 현재 적용되지 않은 마이그레이션을 적용하는 역할을 합니다.

마이그레이션이 정상적으로 적용될 경우, `앱_클래스명`의 형태로 테이블이 생성됩니다.

예제를 기준으로 테이블의 이름을 확인한다면, `first_app_usermodel` 테이블이 생성됩니다.

정상적으로 마이그레이션이 완료되었다면, 프로젝트를 실행할 수 있습니다.

<br>
<br>

## 마이그레이션시 주의사항

마이그레이션을 진행할 때, `모델(model.py)`에서 하나라도 변경이 발생했다면 마이그레이션을 다시 진행해야 합니다.

모델 수정이 발생할 경우, 다음과 같은 절차로 마이그레이션을 적용할 수 있습니다.

<br>

{% highlight python %}

python manage.py makemigrations [앱 이름]
python manage.py migrate [앱 이름]

{% endhighlight %}

특정 앱에 대해 마이그레이션 파일을 생성 후, 모든 변경사항을 적용합니다.

모델 마이그레이션 진행 시, 경고 문구가 발생한다면 필수 필드가 생성되었지만 **기본값이 할당되어 있지 않아서 발생하는 문제입니다.**

**임의의 값을 모두 채워주거나, 취소하여 건너 뛸 수 있습니다.**

단, 임의의 값으로 채울 때 올바르지 않은 값을 채운다면 `치명적인 오류`가 발생할 수 있습니다.

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
    🖹 view.py
  🖹 db.sqlite3
  🖹 manage.py

```

<br>

마이그레이션은 `Git`과 다르므로, **마이그레이션은 한 명만 진행**하는 것이 좋습니다.

만약, 여러 명이 작업하게 된다면 데이터베이스가 꼬이는 주된 원인이 됩니다.

마이그레이션은 데이터베이스 스키마에 변화를 발생시키지 않더라도 수행하는 것을 권장합니다.

마이그레이션은 모델의 변경 내역을 누적하는 역할을 하며, **적용된 마이그레이션 파일은 제거하면 안됩니다.**

만약, **마이그레이션을 취소하거나 돌아가야하는 상황**이라면 다음과 같이 적용할 수 있습니다.

<br>

{% highlight python %}

python manage.py migrate [앱 이름] 0001_initial

{% endhighlight %}

위의 명령어를 실행할 경우, `0001_initial`의 상태로 되돌아갑니다.

현재 마이그레이션이 적용된 상태가 `0001_initial` 이전이라면, **정방향(forward)으로 마이그레이션이 진행됩니다.**

만약, 현재 마이그레이션이 적용된 상태가 `0001_initial` 이후라면, 순차적으로 지정된 마이그레이션까지 **역방향(backward)으로 마이그레이션이 진행됩니다.**

`마이그레이션을 초기화` 해야하는 경우에는 다음과 같이 실행할 수 있습니다.

<br>

{% highlight python %}

python manage.py migrate [앱 이름] zero

{% endhighlight %}

현재 앱에 적용된 모든 마이그레이션을 삭제합니다. 

마이그레이션은 **디펜던시(dependencies) 순서에 의해 진행됩니다.**

만약, `no such column` 오류 발생시 마이그레이션이 진행되지 않았다는 의미가 됩니다.

<br>
<br>

## 데이터베이스 완전 초기화

데이터베이스를 삭제하고 완전하게 처음의 상태로 돌아가기 위해서는 다음과 같은 파일을 제거하면 처음 상태로 돌아갈 수 있습니다.

```

[현재 프로젝트]/
  ⬇ 📁 [장고 프로젝트 이름]
    > 📁 __pycache__
  ⬇ 📁 [장고 앱 이름]
    > 📁 __pycache__
    ⬇ 📁 migrations
      > 📁 __pycache__
      🖹 0001_initial.py
  🖹 db.sqlite3

```

위 구조에서 `[장고 프로젝트 이름]/__pycache__`, `[장고 앱 이름]/__pycache__`, `[장고 앱 이름]/migrations/__pycache__`, `[장고 앱 이름]/migrations/0001_initial.py`, `db.sqlite3`을 삭제합니다.

**모든 캐시 파일(\_\_pycache\_\_), 마이그레이션 내역(0001_initial.py), 데이터베이스(db.sqlite3)**를 삭제한다면 초기 상태로 돌아갈 수 있습니다.

위와 같은 파일을 제거할 경우, `기본 마이그레이션`부터 다시 진행하셔야 합니다.
