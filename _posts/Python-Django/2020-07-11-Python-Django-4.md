---
layout: post
title: "Python Django 강좌 : 제 4강 - 애플리케이션 생성"
tagline: "Python Django Start App"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django startapp, Python Django INSTALLED_APPS, Python Django migrate, Python Django showmigrations
ref: Python-Django
category: Python
permalink: /posts/Python-Django-4/
comments: true
toc: true
---

## Django Start Application

장고에서 `앱(App)`은 `시스템` 및 `데이터베이스` 등을 통해 서비스를 제공하는 **웹 애플리케이션(Web Application)**입니다.

앱에는 **모델(model)**, **템플릿(template)**, **뷰(view)**를 포함하고 있으며, 여러 앱이 프로젝트를 구성하게 됩니다.

프로젝트를 Python의 `클래스(class)`로 생각한다면, 앱은 `함수(function)`로 볼 수 있습니다.

앱은 재사용성 유/무로 앱의 개수가 결정되며, 재사용성이 없는 경우 하나의 앱으로 사용합니다.

앱은 하나의 **서비스**이며, 앱의 이름은 프로젝트 구성에서 중복되지 않아야 합니다.

<br>

{% highlight django %}

python manage.py startapp first_app

{% endhighlight %}

`python manage.py startapp [앱 이름]`을 통해 앱 생성이 가능합니다.

`manage.py` 파일을 통해 앱을 생성하므로, `manage.py` 파일이 존재하는 위치에서 명령어를 실행합니다.

정상적으로 앱이 생성된다면 아래의 디렉토리 구조로 폴더와 파일이 생성됩니다.

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
  🖹 manage.py

```
<br>

`[장고 앱 이름]` : **python manage.py startapp**로 생성한 장고 앱 이름입니다. 앱 실행을 위한 패키지가 생성됩니다.

`migrations` : 모델(model)에 대한 마이그레이션(migrations) 내역을 저장합니다.

`__init__.py` : 해당 폴더를 패키지로 인식합니다.

`admin.py` : 해당 앱에 대한 관리자 인터페이스를 등록합니다.

`apps.py` : 해당 앱의 경로를 설정합니다.

`models.py` : 데이터베이스의 필드 및 데이터를 관리합니다. **MVT 패턴** 중 `모델(Model)`을 의미합니다.

`tests.py` : 테스트를 위한 실행파일 입니다.

`view.py` : 모델의 정보를 받아 로직을 처리합니다. **MVT 패턴** 중 `뷰(View)`를 의미합니다.

<br>

기본적으로 위의 디렉터리 및 파일을 지원합니다.

별도로 `템플릿(template)` 디렉터리, 앱에서 URL을 관리할 수 있도록 `urls.py`을 생성하기도 합니다.

복잡한 로직이나 비지니스 로직을 위한 `serializer.py` 등을 생성할 수 있습니다. 

앱은 `models.py`, `view.py`, `serializer.py`, `urls.py`, `template` 등을 위주로 코드를 구현합니다.

일반적으로 `__init__.py`, `apps.py`, `tests.py`는 거의 수정하지 않습니다.

`migrations`의 폴더 내부에 생성될 파일들은 특별한 경우가 아닌 이상 인위적으로 수정하지 않습니다.

<br>
<br>

## Django Project 등록

{% highlight Python %}

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'first_app',
]

{% endhighlight %}

`장고 프로젝트 이름/settings.py`로 이동하여 `INSTALLED_APPS`에 생성한 앱 이름을 추가합니다.

앱이 추가될 때마다 `INSTALLED_APPS`에 앱 이름을 등록해야 합니다.

설치된 앱은 `apps.py`의 경로 설정을 따라갑니다.

- Tip : 만약, 앱의 이름을 변경해야 한다면 **앱 내부의 모든 설정** 및 **INSTALLED_APPS**의 설정을 모두 바꾸어야 합니다.

<br>
<br>

## Django migrate

{% highlight Python %}

python manage.py migrate

{% endhighlight %}

일반적으로 `Model` 클래스의 설계가 완료된 후, 모델에 대응되는 테이블을 데이터베이스에서 생성합니다.

하지만, 모델 클래스를 제외하고도 추가되어야하는 테이블이 존재합니다.

먼저, `python manage.py migrate`을 실행해 기본적인 구조를 적용하도록 합니다.

<br>

{% highlight Python %}

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

정상적으로 마이그레이션이 진행되면, 위와 같은 메세지가 띄워집니다.

데이터베이스의 설정이 변경될 때마다 마이그레이션을 진행해야 정상적으로 적용됩니다.

<br>

{% highlight Python %}

python manage.py showmigrations

{% endhighlight %}

마이그레이션이 정상적으로 적용됬는지 확인합니다.

정상적으로 마이그레이션이 적용됐다면, `[X]`로 마이그레이션이 되었다고 표시됩니다.

만약, 마이그레이션이 적용되지 않았다면, `[ ]`로 마이그레이션이 적용되지 않았다고 표시됩니다.

<br>

{% highlight Python %}

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
sessions
 [X] 0001_initial

{% endhighlight %}
