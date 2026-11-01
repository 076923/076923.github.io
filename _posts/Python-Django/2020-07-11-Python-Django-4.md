---
layout: post
title: "Python Django 강좌 : 제 4강 - 애플리케이션 생성"
tagline: "Python Django Start App"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django startapp, Python Django INSTALLED_APPS, Python Django AppConfig, Python Django migrate, Python Django showmigrations
ref: Python-Django
category: Python
permalink: /posts/Python-Django-4/
comments: true
toc: true
---

## Django Start Application

장고에서 `앱(App)`은 특정 기능을 담당하는 **웹 애플리케이션(Web Application)** 단위입니다.

앱에는 **모델(model)**, **템플릿(template)**, **뷰(view)**를 포함하고 있으며, 여러 앱이 프로젝트를 구성하게 됩니다.

`프로젝트(Project)`는 설정과 여러 앱을 묶은 **웹 사이트 전체**이며, `앱(App)`은 게시판, 회원 관리처럼 **하나의 기능**을 담당합니다.

하나의 앱은 여러 프로젝트에서 재사용할 수 있으며, 기능의 경계가 명확하다면 앱을 나누고 그렇지 않다면 하나의 앱으로 사용합니다.

앱의 이름은 프로젝트 구성에서 중복되지 않아야 합니다.

<br>

{% highlight bash %}

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
    🖹 views.py
  🖹 manage.py

```
<br>

`[장고 앱 이름]` : **python manage.py startapp**으로 생성한 장고 앱 이름입니다. 앱 실행을 위한 패키지가 생성됩니다.

`migrations` : 모델(model)에 대한 마이그레이션(migrations) 내역을 저장합니다.

`__init__.py` : 해당 폴더를 패키지로 인식합니다.

`admin.py` : 해당 앱의 모델을 **관리자 페이지(admin)**에 등록합니다.

`apps.py` : 해당 앱의 **설정 클래스(AppConfig)**를 정의합니다. 앱의 이름, 표시 이름 등을 설정합니다.

`models.py` : 데이터베이스의 필드 및 데이터를 관리합니다. **MTV 패턴** 중 `모델(Model)`을 의미합니다.

`tests.py` : 앱의 **단위 테스트** 코드를 작성하는 파일입니다.

`views.py` : 요청을 받아 모델의 정보를 가공하고 응답을 반환합니다. **MTV 패턴** 중 `뷰(View)`를 의미합니다.

<br>

기본적으로 위의 디렉터리 및 파일을 지원합니다.

별도로 `템플릿(templates)` 디렉터리, 앱에서 URL을 관리할 수 있도록 `urls.py`를 생성하기도 합니다.

DRF를 사용한다면 데이터 변환 및 유효성 검사를 위한 `serializers.py` 등을 생성합니다.

앱은 `models.py`, `views.py`, `serializers.py`, `urls.py`, `templates` 등을 위주로 코드를 구현합니다.

일반적으로 `__init__.py`, `apps.py`는 거의 수정하지 않습니다.

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

앱 이름을 등록하면 장고는 `apps.py`에 정의된 `AppConfig` 클래스(`FirstAppConfig`)를 자동으로 찾아 사용합니다.

<br>

{% highlight Python %}

## first_app/apps.py
from django.apps import AppConfig


class FirstAppConfig(AppConfig):
    name = 'first_app'

{% endhighlight %}

`name`은 앱의 Python 경로입니다. `verbose_name = "첫 번째 앱"`처럼 관리자 페이지에 표시될 이름을 추가할 수도 있습니다.

- Tip : `'first_app.apps.FirstAppConfig'`처럼 설정 클래스의 경로를 직접 등록해도 됩니다.

- Tip : 만약, 앱의 이름을 변경해야 한다면 **앱 내부의 모든 설정**, **INSTALLED_APPS**, **import 경로**, **마이그레이션 내역**을 모두 바꾸어야 하므로, 처음부터 신중하게 이름을 정합니다.

<br>
<br>

## Django migrate

{% highlight bash %}

python manage.py migrate

{% endhighlight %}

일반적으로 `Model` 클래스의 설계가 완료된 후, 모델에 대응되는 테이블을 데이터베이스에서 생성합니다.

하지만, 직접 작성한 모델 외에도 장고의 기본 앱(`admin`, `auth`, `contenttypes`, `sessions`)이 사용하는 테이블이 필요합니다.

먼저, `python manage.py migrate`를 실행해 기본 앱의 테이블을 생성합니다.

<br>

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

정상적으로 마이그레이션이 진행되면, 위와 같은 메세지가 출력됩니다.

모델이 변경될 때마다 마이그레이션을 진행해야 데이터베이스에 정상적으로 적용됩니다.

<br>

{% highlight bash %}

python manage.py showmigrations

{% endhighlight %}

마이그레이션이 정상적으로 적용됐는지 확인합니다.

정상적으로 마이그레이션이 적용됐다면, `[X]`로 마이그레이션이 되었다고 표시됩니다.

만약, 마이그레이션이 적용되지 않았다면, `[ ]`로 마이그레이션이 적용되지 않았다고 표시됩니다.

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
 (no migrations)
sessions
 [X] 0001_initial

{% endhighlight %}

`first_app`은 아직 모델을 작성하지 않았으므로 `(no migrations)`로 표시됩니다.

앱의 모델에 대한 마이그레이션은 5강에서 모델을 작성한 다음, 9강에서 자세히 다룹니다.
