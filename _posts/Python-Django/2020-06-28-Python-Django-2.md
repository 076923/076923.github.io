---
layout: post
title: "Python Django 강좌 : 제 2강 - 프로젝트 생성"
tagline: "Python Django Project Creation"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django startproject, Python Django asgi.py, Python Django settings.py, Python Django urls.py, Python Django wsgi.py, Python Django manage.py
ref: Python-Django
category: Python
permalink: /posts/Python-Django-2/
comments: true
toc: true
---

## Django Project

장고(Django)를 원활하게 사용하기 위해선 기본 프로젝트 구성을 사용합니다.

기본 프로젝트를 사용할 폴더로 이동합니다.

<br>

{% highlight django %}

django-admin startproject daehee .

{% endhighlight %}

`django-admin startproject [프로젝트 이름]`을 통해 장고 기본 프로젝트 생성이 가능합니다.

만약, `django-admin startproject [프로젝트 이름] .`의 형태로 온점(.)을 추가한다면 현재 디렉토리에서 생성합니다.

정상적으로 프로젝트가 생성된다면 아래의 디렉토리 구조로 폴더와 파일이 생성됩니다.

<br>

```

[현재 프로젝트]/
  ⬇ 📁 [장고 프로젝트 이름]
    🖹 __init__.py
    🖹 asgi.py
    🖹 settings.py
    🖹 urls.py
    🖹 wsgi.py
  🖹 manage.py

```

`[장고 프로젝트 이름]` : **django-admin startproject**로 생성한 프로젝트 이름입니다. 프로젝트 실행을 위한 Python 패키지가 저장됩니다.

`__init__.py` : 해당 폴더를 패키지로 인식합니다.

`asgi.py` : 현재 프로젝트를 서비스하기 위한 **ASGI(Asynchronous Server Gateway Interface)** 호환 웹 서버 진입점입니다.

`settings.py` : 현재 Django **프로젝트의 환경 및 구성**을 설정합니다.

`urls.py` : 현재 Django **프로젝트의 URL**을 설정합니다.

`wsgi.py` : 현재 프로젝트를 서비스하기 위한 **WSGI(Web Server Gateway Interface)** 호환 웹 서버의 진입점입니다.

`manage.py` : 현재 Django를 서비스를 실행시키기 위한 **커맨드라인의 유틸리티**입니다.

<br>

현재 프로젝트의 파일에서 가장 중요한 요소는 `settings.py`, `urls.py`, `manage.py` 입니다.

위 세 가지 파일 중, `manage.py`를 제외하고 직접 수정해 변경합니다.

- Tip : 일반적으로 `manage.py`은 수정하지 않습니다.

<br>
<br>

## Django Test RunServer

먼저 간단하게 `manage.py`를 활용해 Django 서버를 구동해보겠습니다.

`manage.py`가 존재하는 디렉토리로 이동합니다.

디렉토리 이동은 `cd [장고 프로젝트 이름]` 등으로 이동할 수 있습니다.

<br>

{% highlight django %}

python manage.py runserver

{% endhighlight %}

**결과**
:    
Watching for file changes with StatReloader<br>
Performing system checks...<br>
<br>
System check identified no issues (0 silenced).<br>
<br>
You have 17 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.<br>
Run 'python manage.py migrate' to apply them.<br>
June 28, 2020 - 19:11:48<br>
Django version 3.0.7, using settings 'daehee.settings'<br>
Starting development server at http://127.0.0.1:8000/<br>
Quit the server with CTRL-BREAK.<br>
[28/Jun/2020 19:11:54] "GET / HTTP/1.1" 200 16351<br>
<br>

서버를 구동하게 되면 **파일 변경 사항**, **시스템 점검**, **마이그레이션 점검**, **설정 반영** 등을 통해 프로젝트를 실행시킵니다.

테스트 프로젝트는 `http://127.0.0.1:8000/`에서 확인할 수 있습니다.

해당 url로 이동해 서버가 정상적으로 구동되는지 확인합니다.

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-2/1.webp" class="lazyload" width="100%" height="100%"/>

서버를 종료하려면, `Ctrl + C`키를 눌러 서버를 종료할 수 있습니다.

현재 포트는 `8000`에 연결되어 있습니다. 만약 포트를 변경하거나 외부접속을 허용한다면 다음과 같이 문장을 추가합니다.

- Tip : 기본 포트는 `8000`번에 연결되어 있습니다.

<br>

{% highlight django %}

python manage.py runserver 8080
python manage.py runserver 0:8080
{% endhighlight %}

python manage.py runserver 뒤에 `8080`을 추가해 포트를 8080으로 변경할 수 있습니다.

python manage.py runserver 뒤에 `0:8080`을 추가해 외부 접속을 허용하고 포트를 8080으로 변경합니다.

- Tip : 모든 공용 IP 접속 허용은 `0`을 추가합니다. `0`은 `0.0.0.0`의 축약입니다.

- Tip : 외부 접속 허용시 `Settgins.py`에서 `ALLOWED_HOST=['*']`로 변경해야합니다.

- Tip : 마이그레이션 경고는 현재 사용하고 있는 데이터베이스에 반영되지 않아 나타나는 경고입니다.
