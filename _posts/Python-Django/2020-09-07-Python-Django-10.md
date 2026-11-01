---
layout: post
title: "Python Django 강좌 : 제 10강 - Test"
tagline: "Python Django Test & Debug"
image: /assets/images/django.png
tags: ['Django']
keywords: Python, Python Django, Python Django 6, Python Django Test, Python Django Debug, Python Django APITestCase, Python Django manage.py test, Python Django Requests, Python Django Requests Post, Python Django Requests Get, Python Django Requests Put, Python Django Requests Delete, Python Django Postman, Python Django Postman Get, Python Django Postman Post, Python Django Postman Put,  Python Django Postman Delete, Python Django Postman Content-Type, Python Django Postman JSON
ref: Python-Django
category: Python
permalink: /posts/Python-Django-10/
comments: true
toc: true
---

## Django Test

프로젝트의 설정 및 마이그레이션이 완료되면 프로그램이 정상적으로 구동되는지 **테스트**를 진행해야 합니다.

이 과정은 전체적인 실행에 문제가 없더라도, **프로그램이 의도한대로 작동이 되는지 확인**하는 과정도 포함됩니다.

장고 프로젝트를 테스트하는 방법은 크게 네 가지의 방법이 있습니다.

`DRF 웹 페이지(Browsable API)`, `requests` 코드, `포스트맨(Postman)`으로 직접 요청을 보내 확인하거나, 장고의 `테스트 프레임워크`로 테스트 코드를 작성해 자동으로 확인할 수 있습니다.

이 중, 두 가지 이상은 병행하여 테스트하는 것을 권장드립니다.

<br>
<br>

## Django Runserver

{% highlight bash %}

python manage.py runserver

{% endhighlight %}

**결과**
:    
Watching for file changes with StatReloader<br>
Performing system checks...<br>
<br>
System check identified no issues (0 silenced).<br>
November 01, 2026 - 21:40:12<br>
Django version 6.1.2, using settings 'daehee.settings'<br>
Starting WSGI development server at http://127.0.0.1:8000/<br>
Quit the server with CTRL-BREAK.<br>
<br>
WARNING: This is a development server. Do not use it in a production setting. Use a production WSGI or ASGI server instead.<br>
For more information on production servers see: https://docs.djangoproject.com/en/6.1/howto/deployment/<br>
<br>

먼저, 테스트를 진행하기 위해 서버를 실행시킵니다.

마이그레이션을 모두 적용했다면, 2강과 달리 마이그레이션 경고가 표시되지 않습니다.

만약 이 단계에서 오류가 발생한다면, `구문 오류(SyntaxError)`, `import 오류`, `설정 오류` 등이 발생했을 가능성이 높습니다.

오류 메세지의 **마지막 줄**과 **추적 정보(Traceback)**를 확인하면 오류가 발생한 파일과 줄 번호를 알 수 있습니다. 오류가 표시된 위치가 아닌, 그 파일을 불러온 다른 위치가 원인일 수도 있습니다.

위 단계에서 정상적으로 동작하더라도, 실제 요청에서 오류가 발생하거나 의도와 다른 결과가 나온다면 `논리 오류(logic error)`이므로 해당 기능을 다시 검토해야합니다.

- Tip : `python manage.py check` 명령어로 서버를 실행하지 않고도 설정과 모델의 문제를 점검할 수 있습니다.

<br>
<br>

## Browsable API Test

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/1.webp" class="lazyload" width="100%" height="100%"/>

DRF는 웹 브라우저로 접속했을 때 API를 직접 사용해볼 수 있는 **Browsable API** 페이지를 제공합니다.

`장고 프로젝트 설정 파일(settings.py)`에서 `INSTALLED_APPS`에 `rest_framework`를 등록했다면 원활하게 테스트를 진행할 수 있습니다.

`로컬 서버(http://127.0.0.1:8000/users/)`에 접속하면 위와 같은 화면을 확인할 수 있습니다.

각각의 필드에 `이메일`, `이름`, `나이`를 작성한 다음, `[POST]` 버튼을 클릭합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/2.webp" class="lazyload" width="100%" height="100%"/>

정상적인 값을 입력했다면, `View`에서 작성한 `{"message": "Operate successfully"}` 구문이 반환됩니다.

다음으로, 우측 상단의 `[GET]` 버튼을 클릭합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/3.webp" class="lazyload" width="100%" height="100%"/>

필드에 `이메일`, `이름`, `나이`만 작성했지만, 나머지 필드가 모두 입력되어 표시되는 것을 확인할 수 있습니다.

`id`는 `default=uuid.uuid4`로, `created_date`와 `updated_date`는 `auto_now_add`, `auto_now`로 자동 입력되며, `event_age`는 `SerializerMethodField`로 계산된 값입니다.

이번에도 `이메일`, `이름` 그리고 `나이`의 값을 19 미만으로 작성한 다음, `[POST]` 버튼을 클릭합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/4.webp" class="lazyload" width="100%" height="100%"/>

`age` 필드의 유효성 검사의 `value < 19` 조건으로 **회원 가입이 불가능한 나이입니다.**의 메세지가 반환됩니다.

오류가 발생하거나, 유효성 검사에 실패했을 경우 데이터베이스에 저장되지 않습니다.

이제 다시, 우측 상단의 `[GET]` 버튼을 클릭합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/5.webp" class="lazyload" width="100%" height="100%"/>

처음에 작성한 데이터의 `id` 필드에서 **835e1ca3-5383-4e2e-a051-fc2b8ad11f5a(UUID)**를 복사해 `URL`에 입력합니다.

즉, `http://127.0.0.1:8000/users/835e1ca3-5383-4e2e-a051-fc2b8ad11f5a`로 이동합니다.

값을 수정한 다음 우측 하단의 `[PUT]` 버튼으로 값을 수정할 수 있으며, 우측 상단의 `[DELETE]` 버튼으로 입력된 데이터를 삭제할 수 있습니다.

- Tip : Browsable API는 `DEBUG` 설정과 관계없이 동작합니다. 운영 환경에서 사용하지 않으려면 `REST_FRAMEWORK` 설정의 `DEFAULT_RENDERER_CLASSES`에서 `BrowsableAPIRenderer`를 제외합니다.

<br>
<br>

## Requests Test

{% highlight bash %}

pip install requests

{% endhighlight %}

`requests`는 Python에서 HTTP 요청을 보낼 수 있는 패키지입니다.

서버가 구동되고 있는 상태에서 **새로운 터미널**을 열어 아래의 코드를 실행시킵니다.

<br>

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/users/"

response = requests.get(url)

status_code = response.status_code
data = response.json()

print(status_code)
print(data)

{% endhighlight %}

**결과**
:    
200<br>
[{'id': '835e1ca3-5383-4e2e-a051-fc2b8ad11f5a', 'event_age': True, 'email': 's076923@gmail.com', 'name': '윤대희', 'age': 20, 'created_date': '2026-11-01T21:42:23.066156+09:00', 'updated_date': '2026-11-01T21:42:23.066156+09:00'}]<br>
<br>

앞선 예제에서 데이터를 삭제하지 않았다면, 위의 코드로 데이터를 가져올 수 있습니다.

`requests` 모듈을 활용해 `CRUD` 기능을 모두 활용할 수 있습니다.

`[GET]`은 `requests.get(url)`을 활용해 데이터를 가져옵니다.

`response.json()`은 응답 본문(JSON)을 Python의 `사전(Dictionary)`이나 `목록(List)`으로 변환합니다.

- Tip : 시간이 `+09:00`으로 표시되는 것은 3강에서 `TIME_ZONE = 'Asia/Seoul'`로 설정했기 때문입니다.

<br>

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/users/"
data = {
    "email" : "s076923@gmail.com",
    "name" : "윤대희",
    "age" : 21
}

response = requests.post(url, json=data)

status_code = response.status_code
data = response.json()

print(status_code)
print(data)

{% endhighlight %}

**결과**
:    
201<br>
{'message': 'Operate successfully'}<br>
<br>

`[POST]`는 `[GET]` 방식과 코드에서 큰 차이를 보이지는 않지만, 데이터를 입력해야 하므로 `사전(Dictionary)` 형식의 값을 생성합니다.

다음으로, `requests.get` 메서드가 아닌 `requests.post` 메서드의 `json` 인자에 `data` 변수를 입력합니다.

`json` 인자를 사용하면 데이터를 JSON 문자열로 변환하고, `Content-Type: application/json` 헤더를 자동으로 추가합니다.

다시 `[GET]` 메서드를 실행해보거나, Browsable API 페이지를 새로고침한다면 적용된 것을 확인할 수 있습니다.

- Tip : `data=data`로 전달하면 HTML 폼 형식(`application/x-www-form-urlencoded`)으로 전송됩니다. 단순한 값은 처리할 수 있지만, 중첩된 사전이나 목록은 표현할 수 없으므로 API 요청에는 `json`을 사용합니다.

<br>

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/users/835e1ca3-5383-4e2e-a051-fc2b8ad11f5a"
data = {
    "email" : "s076923@gmail.com",
    "name" : "윤대희",
    "age" : 22
}

response = requests.put(url, json=data)

status_code = response.status_code
data = response.json()

print(status_code)
print(data)

{% endhighlight %}

**결과**
:    
200<br>
{'message': 'Operate successfully'}<br>
<br>

`[PUT]`은 `[POST]`에서 `url`을 특정 `id`로 변경하고 메서드의 이름만 `post`에서 `put`으로 변경합니다.

`PUT`은 리소스 전체를 수정하므로, 필수 필드를 모두 입력한 다음 명령을 실행합니다.

뷰의 `update()` 메서드에서 `status.HTTP_200_OK`를 반환하므로 `200`이 출력됩니다.

<br>

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/users/835e1ca3-5383-4e2e-a051-fc2b8ad11f5a"
data = {
    "age" : 23
}

response = requests.patch(url, json=data)

print(response.status_code)
print(response.json())

{% endhighlight %}

**결과**
:    
200<br>
{'message': 'Operate successfully'}<br>
<br>

`[PATCH]`는 수정할 필드만 전달해 **일부만 수정**합니다.

6강에서 `update()` 메서드가 `partial` 인수를 받도록 작성했으므로 정상적으로 처리됩니다.

<br>

{% highlight python %}

import requests

url = "http://127.0.0.1:8000/users/835e1ca3-5383-4e2e-a051-fc2b8ad11f5a"

response = requests.delete(url)

status_code = response.status_code

print(status_code)

{% endhighlight %}

**결과**
:    
204<br>
<br>

`[DELETE]`는 `[GET]`과 같이 요청 데이터를 사용하지 않으므로 작성하지 않습니다.

또한, `204 No Content`는 응답 본문이 없으므로, `response.json()`을 호출하지 않습니다.

<br>
<br>

## Postman Test

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/6.webp" class="lazyload" width="100%" height="100%"/>

`포스트맨(Postman)` 애플리케이션을 이용하여 `API` 테스트를 진행할 수 있습니다.

[다운로드 링크](https://www.postman.com/downloads/)를 통하여 설치를 진행합니다.

- Tip : 아래 이미지는 이전 버전의 화면이므로, 현재 버전과 버튼의 위치나 디자인이 다를 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/7.webp" class="lazyload" width="100%" height="100%"/>

정상적으로 설치가 완료된 후, `+` 모양의 탭 추가 버튼을 클릭합니다.

새로운 탭을 추가한다면, 다음과 같은 형태로 탭이 추가됩니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/8.webp" class="lazyload" width="100%" height="100%"/>

위와 같은 `Untitled Request` 탭이 추가됩니다.

이 탭을 통하여 요청할 API의 `Method`, `URL`, `Authorization`, `Headers`, `Body` 등을 설정할 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/9.webp" class="lazyload" width="100%" height="100%"/>

메서드는 **리스트 박스**를 이용해 설정할 수 있습니다.

URL은 리스트 박스 옆 **텍스트 박스**에 입력합니다.

`GET` 메서드와 함께, `http://127.0.0.1:8000/users/` URL을 입력합니다.

그 다음으로, `[Send]` 버튼을 클릭합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/10.webp" class="lazyload" width="100%" height="100%"/>

위 이미지와 같이 하단의 `Body`란에 반환된 데이터가 표시됩니다.

동일하게 `[POST]`, `[PUT]`, `[PATCH]`, `[DELETE]` 등을 진행할 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/11.webp" class="lazyload" width="100%" height="100%"/>

`[POST]` 메서드 이용시, 상단의 `Body` 탭에서 필드의 값들을 채워준 다음, `[Send]` 버튼을 클릭합니다.

`[PUT]`이나 `[DELETE]`도 비슷한 방식으로 진행됩니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/Django/lecture-10/12.webp" class="lazyload" width="100%" height="100%"/>

만약, `Body` 탭을 코드와 비슷한 형태로 작성하려면 `form-data`가 아닌 `raw`에서 작성합니다.

`raw`에서 작성하는 경우, 입력하는 데이터가 **JSON 형식**임을 알려야 합니다.

`raw` 오른쪽의 형식 목록에서 `JSON`을 선택하면 `Content-Type: application/json` 헤더가 자동으로 추가됩니다.

직접 추가하려면 `Headers` 탭에서 `KEY`에 `Content-Type`, `VALUE`에 `application/json`을 입력합니다.

이후, `raw`에서 작성이 완료됐다면 `[Send]` 버튼을 클릭해 요청할 수 있습니다.

<br>
<br>

## Unit Test

앞선 방법들은 서버를 실행한 상태에서 **사람이 직접 요청을 보내고 결과를 확인**하는 방법입니다.

기능이 많아지면 코드를 수정할 때마다 모든 기능을 직접 확인하기 어려우므로, **테스트 코드**를 작성해 자동으로 확인하는 것이 좋습니다.

장고는 Python의 `unittest`를 기반으로 한 테스트 프레임워크를 제공하며, DRF는 API 테스트를 위한 `APITestCase`를 제공합니다.

앱의 `tests.py`에 아래와 같이 테스트 코드를 작성합니다.

<br>

{% highlight python %}

## first_app/tests.py
from rest_framework import status
from rest_framework.test import APITestCase

from first_app.models import UserModel


class UserAPITest(APITestCase):
    def test_create_user(self):
        data = {"email": "s076923@gmail.com", "name": "윤대희", "age": 20}
        response = self.client.post("/users/", data, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(UserModel.objects.count(), 1)

    def test_create_user_under_age(self):
        data = {"email": "s076923@gmail.com", "name": "윤대희", "age": 18}
        response = self.client.post("/users/", data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("age", response.data["message"])
        self.assertEqual(UserModel.objects.count(), 0)

    def test_retrieve_not_found(self):
        response = self.client.get("/users/00000000-0000-0000-0000-000000000000")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

{% endhighlight %}

`APITestCase`를 상속받은 클래스에서 `test`로 시작하는 메서드가 각각 하나의 테스트가 됩니다.

`self.client`는 서버를 실행하지 않고도 요청을 보낼 수 있는 **테스트 클라이언트**입니다. `format="json"`으로 JSON 요청을 보냅니다.

`assertEqual`, `assertIn` 등의 메서드로 **기대한 결과와 실제 결과가 같은지** 확인합니다.

<br>

{% highlight bash %}

python manage.py test first_app

{% endhighlight %}

**결과**
:    
Found 3 test(s).<br>
Creating test database for alias 'default'...<br>
System check identified no issues (0 silenced).<br>
...<br>
----------------------------------------------------------------------<br>
Ran 3 tests in 0.032s<br>
<br>
OK<br>
Destroying test database for alias 'default'...<br>
<br>

`python manage.py test [앱 이름]`으로 테스트를 실행합니다.

테스트는 **별도의 테스트용 데이터베이스**를 생성해 실행하고, 테스트가 끝나면 삭제하므로 실제 데이터에 영향을 주지 않습니다.

또한, 각 테스트 메서드가 끝날 때마다 데이터가 되돌려지므로, 테스트끼리 서로 영향을 주지 않습니다.

테스트가 실패하면 `FAIL`과 함께 어떤 값이 기대와 달랐는지 출력됩니다.

- Tip : 테스트 데이터베이스에도 마이그레이션이 적용되므로, 마이그레이션 파일이 없는 모델은 테스트에서 사용할 수 없습니다.
