---
layout: post
title: "Python Django 강좌 : 제 10강 - Test"
tagline: "Python Django Test & Debug"
image: /assets/images/django.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Django']
keywords: Python, Python Django, Python Django 3, Python Django Test, Python Django Debug, Python Django Requests, Python Django Requests Post, Python Django Requests Get, Python Django Requests Put, Python Django Requests Delete, Python Django Postman, Python Django Postman Get, Python Django Postman Post, Python Django Postman Put,  Python Django Postman Delete, Python Django Postman Content-Type, Python Django Postman JSON
ref: Python-Django
category: Python
permalink: /posts/Python-Django-10/
comments: true
toc: true
---

## Django Test

프로젝트의 설정 및 마이그레이션이 완료되면 프로그램이 정상적으로 구동되는지 **테스트**를 진행해야 합니다.

이 과정은 전체적인 실행에 문제가 없더라도, **프로그램이 의도한대로 작동이 되는지 확인**하는 과정도 포함됩니다.

장고 프로젝트를 테스트하는 방법은 크게 세 가지의 방법이 있습니다.

이 중, 두 가지 이상은 병행하여 테스트하는 것을 권장드립니다.

<br>
<br>

## Django Runserver

{% highlight python %}

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
September 07, 2020 - 18:50:24<br>
Django version 3.0.7, using settings 'daehee.settings'<br>
Starting development server at http://127.0.0.1:8000/<br>
Quit the server with CTRL-BREAK.<br>
[28/Jun/2020 19:11:54] "GET / HTTP/1.1" 200 16351<br>
<br>

먼저, 테스트를 진행하기 위해 서버를 실행시킵니다.

만약 이 구문에서 오류가 발생한다면, `구문 오류(syntax error)`가 발생했을 가능성이 매우 높습니다.

오류가 발생한 위치에서 발생했을수도 있으나, 전혀 다른 위치에서 발생한 오류일 수도 있습니다.

위 구문에서 정상적으로 동작하더라도, 실제 테스트에서 오류가 발생한다면 `논리 오류(logic error)`이므로 전체적으로 다시 검토해야합니다.

<br>
<br>

## Django Test

![1]({{ site.images }}/assets/posts/Python/Django/lecture-10/1.webp){: width="100%" height="100%"}

`장고 프로젝트 설정 파일(settings.py)`에서 **DEBUG`를 True로 설정**했다면 원활하게 테스트를 진행할 수 있습니다.

`로컬 서버(http://127.0.0.1:8000/users)`에 접속하면 위와 같은 화면을 확인할 수 있습니다.

각각의 필드에 `이메일`, `이름`, `나이`를 작성한 다음, `[POST]` 버튼을 클릭합니다.

<br>

![2]({{ site.images }}/assets/posts/Python/Django/lecture-10/2.webp){: width="100%" height="100%"}

정상적인 값을 입력했다면, `View`에서 작성한 `{"message": "Operate successfully"}` 구문이 반환됩니다.

다음으로, 우측 상단의 `[GET]` 버튼을 클릭합니다.

<br>

![3]({{ site.images }}/assets/posts/Python/Django/lecture-10/3.webp){: width="100%" height="100%"}

필드에 `이메일`, `이름`, `나이`만 작성했지만, 나머지 필드가 모두 입력되어 표시되는 것을 확인할 수 있습니다.

이번에도 `이메일`, `이름` 그리고 `나이`의 값을 20미만으로 작성한 다음, `[POST]` 버튼을 클릭합니다.

<br>

![4]({{ site.images }}/assets/posts/Python/Django/lecture-10/4.webp){: width="100%" height="100%"}

`age` 필드의 유효성 검사의 `instance < 19` 조건으로 **회원 가입이 불가능한 나이입니다.**의 메세지가 반환됩니다. 

오류가 발생하거나, 유효성 검사에 실패했을 경우 데이터베이스에 저장되지 않습니다.

이제 다시, 우측 상단의 `[GET]` 버튼을 클릭합니다.

<br>

![5]({{ site.images }}/assets/posts/Python/Django/lecture-10/5.webp){: width="100%" height="100%"}

처음에 작성한 데이터의 `id` 필드에서 **835e1ca3-5383-4e2e-a051-fc2b8ad11f5a(UUID)**를 복사해 `URL`에 입력합니다.

즉, `http://127.0.0.1:8000/users/835e1ca3-5383-4e2e-a051-fc2b8ad11f5a`로 이동합니다.

값을 수정한 다음 우측 하단의 `[PUT]` 버튼으로 값을 수정할 수 있으며, 우측 상단의 `[DELETE]` 버튼으로 입력된 데이터를 삭제할 수 있습니다.

<br>
<br>

## Requests Test

{% highlight python %}

import json
import requests

url = "http://127.0.0.1:8000/users"

response = requests.get(url)

status_code = response.status_code
data = json.loads(response.text)

print(status_code)
print(data)

{% endhighlight %}

**결과**
:    
200<br>
[{'id': '835e1ca3-5383-4e2e-a051-fc2b8ad11f5a', 'event_age': True, 'email': 's076923@gmail.com', 'name': '윤대희', 'age': 20, 'created_date': '2020-09-07T18:59:23.066156+09:00', 'updated_date': '2020-09-07T18:59:23.066156+09:00'}]<br>
<br>

서버가 구동되고 있는 상태에서 새로운 코드 프로젝트를 실행시킵니다.

앞선 예제에서 데이터를 삭제하지 않았다면, 위의 코드로 데이터를 가져올 수 있습니다.

`requests` 모듈을 활용해 `CRUD` 기능을 모두 활용할 수 있습니다.

`[GET]`은 `requests.get(url)`을 활용해 데이터를 가져옵니다.

<br>

{% highlight python %}

import json
import requests

url = "http://127.0.0.1:8000/users"
data = {
    "email" : "s076923@gmail.com",
    "name" : "윤대희",
    "age" : 21
}

response = requests.post(url, data=data)

status_code = response.status_code
data = json.loads(response.text)

print(status_code)
print(data)

{% endhighlight %}

**결과**
:    
201<br>
{'message': 'Operate successfully'}<br>
<br>

`[POST]`는 `[GET]` 방식과 코드에서 큰 차이를 보이지는 않지만, 데이터를 입력해야 하므로 `사전(Dictionary)` 형식의 값을 생성합니다.

다음으로, `requests.get` 메서드가 아닌 `requests.post` 메서드에 `data` 인자에 `data` 변수를 입력합니다.

다시 `[GET]` 메서드를 실행해보거나, 장고 테스트를 실행한다면 적용된 것을 확인할 수 있습니다.

<br>

{% highlight python %}

import json
import requests

url = "http://127.0.0.1:8000/users/835e1ca3-5383-4e2e-a051-fc2b8ad11f5a"
data = {
    "email" : "s076923@gmail.com",
    "name" : "윤대희",
    "age" : 22
}

response = requests.put(url, data=data)

status_code = response.status_code
data = json.loads(response.text)

print(status_code)
print(data)

{% endhighlight %}

**결과**
:    
201<br>
{'message': 'Operate successfully'}<br>
<br>

`[PUT]`은 `[POST]`에서 `url`을 특정 `id`로 변경하고 메서드의 이름만 `post`에서 `put`으로 변경합니다.

`data` 필드는 모두 필수 필드이기 때문에, 모두 입력한 다음 명령을 실행합니다.

<br>

{% highlight python %}

import json
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

`[DELETE]`는 `[GET]`과 같이 데이터를 사용하지 않으므로 작성하지 않습니다.

또한, 반환값이 없으므로, `response.text`를 읽지 않습니다.

<br>
<br>

## Postman Test

![6]({{ site.images }}/assets/posts/Python/Django/lecture-10/6.webp){: width="100%" height="100%"}

`포스트맨(Postman)` 애플리케이션을 이용하여 `API` 테스트를 진행할 수 있습니다.

[다운로드 링크](https://www.postman.com/downloads/)를 통하여 설치를 진행합니다.

<br>

![7]({{ site.images }}/assets/posts/Python/Django/lecture-10/7.webp){: width="100%" height="100%"}

정상적으로 설치가 완료된 후, `+` 모양의 탭 추가 버튼을 클릭합니다.

새로운 탭을 추가한다면, 다음과 같은 형태로 탭이 추가됩니다.

<br>

![8]({{ site.images }}/assets/posts/Python/Django/lecture-10/8.webp){: width="100%" height="100%"}

위와 같은 `Untitled Request` 탭이 추가됩니다.

이 탭을 통하여 요청할 API의 `Method`, `URL`, `Authorization`, `Headers`, `Body` 등을 설정할 수 있습니다.

<br>

![9]({{ site.images }}/assets/posts/Python/Django/lecture-10/9.webp){: width="100%" height="100%"}

메서드는 **리스트 박스**를 이용해 설정할 수 있습니다.

URL은 리스트 박스 옆 **텍스트 박스**에 입력합니다.

`GET` 메서드와 함께, `http://127.0.0.1:8000/users` URL을 입력합니다.

그 다음으로, `[SEND]` 버튼을 클릭합니다.

<br>

![10]({{ site.images }}/assets/posts/Python/Django/lecture-10/10.png)

위 이미지와 같이 하단의 `Body`란에 반환된 데이터가 표시됩니다.

동일하게 `[POST], [PUT], [UPDATE]` 등을 진행할 수 있습니다.

<br>

![11]({{ site.images }}/assets/posts/Python/Django/lecture-10/11.png)

`[POST]` 메서드 이용시, 상단의 `Body` 탭에서 필드의 값들을 채워준 다음, `[Send]` 버튼을 클릭합니다.

`[PUT]`이나 `[DELETE]`도 비슷한 방식으로 진행됩니다.

<br>

![12]({{ site.images }}/assets/posts/Python/Django/lecture-10/12.png)

만약, `Body` 탭을 코드와 비슷한 형태로 작성하려면 `form-data`가 아닌 `raw`에서 작성합니다.

`raw`에서 작성하는 경우, 입력하는 데이터가 **json 형식**임을 알려야 합니다.

그러므로, `Headers` 탭에서 `Content-Type`을 추가합니다.

우측 `VAULE` 란에 `application/json`을 입력합니다.

이후, `raw`에서 작성이 완료됬다면 `[Send]` 버튼을 클릭해 요청할 수 있습니다.
