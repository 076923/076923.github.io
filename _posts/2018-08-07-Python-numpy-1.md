---
bg: "numpy.png"
layout: post
comments: true
title: "Python numpy 강좌 : 제 1강 - 배열 생성 (1)"
crawlertitle: "Python numpy 강좌 : 제 1강 - 배열 생성 (1)"
summary: "Python numpy array (1)"
date: 2018-08-07
categories: posts
tags: ['Python-numpy']
author: 윤대희
star: true
---

### numpy ###
----------
`numpy`는 `벡터 행렬` 계산을 효율적으로 처리하기 위한 모듈입니다. `Numeric`모듈과 `Numarray` 모듈이 합쳐져 **높은 수준의 다차원 배열 계산**을 `고속 및 효율적`으로 처리할 수 있습니다. 

<br>

`numpy` 모듈은 `pip`를 통하여 설치할 수 있습니다.

`numpy 설치하기` : [28강 바로가기][28강]

<br>

<br>
<br>
### numpy 사용 ###
----------
{% highlight Python %}

import numpy

{% endhighlight %}

상단에 `import numpy`를 사용하여 `numpy`를 포함시킵니다. numpy 함수의 사용방법은 `numpy.*`를 이용하여 사용이 가능합니다.

<br>

{% highlight Python %}

import numpy as np

{% endhighlight %}

`as` 구문이 추가될 경우, `numpy.*`에서 `np.*`으로 축약해서 사용할 수 있습니다.

<br>
<br>
### numpy 배열 생성 ###
----------
{% highlight Python %}

import numpy as np

a = [1, 2, 3, 4, 5]
b = np.array(a)
c = np.array([1, 3, 5])

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[1, 2, 3, 4, 5]<br>
[1 2 3 4 5]<br>
[1 3 5]<br>

<br>

`numpy.array(배열)`을 사용하여 **numpy 배열**을 생성할 수 있습니다.

`numpy`는 `list`와 비슷하지만, 배열안에 `콤마(,)`가 존재하지 않습니다.

또한, `list` 형식으로 생성된 배열을 `numpy` 형식으로 변경할 수 있습니다.

<br>
### numpy 배열 복제 ###
----------
{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a
c = a.copy()

b[0] = 99

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[99  2  3  4  5]<br>
[99  2  3  4  5]<br>
[1 2 3 4 5]<br>

<br>

`numpy` 배열은 `list`, `tuple` 등과 동일하게 복사하여 값을 변경할 경우, **원본의 값도 변경됩니다.**

`numpy배열.copy()`를 통하여 복제할 경우, **원본과 별개의 배열이 생성됩니다.**

<br>
### numpy 배열 호출 ###
----------
{% highlight Python %}

import numpy as np

a = [1, 2, 3, 4, 5]
b = np.array(a)
c = np.array([1, 3, 5])

print(b[2])
print(c[-1])
print(c[0:2])

{% endhighlight %}

**결과**
:    
3<br>
5<br>
[1 3]<br>

<br>

`list`와 동일하게 배열 내의 원소를 `대괄호([])`를 이용하여 호출할 수 있습니다.

<br>
### numpy 배열 계산 ###
----------
{% highlight Python %}

import numpy as np

a = [1, 2, 3, 4, 5]
b = np.array(a)
c = np.array([1, 3, 5])

print(a*2)
print(b*2)
print(c+3)

{% endhighlight %}

**결과**
:    
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]<br>
[ 2  4  6  8 10]<br>
[4 6 8]<br>


<br>

`numpy`는 `list`와 다르게 `수학 기호`를 사용한다면 **각각의 원소로 계산하여 반환합니다.**

`list`의 경우 `*`기호 사용시 해당 배열을 이어붙이지만, `numpy`의 경우 **연산을 실행합니다.**

<br>

* Tip : `numpy`를 사용할 경우, `list comprehension`을 사용하지 않아도 원소의 값을 연산할 수 있습니다.


[28강]: https://076923.github.io/posts/Python-28/
