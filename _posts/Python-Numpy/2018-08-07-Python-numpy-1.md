---
layout: post
title: "Python Numpy 강좌 : 제 1강 - 배열 생성 (1)"
tagline: "Python Numpy Array (1)"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy array
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-1/
comments: true
toc: true
---

## Numpy

`Numpy`는 `벡터 행렬` 계산을 효율적으로 처리하기 위한 모듈입니다.

`Numeric` 모듈과 `Numarray` 모듈을 통합하여 만들어졌으며, **높은 수준의 다차원 배열 계산**을 `고속 및 효율적`으로 처리할 수 있습니다. 

`numpy` 모듈은 `pip`를 통하여 설치할 수 있습니다.

- `numpy 설치하기` : [28강 바로가기][28강]

<br>
<br>

## Numpy 사용

{% highlight Python %}

import numpy

{% endhighlight %}

상단에 `import numpy`를 사용하여 `numpy`를 포함시킵니다.

numpy 함수의 사용방법은 `numpy.*`를 이용하여 사용이 가능합니다.

<br>

{% highlight Python %}

import numpy as np

{% endhighlight %}

`as` 구문이 추가될 경우, `numpy.*`에서 `np.*`으로 축약해서 사용할 수 있습니다.

<br>
<br>

## numpy 배열 생성

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

`print()`로 출력하면 `numpy` 배열은 `list`와 달리 원소 사이에 `쉼표(,)`가 표시되지 않습니다. 쉼표가 없는 것은 **출력 형식의 차이**일 뿐이며, 대화형 셸에서 변수만 입력하면 `array([1, 2, 3, 4, 5])`처럼 쉼표가 포함된 형태로 표시됩니다.

또한, `list` 형식으로 생성된 배열을 `numpy` 형식으로 변경할 수 있습니다. 반대로 `numpy 배열.tolist()`를 사용하면 `list`로 되돌릴 수 있습니다.

<br>
<br>

## numpy 배열 복제

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

`b = a`는 배열을 복사하는 것이 아니라, **같은 배열에 이름을 하나 더 붙이는 대입**입니다. `a`와 `b`는 메모리 상의 같은 배열을 가리키므로, `b`의 값을 변경하면 **원본 `a`의 값도 함께 변경됩니다.** 이 동작은 `list`의 대입과 동일하며, `b is a`의 결과도 `True`가 됩니다.

`numpy 배열.copy()`를 통하여 복제할 경우, **원본과 별개의 배열이 생성됩니다.** 원본을 보존해야 한다면 반드시 `copy()`를 사용합니다.

- Tip : `tuple`은 원소를 변경할 수 없는 `불변(immutable)` 자료형이므로, 대입 후 값을 바꾸는 상황 자체가 발생하지 않습니다.

- Tip : 배열을 `슬라이싱`한 결과는 새 배열이 아닌 원본의 `뷰(view)`이므로, 슬라이싱한 배열의 값을 변경해도 원본이 변경됩니다. 자세한 내용은 슬라이싱 강좌에서 다룹니다.

<br>
<br>

## numpy 배열 호출

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
<br>

## numpy 배열 계산

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

- Tip : `numpy`를 사용할 경우, `list comprehension`을 사용하지 않아도 원소의 값을 연산할 수 있습니다. 반복문 없이 배열 전체에 연산을 적용하는 방식을 `벡터화(vectorization)` 연산이라 하며, 내부적으로 C 언어로 처리되어 `list`를 반복문으로 계산하는 것보다 **훨씬 빠릅니다.**

<br>
<br>

## numpy 배열의 자료형

{% highlight Python %}

import numpy as np

a = np.array([1, 2.5, 3])
b = np.array([1, 2.5, "3"])
c = np.array([True, 2, 3])

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[1.  2.5 3. ]<br>
['1' '2.5' '3']<br>
[1 2 3]<br>
<br>

`list`는 서로 다른 자료형의 원소를 함께 담을 수 있지만, `numpy` 배열은 **모든 원소가 하나의 자료형**을 가집니다.

서로 다른 자료형이 섞여 있다면, 모든 원소를 표현할 수 있는 자료형으로 **자동 변환**됩니다.

정수와 실수가 섞이면 `실수`로, 문자열이 섞이면 `문자열`로, 논리값과 정수가 섞이면 `정수`로 변환됩니다.

- Tip : 문자열로 변환된 배열은 `b*2`와 같은 수치 연산을 수행할 수 없으므로, 배열을 생성할 때 원소의 자료형을 통일하는 것이 좋습니다.

[28강]: https://076923.github.io/posts/Python-28/
