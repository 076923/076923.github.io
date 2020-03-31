---
layout: post
title: "Python Numpy 강좌 : 제 17강 - 다항식 계산 (2)"
tagline: "Python Numpy Poly & Roots (1)"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy polyadd, Numpy polysub, Numpy polymul, Numpy polydiv
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-17/
comments: true
---

## 다항식 사칙연산 ##
----------

{% highlight Python %}

import numpy as np

p1 = np.poly([1, -1])
p2 = np.poly1d([1, 0, 1])
add = np.polyadd(p1, p2)

print(p1)
print(p2)
print(add)

{% endhighlight %}

**결과**
:    
[ 1.  0. -1.]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
1 x + 1<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
2 x<br>

<br>

`다항식 사칙연산`은 `입력 배열`을 기준으로 연산을 진행합니다.

`입력 배열`은 `목록(List)`, `배열(Numpy)`, `다항식(poly, poly1d)` 등으로 선언할 수 있습니다.

입력 배열을 다항식으로 간주하여, 연산을 진행합니다.

`덧셈(np.polyadd)`, `뺄셈(np.polysub)`, `곱셉(np.polymul)`, `나눗셈(np.polydiv)`로 연산이 가능합니다.

`np.함수명(입력 배열1, 입력 배열2)`를 통해 결과를 계산합니다.

단, `나눗셈(np.polydiv)`은 반환 형식이 **튜플(Tuple)** 형태로 `(몫, 나머지)`로 반환합니다.

나눗셈은 `입력 배열1 = 입력 배열2 * 몫 + 나머지`의 형태가 됩니다.

<br>
<br>

## 다항식 적분(Poly Integral) ##
----------

{% highlight Python %}

import numpy as np

p = np.poly1d([3, 2, 1])
integral = np.polyint(p, m = 1, k = 99)

print(p)
print(integral)

{% endhighlight %}

**결과**
:    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
3 x + 2 x + 1<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
1 x + 1 x + 1 x + 99<br>
<br>

`다항식 적분 함수(np.polyint)`는 **입력 배열을 다항식**으로 간주해 `부정적분(Indefinite Integral)`을 계산합니다.

`np.polyint(입력 배열, 반복 횟수, 상수)`로 다항식을 **부정적분**합니다.

`입력 배열`에 대해 `반복 횟수(m)`만큼 **반복적분(Iterated Integrals)**을 진행합니다.

`상수(k)`는 부정적분을 했을 때 생기는 **적분상수(Integral Constant)**를 의미합니다.

적분 함수를 수식으로 나타낼 경우, 다음과 같이 표현할 수 있습니다.

$$ {d^m\over dx^m} P(x) = p(x) $$

$$ P(x) = {k_{m-1}\over 0!}x^0 + \dots + {k_0\over (m-1)!}x^{m-1} $$

<br>
<br>

## 다항식 미분(Poly Derivative) ##
----------

{% highlight Python %}

import numpy as np

p = np.poly1d([3, 2, 1])
differential = np.polyder(p, m = 1)

print(p)
print(differential)

{% endhighlight %}

**결과**
:    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
3 x + 2 x + 1<br>
<br>
6 x + 2<br>
<br>

`다항식 미분 함수(np.polyder)`는 **입력 배열을 다항식**으로 간주해 `미분(Derivative)`을 계산합니다.

`np.polyder(입력 배열, 반복 횟수)`로 다항식을 **미분**합니다.

`입력 배열`에 대해 `반복 횟수(m)`만큼 **고계도함수(Higher Order Derivative)**을 진행합니다.

미분 함수를 수식으로 나타낼 경우, 다음과 같이 표현할 수 있습니다.

$$ P(x) = {d^n p(x)\over dx^n} $$

<br>
<br>
