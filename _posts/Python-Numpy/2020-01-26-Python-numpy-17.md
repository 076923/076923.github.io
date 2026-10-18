---
layout: post
title: "Python Numpy 강좌 : 제 17강 - 다항식 계산 (2)"
tagline: "Python Numpy Poly & Roots (2)"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy polyadd, Numpy polysub, Numpy polymul, Numpy polydiv, Numpy polyint, Numpy polyder, Numpy polyfit, Numpy Polynomial
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-17/
comments: true
toc: true
---

## 다항식 사칙연산

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

`입력 배열`은 `목록(List)`, `배열(Numpy)`, `다항식(poly1d)` 등으로 선언할 수 있습니다. `np.poly`는 다항식 객체가 아닌 **계수 배열**을 반환하므로, `p1`은 `배열(Numpy)` 형태의 입력입니다.

입력 중 하나라도 `poly1d`라면 결과도 `poly1d`로 반환되며, 모두 배열이라면 결과도 **계수 배열**로 반환됩니다.

입력 배열을 다항식으로 간주하여, 연산을 진행합니다.

`덧셈(np.polyadd)`, `뺄셈(np.polysub)`, `곱셈(np.polymul)`, `나눗셈(np.polydiv)`로 연산이 가능합니다.

`np.함수명(입력 배열1, 입력 배열2)`를 통해 결과를 계산합니다.

단, `나눗셈(np.polydiv)`은 반환 형식이 **튜플(Tuple)** 형태로 `(몫, 나머지)`로 반환합니다.

나눗셈은 `입력 배열1 = 입력 배열2 * 몫 + 나머지`의 형태가 됩니다.

<br>

{% highlight Python %}

import numpy as np

p1 = np.poly1d([1, 0, -1])
p2 = np.poly1d([1, 1])

print(np.polysub(p1, p2))
print(np.polymul(p1, p2))
print(np.polydiv(p1, p2))

{% endhighlight %}

**결과**
:    
&nbsp;&nbsp;&nbsp;2<br>
1 x - 1 x - 2<br>
&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
1 x + 1 x - 1 x - 1<br>
(poly1d([ 1., -1.]), poly1d([0.]))<br>
<br>

$$ (x^2 - 1) - (x + 1) = x^2 - x - 2 $$이며, $$ (x^2 - 1)(x + 1) = x^3 + x^2 - x - 1 $$입니다.

$$ x^2 - 1 = (x + 1)(x - 1) $$이므로, 나눗셈의 몫은 $$ x - 1 $$, 나머지는 $$ 0 $$이 됩니다.

<br>
<br>

## 다항식 적분(Poly Integral)

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

`입력 배열`에 대해 `반복 횟수(m)`만큼 **반복적분(Iterated Integrals)**을 진행합니다. 기본값은 `1`입니다.

`상수(k)`는 부정적분을 했을 때 생기는 **적분상수(Integral Constant)**를 의미합니다. 기본값은 모두 `0`입니다.

$$ 3x^2 + 2x + 1 $$을 적분하면 $$ x^3 + x^2 + x + C $$가 되며, 적분상수 $$ C $$에 `99`를 할당했으므로 결과는 $$ x^3 + x^2 + x + 99 $$가 됩니다.

적분 함수를 수식으로 나타낼 경우, `m`차 부정적분 $$ P(x) $$는 다음을 만족합니다.

$$ {d^m\over dx^m} P(x) = p(x) $$

`m`번 적분하면 `m`개의 적분상수가 생기며, 이 상수들은 $$ P(x) $$의 **낮은 차수 부분**을 다음과 같이 결정합니다.

$$ {k_{m-1}\over 0!}x^0 + {k_{m-2}\over 1!}x^1 + \dots + {k_0\over (m-1)!}x^{m-1} $$

즉, `상수(k)`는 **적분 순서대로** 입력하며, `m`이 `2` 이상이라면 `k`에 `m`개의 값을 갖는 목록을 입력합니다. `m`이 `1`이라면 스칼라 값 하나를 입력할 수 있습니다.

<br>
<br>

## 다항식 미분(Poly Derivative)

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

`입력 배열`에 대해 `반복 횟수(m)`만큼 미분해 **고계도함수(Higher Order Derivative)**를 계산합니다. 기본값은 `1`입니다.

$$ 3x^2 + 2x + 1 $$을 미분하면 $$ 6x + 2 $$가 됩니다.

미분 함수를 수식으로 나타낼 경우, 다음과 같이 표현할 수 있습니다.

$$ P(x) = {d^m p(x)\over dx^m} $$

<br>
<br>

## 다항식 근사(Polyfit)

{% highlight Python %}

import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = np.array([1.1, 2.9, 9.2, 19.1, 32.8])

coef = np.polyfit(x, y, 2)
p = np.poly1d(coef)

print(np.round(coef, 3))
print(np.round(p(5), 3))

{% endhighlight %}

**결과**
:    
[1.957 0.131 1.014]<br>
50.6<br>
<br>

`다항식 근사 함수(np.polyfit)`는 주어진 데이터에 가장 잘 맞는 다항식의 계수를 **최소제곱법(Least Squares)**으로 계산합니다.

`np.polyfit(x 좌표, y 좌표, 차수)`로 사용하며, 반환값은 **최고차항부터 내림차순**으로 정렬된 계수 배열입니다.

예시의 데이터는 $$ y \approx 1.957x^2 + 0.131x + 1.014 $$로 근사되며, 반환된 계수를 `np.poly1d`로 감싸면 $$ x = 5 $$일 때의 값인 `50.6`을 예측할 수 있습니다.

- Tip : 차수를 지나치게 높이면 주어진 점은 정확히 지나지만 그 사이와 바깥에서 크게 요동치는 **과적합(Overfitting)**이 발생합니다. 데이터의 개수보다 충분히 작은 차수를 사용해야 합니다.

<br>
<br>

## 다항식 클래스(Polynomial)

{% highlight Python %}

import numpy as np
from numpy.polynomial import Polynomial

p = Polynomial([1, 2, 3])

print(p.coef)
print(p(2))
print(p.deriv().coef)
print(p.integ(k=99).coef)
print(p.roots())

{% endhighlight %}

**결과**
:    
[1. 2. 3.]<br>
17.0<br>
[2. 6.]<br>
[99.  1.  1.  1.]<br>
[-0.33333333-0.47140452j -0.33333333+0.47140452j]<br>
<br>

`np.poly1d`와 `np.polyadd`, `np.polyint`, `np.polyder`, `np.polyfit` 등의 함수는 **레거시(Legacy) API**입니다.

공식 문서에서는 새로운 코드에 `numpy.polynomial` 패키지의 `Polynomial` 클래스를 사용할 것을 권장합니다.

`Polynomial` 클래스는 `np.poly1d`와 **계수의 순서가 반대**입니다. 계수를 **최저차항부터 오름차순**으로 입력하므로, `Polynomial([1, 2, 3])`은 $$ 1 + 2x + 3x^2 $$을 의미합니다.

`np.poly1d([3, 2, 1])`과 같은 다항식이며, 레거시 계수를 변환할 때는 `Polynomial(계수[::-1])`처럼 순서를 뒤집어야 합니다.

`p(2)`로 값을 계산하며, $$ 1 + 4 + 12 = 17 $$이 됩니다.

`deriv(m)`은 미분, `integ(m, k)`는 적분, `roots()`는 근을 계산합니다. 반환되는 계수 역시 **오름차순**이므로, 적분 결과 `[99, 1, 1, 1]`은 $$ 99 + x + x^2 + x^3 $$을 의미합니다.

`Polynomial` 객체끼리도 `+`, `-`, `*`, `//`, `%`, `**` 연산자로 사칙연산을 수행할 수 있습니다.

<br>
<br>

## 클래스를 이용한 다항식 근사

{% highlight Python %}

import numpy as np
from numpy.polynomial import Polynomial

x = np.array([0, 1, 2, 3, 4])
y = np.array([1.1, 2.9, 9.2, 19.1, 32.8])

p = Polynomial.fit(x, y, 2).convert()

print(np.round(p.coef, 3))
print(np.round(p(5), 3))

{% endhighlight %}

**결과**
:    
[1.014 0.131 1.957]<br>
50.6<br>
<br>

`Polynomial.fit(x 좌표, y 좌표, 차수)`는 `np.polyfit`을 대체하는 클래스 메서드입니다.

`Polynomial.fit`은 수치 안정성을 위해 내부적으로 `x` 좌표의 범위를 `[-1, 1]` 구간으로 **선형 변환**한 다음 근사합니다.

그러므로 반환된 객체의 `coef` 속성은 변환된 구간에 대한 계수입니다. 원래 `x` 좌표에 대한 계수가 필요하다면 `convert()` 메서드로 변환해야 합니다.

변환된 계수 `[1.014, 0.131, 1.957]`은 앞선 `np.polyfit`의 결과와 **순서만 반대**인 같은 값임을 확인할 수 있습니다.

- Tip : `convert()`를 호출하지 않아도 `p(5)`처럼 값을 계산하는 결과는 같습니다. 계수를 직접 읽거나 다른 프로그램에 전달하는 경우에만 `convert()`가 필요합니다.
