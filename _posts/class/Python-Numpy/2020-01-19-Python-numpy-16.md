---
layout: post
title: "Python Numpy 강좌 : 제 16강 - 다항식 계산 (1)"
tagline: "Python Numpy Poly & Roots (1)"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy poly1d, Numpy roots, Numpy poly, Numpy polyval
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-16/
comments: true
---

## 다항식 생성(Poly1d) ##
----------

{% highlight Python %}

import numpy as np

equation1 = np.poly1d([1, 2], True, variable = 'a')
equation2 = np.poly1d([1, 2], True)
equation3 = np.poly1d([1, 2], False)

print(equation1)
print(equation2)
print(equation3)

{% endhighlight %}

**결과**
:    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
1 a - 3 a + 2<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
1 x - 3 x + 2<br>
<br>
1 x + 2<br>
<br>

`다항식 생성 함수(np.poly1d)`는 **입력 배열을 항**으로 간주하여 다항식을 생성합니다.

`np.poly1d(입력 배열, 근 설정, 변수명)`으로 다항식을 생성합니다.

`입력 배열`을 다항식의 **계수**나 **근**으로 사용합니다.

`근 설정`이 `True`라면 **근**으로 사용하며, `False`라면 **계수**로 사용합니다.

`근 설정` 값을 입력하지 않는다면 `False`로 간주합니다.

`변수명`은 다항식의 변수의 이름을 설정합니다.

`변수명` 값을 입력하지 않는다면 `x`로 간주합니다.

<br>

`equation1`은 `입력 배열`을 `근`으로 사용하며, `변수명`을 `a`로 사용합니다.

그러므로, $$equation1 = (a - 1)(a - 2) = a^2 - 3a + 2 $$가 됩니다.

<br>

`equation2`은 `입력 배열`을 `근`으로 사용합니다.

그러므로, $$equation2 = (x - 1)(x - 2) = x^2 - 3x + 2 $$가 됩니다.

<br>

`equation3`은 `입력 배열`을 `계수`으로 사용합니다.

그러므로, $$equation3 = x + 2 $$가 됩니다.

<br>

`입력 배열`의 값의 순서가 방정식의 **계수**나 **근**이 됩니다.

만약, $$ 2x^3 - 1$$ 수식을 입력한다면, `np.poly1d([2, 0, 0, -1])`으로 사용합니다.

<br>
<br>

## 다항식 속성(Poly1d Attributes) ##
----------

{% highlight Python %}

import numpy as np

equation = np.poly1d([1, 2], True)
print(equation.c, equation.coef, equation.coefficients)
print(equation.o, equation.order)
print(equation.r, equation.roots)
print(equation.variable)

{% endhighlight %}

**결과**
:    
[ 1. -3.  2.] [ 1. -3.  2.] [ 1. -3.  2.]<br>
2 2<br>
[2. 1.] [2. 1.]<br>
x<br>
<br>

다항식은 크게 **네 가지 종류**의 속성을 갖고 있습니다.

<br>

`c`, `coef`, `coefficients`는 **다항식의 계수**를 반환합니다.

즉, `[1, -3, 2]`은 $$ x^2 - 3x + 2 $$이 되며, 앞선 예시의 `equation2`의 반환값과 같은 것을 확인할 수 있습니다.

<br>

`o`, `order`는 **다항식의 최고 차수**를 반환합니다.

그러므로, 다항식의 최대 차수인 `2`를 반환합니다.

<br>

`r`, `roots`는 **다항식의 근**을 반환합니다.

다항식에서 `0`이 되는 값인 `[2, 1]`을 반환합니다.

<br>

`variable`은 **다항식의 변수명**을 반환합니다.

변수명을 별도로 입력하지 않았으므로, 기본값인 `x`를 반환합니다.

<br>
<br>

## 다항식 근 계산(Roots) ##
----------

{% highlight Python %}

import numpy as np

p = np.poly1d([1, 10])
equation1 = np.roots(p)
equation2 = np.roots([1, 10])

print(equation1)
print(equation2)

{% endhighlight %}

**결과**
:    
[-10.]<br>
[-10.]<br>
<br>

`다항식 근 계산 함수(np.roots)`는 **입력 배열을 다항식**으로 간주하여 `근`을 계산합니다.

`np.roots(입력 배열)`로 다항식의 근을 계산합니다.

`입력 배열`은 **다항식** 값을 입력하거나 **계수**를 직접 입력할 수 있습니다.

<br>

`입력 배열`은 $$ x + 10 $$이므로, 근은 `-10`이 됩니다.

<br>
<br>

## 다항식 계수 계산(Poly) ##
----------

{% highlight Python %}

import numpy as np

p = np.poly1d([1, -6, 8])
equation1 = np.poly(p)
equation2 = np.poly([1, -6, 8])

print(equation1)
print(equation2)

{% endhighlight %}

**결과**
:    
[  1.  -3. -46.  48.]<br>
[  1.  -3. -46.  48.]<br>
<br>

`다항식 계수 계산 함수(np.poly)`는 **입력 배열을 다항식**의 근의 순서로 `계수`를 계산합니다.

`np.poly(입력 배열)`로 다항식의 계수를 계산합니다.

`입력 배열`은 **다항식** 값을 입력하거나 **계수**를 직접 입력할 수 있습니다.

<br>

`입력 배열`은 $$ (x - 1)(x + 6)(x - 8) $$이므로, $$ x^3 - 3x^2 - 46x +  48 $$이 됩니다.

그러므로, 반환값은 `[1, -3, -46, 48]`이 됩니다.

<br>
<br>

## 다항식 계산(Polyval) ##
----------

{% highlight Python %}

import numpy as np

p = np.poly1d([1, -6, 8])
x = 5
equation1 = np.polyval(p, x)
equation2 = np.polyval([1, -6, 8], np.poly1d([x, x]))

print(equation1)
print(equation2)

{% endhighlight %}

**결과**
:    
3<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
25 x + 20 x + 3<br>
<br>

`다항식 계산 함수(np.polyval)`는 **입력 배열을 다항식**으로 간주하고, **입력값**을 넣어 계산된 값을 반환합니다.

`np.polyval(입력 배열, 입력값)`로 다항식의 계산 결과를 반환합니다.

`입력 배열`은 **다항식** 값을 입력하거나 **계수**를 직접 입력할 수 있습니다.

`입력값`은 **숫자**나 **poly1d**값을 입력합니다.

<br>

**equation1**의 `입력 배열`은 $$ x^2 - 6x + 8 $$이며, $$ x $$에 $$ 5 $$를 입력하므로, 반환값은 $$ 3 $$이 됩니다.

<br>

**equation2**의 `입력 배열`은 $$ x^2 - 6x + 8 $$이며, $$ x $$에 $$ 5x + 5 $$를 입력하므로, 반환값은 $$ 25x^2 + 20x + 3 $$이 됩니다.

<br>
<br>