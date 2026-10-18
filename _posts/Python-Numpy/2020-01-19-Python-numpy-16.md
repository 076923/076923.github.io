---
layout: post
title: "Python Numpy 강좌 : 제 16강 - 다항식 계산 (1)"
tagline: "Python Numpy Poly & Roots (1)"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy poly1d, Numpy roots, Numpy poly, Numpy polyval, Complex Roots
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-16/
comments: true
toc: true
---

## 다항식 생성(Poly1d)

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

`다항식 생성 함수(np.poly1d)`는 **입력 배열을 계수 또는 근**으로 간주하여 다항식을 생성합니다.

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

계수로 사용하는 경우, **최고차항의 계수부터 내림차순**으로 입력합니다.

만약, $$ 2x^3 - 1$$ 수식을 입력한다면, 빠진 차수의 계수인 `0`도 함께 입력해 `np.poly1d([2, 0, 0, -1])`으로 사용합니다.

- Tip : `np.poly1d`를 비롯한 `np.poly*` 함수들은 **레거시(Legacy) API**입니다. 공식 문서에서는 새로운 코드에 `numpy.polynomial` 패키지의 `Polynomial` 클래스를 권장하며, 이 클래스는 계수를 **최저차항부터 오름차순**으로 입력합니다. 자세한 내용은 [제 17강][17강]에서 다룹니다.

<br>
<br>

## 다항식 속성(Poly1d Attributes)

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

## 다항식 근 계산(Roots)

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

`입력 배열`은 **다항식(poly1d)** 값을 입력하거나 **계수**를 내림차순으로 직접 입력할 수 있습니다.

`입력 배열`은 $$ x + 10 $$이므로, 근은 `-10`이 됩니다.

`np.roots`는 다항식의 **동반 행렬(Companion Matrix)의 고윳값**을 계산하는 방식으로 근을 구하므로, 결과는 항상 `float` 또는 `complex` 자료형의 배열이 됩니다.

<br>
<br>

## 다항식 계수 계산(Poly)

{% highlight Python %}

import numpy as np

p = np.poly1d([1, -6, 8])
equation1 = np.poly(p.r)
equation2 = np.poly([1, -6, 8])

print(equation1)
print(equation2)

{% endhighlight %}

**결과**
:    
[ 1. -6.  8.]<br>
[  1.  -3. -46.  48.]<br>
<br>

`다항식 계수 계산 함수(np.poly)`는 **입력 배열을 다항식의 근**으로 간주하여 `계수`를 계산합니다. `np.roots`의 역연산입니다.

`np.poly(입력 배열)`로 다항식의 계수를 계산합니다.

`입력 배열`에는 **근의 목록**을 입력합니다. 반환값은 `poly1d`가 아닌 **계수 배열(ndarray)**입니다.

<br>

**equation1**은 $$ x^2 - 6x + 8 $$의 근인 `p.r`(`[4, 2]`)을 입력했으므로, 원래의 계수인 `[1, -6, 8]`이 반환됩니다.

**equation2**의 `입력 배열`은 근 `1`, `-6`, `8`이므로 $$ (x - 1)(x + 6)(x - 8) $$이 되고, 이를 전개하면 $$ x^3 - 3x^2 - 46x +  48 $$이 됩니다.

그러므로, 반환값은 `[1, -3, -46, 48]`이 됩니다.

- Tip : `np.poly`에 `poly1d` 객체를 그대로 입력하면 그 **계수 배열이 근으로 해석**됩니다. `np.poly(p)`는 `np.poly([1, -6, 8])`과 같은 결과가 되므로 주의해야 합니다.

<br>
<br>

## 다항식 계산(Polyval)

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

`입력 배열`은 **다항식(poly1d)** 값을 입력하거나 **계수**를 내림차순으로 직접 입력할 수 있습니다.

`입력값`은 **숫자**, **배열**, **poly1d**값을 입력합니다. 배열을 입력하면 각 요소에 대한 계산값을 배열로 반환합니다.

**equation1**의 `입력 배열`은 $$ x^2 - 6x + 8 $$이며, $$ x $$에 $$ 5 $$를 입력하므로, 반환값은 $$ 3 $$이 됩니다.

**equation2**의 `입력 배열`은 $$ x^2 - 6x + 8 $$이며, $$ x $$에 $$ 5x + 5 $$를 입력하므로, 반환값은 $$ 25x^2 + 20x + 3 $$이 됩니다.

<br>
<br>

## 다항식 객체 호출과 연산

{% highlight Python %}

import numpy as np

p = np.poly1d([1, -6, 8])

print(p(5))
print(p([0, 1, 2]))
print(p * p)
print(p ** 2 == p * p)

{% endhighlight %}

**결과**
:    
3<br>
[8 3 0]<br>
&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2<br>
1 x - 12 x + 52 x - 96 x + 64<br>
True<br>
<br>

`poly1d` 객체는 **함수처럼 호출**할 수 있습니다. `p(5)`는 `np.polyval(p, 5)`와 같은 결과를 반환합니다.

배열을 입력하면 각 요소에 대한 계산값을 배열로 반환합니다.

또한, `poly1d` 객체끼리는 `+`, `-`, `*`, `**` 등의 **산술 연산자**를 사용해 다항식 연산을 수행할 수 있습니다.

$$ (x^2 - 6x + 8)^2 = x^4 - 12x^3 + 52x^2 - 96x + 64 $$이므로, `p * p`와 `p ** 2`는 같은 다항식이 됩니다.

<br>
<br>

## 복소수 근과 중근

{% highlight Python %}

import numpy as np

print(np.roots([1, 0, 1]))
print(np.roots([1, -2, 1]))

{% endhighlight %}

**결과**
:    
[-0.+1.j  0.-1.j]<br>
[1. 1.]<br>
<br>

$$ x^2 + 1 $$은 실수 범위에서 근이 없으므로, `np.roots`는 **복소수 근**인 $$ \pm i $$를 `complex` 자료형으로 반환합니다. Python에서는 허수 단위를 `j`로 표기합니다.

$$ x^2 - 2x + 1 = (x - 1)^2 $$은 `1`을 **중근**으로 가지므로, 같은 값이 차수만큼 반복되어 반환됩니다.

- Tip : `np.roots`는 수치 계산으로 근을 구하므로, 중근이나 계수가 큰 다항식에서는 `0.9999999` 처럼 **미세한 오차**가 발생할 수 있습니다. 근을 비교할 때는 `==` 대신 `np.isclose`를 사용합니다.

[17강]: https://076923.github.io/posts/Python-numpy-17/
