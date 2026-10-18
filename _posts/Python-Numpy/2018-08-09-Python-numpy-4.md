---
layout: post
title: "Python Numpy 강좌 : 제 4강 - 등간격"
tagline: "Python Numpy equal interval"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy array, equal interval, arange, linspace, logspace, geomspace
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-4/
comments: true
toc: true
---

## 등간격

{% highlight Python %}

import numpy as np

a = np.arange(0, 10, step=5)
b = np.arange(1, 10, step=5)
c = np.arange(0, 10, step=1)

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[0 5]<br>
[1 6]<br>
[0 1 2 3 4 5 6 7 8 9]<br>
<br>

`numpy.arange(start, end, step=간격, dtype=자료형)`을 사용하여 `start` 이상 `end` 미만의 범위에서 `start`부터 `간격`만큼 띄운 값을 배열로 반환합니다.

`start`값은 항상 포함되며, `end`값은 **포함되지 않습니다.** 파이썬의 `range()`와 같은 규칙이지만, `numpy.arange()`는 `0.1`과 같은 **실수 간격**도 사용할 수 있습니다.

`간격`이 `end - start` 이상일 경우, `start`값만 포함합니다.

`start`와 `step`은 생략할 수 있으며, `numpy.arange(5)`처럼 값을 하나만 입력하면 `start=0`, `step=1`로 간주하여 `[0 1 2 3 4]`를 반환합니다. `step`에 음수를 입력하면 `numpy.arange(10, 0, -3)`처럼 감소하는 배열을 생성할 수 있습니다.

- Tip : 실수 간격을 사용할 경우, 부동 소수점 오차로 인해 **`end`값이 포함되는 경우**가 있습니다. 예를 들어 `numpy.arange(1, 1.3, 0.1)`은 `[1.  1.1 1.2 1.3]`을 반환합니다. 실수 구간을 나눌 때는 개수를 정확히 지정할 수 있는 `numpy.linspace()`를 사용하는 것이 안전합니다.

<br>

{% highlight Python %}

import numpy as np

a = np.linspace(0, 10, num=5, endpoint=True, retstep=True)
b = np.linspace(1, 10, num=5, endpoint=True, retstep=False)
c = np.linspace(0, 10, num=5, endpoint=False, retstep=False)

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
(array([ 0. ,  2.5,  5. ,  7.5, 10. ]), np.float64(2.5))<br>
[ 1.    3.25  5.5   7.75 10.  ]<br>
[0. 2. 4. 6. 8.]<br>
<br>

`numpy.linspace(start, end, num=개수, endpoint=True, retstep=False, dtype=자료형)`을 사용하여 `start` ~ `end` 사이를 같은 간격으로 나눈 값을 `개수`만큼 생성하여 배열로 반환합니다.

`개수`를 생략할 경우, 기본값인 `50`개의 값을 생성합니다.

`endpoint`가 `True`일 경우 `end`의 값이 **마지막 값**이 되며, `False`일 경우 `end`의 값을 **마지막 값으로 사용하지 않습니다.**

`retstep`이 `True`일 경우 배열과 값들의 `간격`을 묶은 `(배열, 간격)` 형태의 **tuple을 반환합니다.** 간격은 `numpy.arange()`의 `step`과 동일한 의미를 지닙니다.

- Tip : NumPy 2.0부터 `numpy` 스칼라 값은 `np.float64(2.5)`처럼 자료형이 함께 표시됩니다. NumPy 1.x 버전에서는 `2.5`로 표시되며, 값 자체는 같습니다.

<br>

{% highlight Python %}

import numpy as np

a = np.logspace(0, 10, num=5, endpoint=True, base=10.0)
b = np.logspace(1, 10, num=5, endpoint=True, base=5.0)
c = np.logspace(0, 10, num=5, endpoint=False, base=1.0)

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[1.00000000e+00 3.16227766e+02 1.00000000e+05 3.16227766e+07<br>
 1.00000000e+10]<br>
[5.00000000e+00 1.86918598e+02 6.98771243e+03 2.61226682e+05<br>
 9.76562500e+06]<br>
[1. 1. 1. 1. 1.]<br>
<br>

`numpy.logspace(start, end, num=개수, endpoint=True, base=10.0, dtype=자료형)`을 사용하여 `base ** start` ~ `base ** end` 사이를 로그 배율로 같은 간격이 되도록 값을 `개수`만큼 생성하여 배열로 반환합니다.

`start`와 `end`는 값 자체가 아니라 **지수**입니다. 내부적으로 `numpy.linspace(start, end, num)`으로 지수를 만든 뒤, `base`를 그 지수만큼 거듭제곱합니다. 따라서 `a`는 `10⁰` ~ `10¹⁰` 사이의 값을 생성하며, 이웃한 값의 **비율**이 일정합니다.

`endpoint`가 `True`일 경우 `base ** end`의 값이 **마지막 값**이 되며, `False`일 경우 **마지막 값으로 사용하지 않습니다.**

`base`는 로그의 `밑`을 의미합니다. `c`처럼 `base`가 `1.0`이라면 `1`을 몇 번 거듭제곱해도 `1`이므로 모든 값이 `1`이 됩니다.

<br>

{% highlight Python %}

import numpy as np

a = np.geomspace(1, 1000, num=4)
b = np.logspace(0, 3, num=4)

print(a)
print(b)

{% endhighlight %}

**결과**
:    
[   1.   10.  100. 1000.]<br>
[   1.   10.  100. 1000.]<br>
<br>

`numpy.geomspace(start, end, num=개수, endpoint=True, dtype=자료형)`을 사용하여 `start` ~ `end` 사이에서 이웃한 값의 비율이 일정한 `등비수열`을 `개수`만큼 생성하여 배열로 반환합니다.

`numpy.logspace()`와 달리 지수가 아닌 **시작값과 끝값을 직접 지정**하므로, 원하는 범위의 등비수열을 더 직관적으로 생성할 수 있습니다.
