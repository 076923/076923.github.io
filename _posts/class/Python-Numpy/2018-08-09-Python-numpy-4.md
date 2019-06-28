---
layout: post
title: "Python numpy 강좌 : 제 4강 - 등간격"
tagline: "Python Numpy equal interval"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy array, equal interval
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-4/
comments: true
---

## 등간격 ##
----------

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

`numpy.arange(start, end, step=간격, dtype=자료형)`을 사용하여 `start` ~ `end-1` 사이의 값을 `간격`만큼 띄워 배열로 반환합니다.

`start`값은 항상 포함되며, `end`값은 포함되지 않을 수도 있습니다.

`간격`이 `end`값을 초과할 경우, `start`값만 포함합니다.

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
(array([ 0. ,  2.5,  5. ,  7.5, 10. ]), 2.5)<br>
[ 1.    3.25  5.5   7.75 10.  ]<br>
[0. 2. 4. 6. 8.]<br>
<br>

`numpy.linspace(start, end, num=개수, endpoint=True, retstep=False, dtype=자료형)`을 사용하여 `start` ~ `end` 사이의 값을 `개수`만큼 생성하여 배열로 반환합니다.

`endpoint`가 `True`일 경우 `end`의 값이 **마지막 값**이 되며, `False`일 경우 `end`의 값을 **마지막 값으로 사용하지 않습니다.**

`retstep`이 `True`일 경우 값들의 `간격`을 배열에 포함합니다. `numpy.arange()`의 `step`과 동일한 의미를 지닙니다.

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
[1.00000000e+00 3.16227766e+02 1.00000000e+05 3.16227766e+07 1.00000000e+10]<br>
[5.00000000e+00 1.86918598e+02 6.98771243e+03 2.61226682e+05 9.76562500e+06]<br>
[1. 1. 1. 1. 1.]<br>
<br>

`numpy.logspace(start, end, num=개수, endpoint=True, base=10.0, dtype=자료형)`을 사용하여 `start` ~ `end` 사이의 로그 배율을 사용하여 값을 `개수`만큼 생성하여 배열로 반환합니다.

`endpoint`가 `True`일 경우 `end`의 값이 **마지막 값**이 되며, `False`일 경우 `end`의 값을 **마지막 값으로 사용하지 않습니다.**

`base`는 로그 값의 `간격`을 의미합니다.

