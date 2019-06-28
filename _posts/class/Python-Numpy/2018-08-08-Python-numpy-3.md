---
layout: post
title: "Python Numpy 강좌 : 제 1강 - 배열 생성 (3)"
tagline: "Python Numpy Array (3)"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy array
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-3/
comments: true
---

## 모든 값이 1인 배열 ##
----------

{% highlight Python %}

import numpy as np

a = np.ones((2,2), dtype=int)
b = [1, 2, 3, 4, 5]
c = np.ones_like(b, dtype=int)

print(a)
print(b)
print(c)
{% endhighlight %}

**결과**
:    
[[1 1]<br>
 [1 1]]<br>
[1, 2, 3, 4, 5]<br>
[1 1 1 1 1]<br>
<br>

`numpy.ones(배열, dtype=자료형)`을 사용하여 **모든 원소의 값이 1인 배열을 생성할 수 있습니다.**

`numpy.ones_like(배열, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 모든 원소의 값이 1인 배열을 생성할 수 있습니다.**

<br>
<br>

## 모든 값이 0인 배열 ##
----------

{% highlight Python %}

import numpy as np

a = np.zeros((2,2), dtype=int)
b = [1, 2, 3, 4, 5]
c = np.zeros_like(b, dtype=int)

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[[0 0]<br>
 [0 0]]<br>
[1, 2, 3, 4, 5]<br>
[0 0 0 0 0]<br>
<br>

`numpy.zeros(배열, dtype=자료형)`을 사용하여 **모든 원소의 값이 0인 배열을 생성할 수 있습니다.**

`numpy.zeros_like(배열, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 모든 원소의 값이 0인 배열을 생성할 수 있습니다.**

<br>
<br>

## 모든 값이 비어있는 배열 ##
----------

{% highlight Python %}

import numpy as np

a = np.empty((2,2), dtype=int)
b = [1, 2, 3, 4, 5]
c = np.empty_like(b, dtype=int)

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[[     257 83886080]<br>
 [      18        0]]<br>
[1, 2, 3, 4, 5]<br>
[0 0 0 0 0]<br>
<br>

`numpy.empty(배열, dtype=자료형)`을 사용하여 **특정한 값으로 초기화하지 않는 배열을 생성할 수 있습니다.**

`numpy.empty_like(배열, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 특정한 값으로 초기화하지 않는 배열을 생성할 수 있습니다.**

**난수와 다른 임의의 값이 들어가며 값은 메모리에 저장된 내용에 따라 달라집니다.**

* Tip : `empty`를 이용하여 배열을 생성할 경우, **조금 더 빠른 속도로 생성이 가능합니다.**

<br>
<br>

## 대각의 값이 1인 배열(단위 행렬) ##
----------

{% highlight Python %}

import numpy as np

import numpy as np

a = np.identity(4, dtype=int)
b = np.eye(4, 4, k=1, dtype=int)

print(a)
print(b)
{% endhighlight %}

**결과**
:    
[[1 0 0 0]<br>
 [0 1 0 0]<br>
 [0 0 1 0]<br>
 [0 0 0 1]]<br>
[[0 1 0 0]<br>
 [0 0 1 0]<br>
 [0 0 0 1]<br>
 [0 0 0 0]]<br>
<br>

`numpy.identity(N, dtype=자료형)`을 사용하여 **NxN 크기의 단위 행렬을 반환합니다.**

`numpy.eye(N, M, k=K, dtype=자료형)`을 사용하여 **NxM 크기의 K값 만큼 이격된 단위 행렬을 반환합니다.**

K의 값이 `양수`일 경우, `우상` 방향으로 이동하며, K의 값이 `음수`일 경우, `좌하` 방향으로 이동합니다.