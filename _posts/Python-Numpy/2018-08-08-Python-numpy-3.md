---
layout: post
title: "Python Numpy 강좌 : 제 3강 - 배열 생성 (3)"
tagline: "Python Numpy Array (3)"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy array, ones, zeros, empty, identity, eye, full
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-3/
comments: true
toc: true
---

## 모든 값이 1인 배열

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

`numpy.ones(형태, dtype=자료형)`을 사용하여 **모든 원소의 값이 1인 배열을 생성할 수 있습니다.**

`형태`는 `(2, 2)`처럼 각 차원의 크기를 담은 `tuple`로 지정하며, 1차원 배열이라면 `numpy.ones(5)`처럼 정수 하나로 지정할 수도 있습니다.

`numpy.ones_like(배열, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 모든 원소의 값이 1인 배열을 생성할 수 있습니다.**

- Tip : `dtype`을 생략할 경우, 기본 자료형은 정수가 아닌 `float64`입니다. `numpy.ones(3)`은 `[1. 1. 1.]`을 반환합니다. 반면 `ones_like`는 `dtype`을 생략하면 **입력 배열의 자료형**을 따릅니다.

<br>
<br>

## 모든 값이 0인 배열

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

`numpy.zeros(형태, dtype=자료형)`을 사용하여 **모든 원소의 값이 0인 배열을 생성할 수 있습니다.**

`numpy.zeros_like(배열, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 모든 원소의 값이 0인 배열을 생성할 수 있습니다.**

<br>
<br>

## 모든 값이 비어있는 배열

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

`numpy.empty(형태, dtype=자료형)`을 사용하여 **특정한 값으로 초기화하지 않는 배열을 생성할 수 있습니다.**

`numpy.empty_like(배열, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 특정한 값으로 초기화하지 않는 배열을 생성할 수 있습니다.**

**난수와 다른 임의의 값이 들어가며 값은 메모리에 저장된 내용에 따라 달라집니다.** 위의 결과는 하나의 예시이며, 실행할 때마다 다른 값이 출력될 수 있습니다. `empty_like`의 결과가 `0`으로 출력되었더라도, 0으로 초기화된다는 보장은 없습니다.

- Tip : `empty`를 이용하여 배열을 생성할 경우, 값을 채우는 과정을 생략하므로 **조금 더 빠른 속도로 생성이 가능합니다.** 단, 생성 직후 **모든 원소에 값을 직접 채울 때만** 사용해야 합니다.

<br>
<br>

## 대각의 값이 1인 배열(단위 행렬)

{% highlight Python %}

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

`numpy.eye(N, M, k=K, dtype=자료형)`을 사용하여 **NxM 크기이며, 주대각선에서 K만큼 떨어진 대각선의 값이 1인 행렬을 반환합니다.**

`M`을 생략하면 `N`과 같은 값이 되며, `K`를 생략하면 `0`이 되어 `numpy.identity(N)`과 같은 단위 행렬이 됩니다.

K의 값이 `양수`일 경우, `우상` 방향으로 이동하며, K의 값이 `음수`일 경우, `좌하` 방향으로 이동합니다.

- Tip : `numpy.identity()`와 `numpy.eye()` 모두 `dtype`을 생략할 경우 `float64` 자료형의 행렬을 반환합니다.

<br>
<br>

## 특정한 값으로 채워진 배열

{% highlight Python %}

import numpy as np

a = np.full((2, 3), 7)
b = [1, 2, 3]
c = np.full_like(b, 0.5)
d = np.full_like(b, 0.5, dtype=float)

print(a)
print(c)
print(d)

{% endhighlight %}

**결과**
:    
[[7 7 7]<br>
 [7 7 7]]<br>
[0 0 0]<br>
[0.5 0.5 0.5]<br>
<br>

`numpy.full(형태, 값, dtype=자료형)`을 사용하여 **모든 원소가 지정한 값으로 채워진 배열을 생성할 수 있습니다.**

`numpy.full_like(배열, 값, dtype=자료형)`을 사용하여 **배열의 크기와 동일하며 모든 원소가 지정한 값으로 채워진 배열을 생성할 수 있습니다.**

`full_like`는 `dtype`을 생략하면 **입력 배열의 자료형**을 따릅니다. `b`는 정수 배열이므로 `0.5`가 정수로 변환되어 `0`으로 채워집니다. 실수 값을 채우려면 `dtype=float`을 명시해야 합니다.

- Tip : `ones`, `zeros`, `empty`는 `dtype`을 생략하면 `float64`가 되지만, `full`은 **채우는 값의 자료형**을 따릅니다. `numpy.full((2, 3), 7)`은 정수 배열을, `numpy.full((2, 3), 7.0)`은 실수 배열을 반환합니다.
