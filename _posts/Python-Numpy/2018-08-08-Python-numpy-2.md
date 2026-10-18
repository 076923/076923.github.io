---
layout: post
title: "Python Numpy 강좌 : 제 2강 - 배열 생성 (2)"
tagline: "Python Numpy Array (2)"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy array, dtype, ndim, shape, size, astype
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-2/
comments: true
toc: true
---

## 1차원 배열

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3], dtype=int)
b = np.array([1.1, 2.2, 3.3], dtype=float)
c = np.array([1, 1, 0], dtype=bool)

print(a)
print(b)
print(c)
print(a.dtype)

{% endhighlight %}

**결과**
:    
[1 2 3]<br>
[1.1 2.2 3.3]<br>
[ True  True False]<br>
int64<br>
<br>

`numpy.array(배열, dtype=자료형)`을 사용하여 **배열 생성과 자료형을 설정할 수 있습니다.**

`dtype`이 생략될 경우, 데이터의 자료형을 유추하여 자동적으로 `dtype`을 할당합니다.

`bool`형식의 특정한 배열을 생성시 `dtype`을 사용하여 `명시적`으로 표기합니다. `0`은 `False`로, `0`이 아닌 값은 `True`로 변환됩니다.

- Tip : `dtype`을 생략한 정수 배열의 기본 자료형은 `int64`입니다. NumPy 1.x 버전의 Windows 환경에서는 `int32`가 기본값이었으나, **NumPy 2.0부터는 Windows에서도 `int64`**가 기본값입니다.

- Tip : `int`, `float`, `bool`처럼 파이썬 기본 자료형 대신 `np.int32`, `np.float32`, `np.uint8`처럼 **비트 수를 지정한 자료형**을 사용할 수도 있습니다. 단, `np.int`, `np.float` 같은 별칭은 NumPy 1.24에서 제거되었으므로 사용하지 않습니다.

<br>
<br>

## 다차원 배열

{% highlight Python %}

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

print(a)
print(b)
print(a[0][1])
print(b[0][1][1])

{% endhighlight %}

**결과**
:    
[[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>
[[[ 1  2]<br>
  [ 3  4]]<br>
<br>
 [[ 5  6]<br>
  [ 7  8]]<br>
<br>
 [[ 9 10]<br>
  [11 12]]]<br>
2<br>
4<br>
<br>

`numpy.array(다차원 배열, dtype=자료형)`을 사용하여 **다차원 배열 생성과 자료형을 설정할 수 있습니다.**

`dtype`이 생략될 경우, 데이터의 자료형을 유추하여 자동적으로 `dtype`을 할당합니다.

다차원 배열의 값을 불러올 때, `배열[페이지][행][열]...` 형태로 배열의 값을 불러올 수 있습니다.

- Tip : `numpy` 배열은 `배열[0, 1]`이나 `배열[0, 1, 1]`처럼 **하나의 대괄호 안에 쉼표로 구분하여** 인덱스를 지정할 수도 있습니다. `배열[0][1]`은 `배열[0]`로 중간 배열을 만든 뒤 다시 `[1]`을 적용하지만, `배열[0, 1]`은 한 번에 원소에 접근하므로 더 효율적이며 `numpy`에서 권장되는 방식입니다.

<br>
<br>

## 배열 속성 반환

{% highlight Python %}

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

print(a.ndim)
print(a.shape)
print(a.dtype)

print(np.ndim(b))
print(np.shape(b))

{% endhighlight %}

**결과**
:    
2<br>
(3, 3)<br>
int64<br>
3<br>
(3, 2, 2)<br>
<br>

`배열.ndim` 또는 `np.ndim(배열)`을 사용하여 **배열의 차원을 반환합니다.**

`배열.shape` 또는 `np.shape(배열)`을 사용하여 **배열의 형태를 반환합니다.**

`배열.dtype`을 사용하여 **배열의 자료형을 반환합니다.**

<br>
<br>

## 배열 크기 반환

{% highlight Python %}

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.size)
print(len(a))
print(a.itemsize)
print(a.nbytes)

{% endhighlight %}

**결과**
:    
6<br>
2<br>
8<br>
48<br>
<br>

`배열.size`를 사용하여 **배열의 전체 원소 개수를 반환합니다.** `shape`의 모든 값을 곱한 값과 같습니다.

`len(배열)`은 전체 원소 개수가 아닌, **첫 번째 축(행)의 길이**를 반환합니다.

`배열.itemsize`는 **원소 하나가 차지하는 바이트 수**를, `배열.nbytes`는 **배열 전체가 차지하는 바이트 수**를 반환합니다.

`int64`는 원소 하나당 8바이트를 사용하므로, 6개의 원소로 이루어진 배열은 48바이트를 차지합니다.

<br>
<br>

## 자료형 변환

{% highlight Python %}

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.astype(np.float32)
c = np.array([1.7, -1.7, 2.5]).astype(int)

print(b)
print(b.dtype)
print(c)

{% endhighlight %}

**결과**
:    
[[1. 2. 3.]<br>
 [4. 5. 6.]]<br>
float32<br>
[ 1 -1  2]<br>
<br>

`배열.astype(자료형)`을 사용하여 **자료형을 변환한 새로운 배열을 반환합니다.** 원본 배열의 자료형은 변경되지 않습니다.

실수를 정수로 변환할 경우, 반올림이 아닌 **소수점 이하를 버리는 방식**으로 변환됩니다. 음수 역시 0에 가까운 방향으로 버려지므로 `-1.7`은 `-1`이 됩니다.

- Tip : 반올림한 정수가 필요하다면 `np.round(배열).astype(int)`처럼 먼저 반올림한 뒤 변환합니다. 단, `np.round()`는 `.5`를 **가장 가까운 짝수**로 반올림하므로 `0.5`는 `0`, `2.5`는 `2`가 됩니다.
