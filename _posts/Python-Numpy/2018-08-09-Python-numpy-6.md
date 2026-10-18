---
layout: post
title: "Python Numpy 강좌 : 제 6강 - 리쉐이프"
tagline: "Python Numpy reshape"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy reshape, Numpy ravel, Numpy flatten
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-6/
comments: true
toc: true
---

## 리쉐이프

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]])

print(a)
print(np.shape(a))
a.shape = (6,2)
print(a)

{% endhighlight %}

**결과**
:    
[[ 1  2  3  4]<br>
 [ 5  6  7  8]<br>
 [ 9 10 11 12]]<br>
(3, 4)<br>
[[ 1  2]<br>
 [ 3  4]<br>
 [ 5  6]<br>
 [ 7  8]<br>
 [ 9 10]<br>
 [11 12]]<br>
<br>

`배열.shape = (행, 열)`을 통하여 배열의 형태를 변환할 수 있습니다.

변환할 형태의 `행`과 `열`의 곱은 원본 배열의 **원소 개수**와 같아야합니다.

원본 배열의 크기가 `3x4=12`일 경우, 리쉐이프 할때 `6x2=12`로 **총 개수와 동일해야합니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]],
              [[13, 14, 15], [16, 17, 18]]])

print(a)
print(np.shape(a))
a.shape = (2, 3, 3)
print(a)

{% endhighlight %}

**결과**
:    
[[[ 1  2  3]<br>
  [ 4  5  6]]<br>
<br>
 [[ 7  8  9]<br>
  [10 11 12]]<br>
<br>
 [[13 14 15]<br>
  [16 17 18]]]<br>
(3, 2, 3)<br>
[[[ 1  2  3]<br>
  [ 4  5  6]<br>
  [ 7  8  9]]<br>
<br> 
 [[10 11 12]<br>
  [13 14 15]<br>
  [16 17 18]]]<br>
<br>

**3차원 이상의 배열** 또한 형태를 변환할 수 있습니다.

`배열.shape= (페이지, 행, 열)`의 순서이며, 역시 **총 길이의 개수와 동일해야합니다.**

- Tip : `배열.shape`에 값을 대입하는 방식은 **원본 배열 자체의 형태를 변경**합니다. 이 방식은 `NumPy 2.5`부터 사용이 권장되지 않으며(DeprecationWarning), 아래의 `reshape`를 사용하는 것이 좋습니다.

<br>
<br>

## reshape 함수

{% highlight Python %}

import numpy as np

a = np.arange(12)

b = a.reshape(3, 4)
c = np.reshape(a, (2, -1))
d = a.reshape(-1, 6)

print(b)
print(c)
print(d.shape)

{% endhighlight %}

**결과**
:    
[[ 0  1  2  3]<br>
&nbsp;[ 4  5  6  7]<br>
&nbsp;[ 8  9 10 11]]<br>
[[ 0  1  2  3  4  5]<br>
&nbsp;[ 6  7  8  9 10 11]]<br>
(2, 6)<br>
<br>

`배열.reshape(형태)` 또는 `np.reshape(배열, 형태)`를 사용하여 **형태가 변경된 새로운 배열**을 반환할 수 있습니다. `배열.shape`에 대입하는 방식과 달리 원본 배열의 형태는 변경되지 않습니다.

형태에 `-1`을 입력할 경우, 나머지 차원과 원소 개수를 이용해 **해당 차원의 크기를 자동으로 계산**합니다.

`12`개의 원소를 `(2, -1)`로 변환하면 `-1`은 `12 / 2 = 6`으로 계산되며, `-1`은 **하나의 차원에만** 사용할 수 있습니다.

- Tip : `reshape` 함수는 `order` 매개변수로 원소를 읽고 배치하는 순서를 지정할 수 있습니다. 기본값은 `'C'`(행 우선)이며, 자세한 내용은 [13강 - 메모리 레이아웃][13강]에서 다룹니다.

<br>
<br>

## 리쉐이프와 뷰

{% highlight Python %}

import numpy as np

a = np.arange(6)
b = a.reshape(2, 3)

b[0, 0] = 99

print(a)
print(np.shares_memory(a, b))

{% endhighlight %}

**결과**
:    
[99  1  2  3  4  5]<br>
True<br>
<br>

`reshape`는 가능한 경우 데이터를 복사하지 않고 **원본과 데이터를 공유하는 뷰(View)**를 반환합니다.

그러므로, 리쉐이프된 배열의 값을 변경하면 **원본 배열의 값도 함께 변경됩니다.** 원본과 독립적인 배열이 필요하다면 `배열.reshape(형태).copy()`를 사용합니다.

<br>
<br>

## 1차원 배열로 펼치기

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]])

r = a.ravel()
f = a.flatten()

r[0] = 99
f[1] = -1

print(a)
print(r)
print(f)

{% endhighlight %}

**결과**
:    
[[99  2  3]<br>
&nbsp;[ 4  5  6]]<br>
[99  2  3  4  5  6]<br>
[ 1 -1  3  4  5  6]<br>
<br>

다차원 배열을 **1차원 배열로 펼칠 때**는 `배열.ravel()` 또는 `배열.flatten()`을 사용합니다.

`ravel()`은 가능한 경우 **뷰**를 반환하므로 값을 변경하면 원본도 변경되며, `flatten()`은 항상 **복사본**을 반환하므로 원본에 영향을 주지 않습니다.

- Tip : `배열.reshape(-1)`도 `ravel()`과 동일하게 1차원 배열을 반환합니다.

[13강]: https://076923.github.io/posts/Python-numpy-13/
