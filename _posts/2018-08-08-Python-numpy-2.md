---
bg: "numpy.png"
layout: post
comments: true
title: "Python numpy 강좌 : 제 2강 - 배열 생성 (2)"
crawlertitle: "Python numpy 강좌 : 제 2강 - 배열 생성 (2)"
summary: "Python numpy array (2)"
date: 2018-08-08
categories: posts
tags: ['Python-numpy']
author: 윤대희
star: true
---

### 1차원 배열 ###
----------

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
int32<br>
<br>

`numpy.array(배열, dtype=자료형)`을 사용하여 **배열 생성과 자료형을 설정할 수 있습니다.**

`dtype`이 생략될 경우, 데이터의 자료형을 유추하여 자동적으로 `dtype`을 할당합니다.

`bool`형식의 특정한 배열을 생성시 `dtype`을 사용하여 `명시적`으로 표기합니다.

<br>

### 다차원 배열 ###
----------

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

<br>

### 배열 속성 반환 ###
----------

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
int32<br>
3<br>
(3, 2, 2)<br>
<br>

`배열.ndim` 또는 `np.ndim(배열)`을 사용하여 **배열의 차원을 반환합니다.**

`배열.shape` 또는 `np.shape(배열)`을 사용하여 **배열의 형태를 반환합니다.**

`배열.dtype`을 사용하여 **배열의 자료형을 반환합니다.**





