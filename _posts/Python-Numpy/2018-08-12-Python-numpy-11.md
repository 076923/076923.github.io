---
layout: post
title: "Python Numpy 강좌 : 제 11강 - 병합 및 분할"
tagline: "Python Numpy stack & split"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy stack, Numpy Split, Numpy hstack, Numpy vstack, Numpy dstack, Numpy tile, Numpy c_, Numpy r_, Numpy hsplit, Numpy vsplit, Numpy dsplit
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-11/
comments: true
toc: true
---

## 병합

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.hstack([a, b])
d = np.vstack([a, b])

print(a)
print(b)
print("--------")
print(c)
print(d)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
\-\-\-\-\-\-\-\-<br>
[[1 2 5 6]<br>
 [3 4 7 8]]<br>
[[1 2]<br>
 [3 4]<br>
 [5 6]<br>
 [7 8]]<br>
<br>

`numpy.hstack([배열1, 배열2])`를 이용하여 `배열1` **우측**에 `배열2`를 **이어 붙일 수 있습니다.**

`numpy.vstack([배열1, 배열2])`를 이용하여 `배열1` **하단**에 `배열2`를 **이어 붙일 수 있습니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.stack([a, b], axis=0)
d = np.stack([a, b], axis=1)

print(a)
print(b)
print("--------")
print(c)
print("--------")
print(d)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
\-\-\-\-\-\-\-\-<br>
[[[1 2]<br>
  [3 4]]<br>
<br>
 [[5 6]<br>
  [7 8]]]<br>
\-\-\-\-\-\-\-\-<br>
[[[1 2]<br>
  [5 6]]<br>
<br>
 [[3 4]<br>
  [7 8]]]<br>
<br>

`numpy.stack([배열1, 배열2, axis=축])`를 이용하여 지정한 `축`으로 `배열1`과 `배열2`를 **이어 붙일 수 있습니다.**

`축`은 **이어 붙일 차원의 범위를 넘어갈 수 없습니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.dstack([a, b])

print(a)
print(b)
print("--------")
print(c)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
\-\-\-\-\-\-\-\-<br>
[[[1 5]<br>
  [2 6]]<br>
<br>
 [[3 7]<br>
  [4 8]]]<br>
<br>

`numpy.dstack([배열1, 배열2])`를 이용하여 `새로운 축`으로 `배열1`과 `배열2`를 **이어 붙일 수 있습니다.**

`numpy.stack([a, b], axis=2)`과 **동일한 결과를 반환합니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.tile(a, 2)

print(a)
print(b)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[1 2 1 2]<br>
 [3 4 3 4]]<br>
<br>

`numpy.tile(배열, 반복 횟수)`를 이용하여 `배열`을 `반복 횟수`만큼 **이어 붙일 수 있습니다.**

<br>
<br>

## 특수 병합 메서드

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.c_[a, b]
d = np.r_[a, b]

print(a)
print(b)
print("--------")
print(c)
print(d)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
\-\-\-\-\-\-\-\-<br>
[[1 2 5 6]<br>
 [3 4 7 8]]<br>
[[1 2]<br>
 [3 4]<br>
 [5 6]<br>
 [7 8]]<br>
<br>

`numpy.c_[배열1, 배열2]`를 이용하여 `배열1` **우측**에 `배열2`를 **이어 붙일 수 있습니다.**

`numpy.r_[배열1, 배열2]`를 이용하여 `배열1` **하단**에 `배열2`를 **이어 붙일 수 있습니다.**

`소괄호 ()`를 사용하지 않고 `대괄호 []`를 사용하여 메서드를 생성합니다.

<br>
<br>

## 분할

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])

a = np.hsplit(array, 2)
b = np.hsplit(array, (1, 3))
c = np.vsplit(array, 2)

print(a)
print("--------")
print(b)
print("--------")
print(c)

{% endhighlight %}

**결과**
:    
[array([[1, 2],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[5, 6]]), array([[3, 4],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[7, 8]])]<br>
\-\-\-\-\-\-\-\-<br>
[array([[1],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[5]]), array([[2, 3],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[6, 7]]), array([[4],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[8]])]<br>
\-\-\-\-\-\-\-\-<br>
[array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]])]<br>
<br>

`numpy.hsplit(배열, 개수 또는 구간)`를 이용하여 `배열`을 수평 방향 또는 **열(column)** 단위로 분할합니다.

`numpy.vsplit(배열, 개수 또는 구간)`를 이용하여 `배열`을 수직 방향 또는 **행(row)** 단위로 분할합니다.

축을 따라 입력된 `개수`만큼 분할을 실행하며, `구간`을 입력할 경우 해당 구간마다 잘라 `n+1`개의 배열이 생성됩니다.

<br>

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])

a = np.split(array, 2, axis=0)
b = np.split(array, 2, axis=1)

print(a)
print(b)

{% endhighlight %}

**결과**
:    
[array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]])]<br>
[array([[1, 2],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[5, 6]]), array([[3, 4],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[7, 8]])]<br>
\-\-\-\-\-\-\-\-<br>
[array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]])]<br>
<br>

`numpy.split(배열, 개수 또는 구간, 축)`를 이용하여 `배열`을 축 방향으로 분할합니다.

지정된 `축`을 따라 입력된 `개수`만큼 분할을 실행하며, `구간`을 입력할 경우 해당 구간마다 잘라 `n+1`개의 배열이 생성됩니다.

<br>

{% highlight Python %}

import numpy as np

array = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ],
    [
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
])

a = np.dsplit(array, 2)

print(a)

{% endhighlight %}

**결과**
:    
[array([[[ 1,  2],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ 5,  6]],<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[ 9, 10],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[13, 14]]]), array([[[ 3,  4],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ 7,  8]],<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[11, 12],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[15, 16]]])]<br>
<br>

`numpy.dsplit(배열, 개수 또는 구간)`를 이용하여 `배열`을 `축`을 따라 분할합니다.

해당 함수는 3차원 이상의 배열에서만 작동합니다.

`축`을 따라 입력된 `개수`만큼 분할하며, `구간`을 입력할 경우 해당 구간마다 잘라 `n+1`개의 배열이 생성됩니다.
