---
layout: post
title: "Python Numpy 강좌 : 제 11강 - 병합 및 분할"
tagline: "Python Numpy stack & split"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy stack, Numpy Split, Numpy hstack, Numpy vstack, Numpy dstack, Numpy tile, Numpy c_, Numpy r_, Numpy hsplit, Numpy vsplit, Numpy dsplit, Numpy concatenate, Numpy array_split
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

`hstack`은 붙이는 방향(열)을 제외한 나머지 축의 길이가, `vstack`은 붙이는 방향(행)을 제외한 나머지 축의 길이가 같아야 합니다.

- Tip : 1차원 배열에 `hstack`을 사용하면 하나의 긴 1차원 배열이 되며, `vstack`을 사용하면 각 배열이 한 행이 되는 2차원 배열이 됩니다.

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

`numpy.stack([배열1, 배열2], axis=축)`를 이용하여 지정한 `축` 위치에 **새로운 축을 만들어** `배열1`과 `배열2`를 쌓을 수 있습니다.

`hstack`, `vstack`과 달리 결과 배열의 **차원이 하나 증가**하므로, `(2, 2)` 형태의 배열 두 개를 쌓으면 `(2, 2, 2)` 형태가 됩니다.

`stack`으로 쌓을 배열들은 **형태(shape)가 모두 같아야** 합니다.

`축`은 결과 배열의 차원 범위를 넘어갈 수 없으므로, 2차원 배열을 쌓을 경우 `0` ~ `2` (또는 `-3` ~ `-1`) 사이의 값을 사용할 수 있습니다.

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

`numpy.dstack([배열1, 배열2])`를 이용하여 **깊이 방향(세 번째 축)**으로 `배열1`과 `배열2`를 **이어 붙일 수 있습니다.**

2차원 배열의 경우, `numpy.stack([a, b], axis=2)`과 **동일한 결과를 반환합니다.**

- Tip : `dstack`은 2차원 이하의 배열을 3차원으로 변환한 뒤 세 번째 축을 따라 이어 붙입니다. 이미 3차원인 배열은 차원이 증가하지 않고 세 번째 축의 길이만 늘어나므로, `stack`과 결과가 다릅니다.

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

반복 횟수에 정수를 입력할 경우 **마지막 축(열 방향)**으로 반복합니다. `np.tile(a, (2, 1))`과 같이 `튜플`을 입력하면 축마다 반복 횟수를 지정할 수 있으며, 이 경우 **행 방향**으로 두 번 반복됩니다.

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

`np.c_`와 `np.r_`은 함수가 아닌 **인덱싱 객체**이므로, `소괄호 ()`를 사용하지 않고 `대괄호 []`를 사용합니다.

- Tip : 1차원 배열에 `np.c_`를 사용하면 각 배열을 **열 벡터로 변환**하여 이어 붙이므로, `np.c_[[1, 2], [3, 4]]`는 `(2, 2)` 형태의 배열을 반환합니다.

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
<br>

`numpy.split(배열, 개수 또는 구간, 축)`를 이용하여 `배열`을 축 방향으로 분할합니다.

지정된 `축`을 따라 입력된 `개수`만큼 분할을 실행하며, `구간`을 입력할 경우 해당 구간마다 잘라 `n+1`개의 배열이 생성됩니다.

`축`을 입력하지 않으면 `axis=0`으로 분할하므로, `hsplit`은 `axis=1`, `vsplit`은 `axis=0`으로 분할하는 `split`과 같습니다.

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

`numpy.dsplit(배열, 개수 또는 구간)`를 이용하여 `배열`을 **깊이 방향(세 번째 축, axis=2)**을 따라 분할합니다.

해당 함수는 3차원 이상의 배열에서만 작동합니다.

`축`을 따라 입력된 `개수`만큼 분할하며, `구간`을 입력할 경우 해당 구간마다 잘라 `n+1`개의 배열이 생성됩니다.

<br>
<br>

## 균등하지 않은 분할

{% highlight Python %}

import numpy as np

array = np.arange(7)

print(np.array_split(array, 3))

np.split(array, 3)

{% endhighlight %}

**결과**
:    
[array([0, 1, 2]), array([3, 4]), array([5, 6])]<br>
ValueError: array split does not result in an equal division<br>
<br>

`numpy.split`에 `개수`를 입력할 경우, 배열의 길이가 개수로 **나누어 떨어져야** 합니다. 길이가 7인 배열을 3개로 나누면 `ValueError`가 발생합니다.

`numpy.array_split(배열, 개수, 축)`을 이용하면 나누어 떨어지지 않아도 분할할 수 있습니다.

길이가 `l`인 배열을 `n`개로 나누면, 앞쪽 `l % n`개의 배열은 `l // n + 1`의 길이를, 나머지 배열은 `l // n`의 길이를 갖습니다.

<br>
<br>

## 연결과 쌓기의 차이

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

c = np.concatenate([a, b], axis=0)
d = np.concatenate([a, b], axis=1)
e = np.stack([a, b], axis=0)

print(c.shape, d.shape, e.shape)
print(d)

{% endhighlight %}

**결과**
:    
(4, 2) (2, 4) (2, 2, 2)<br>
[[1 2 5 6]<br>
 [3 4 7 8]]<br>
<br>

`numpy.concatenate([배열1, 배열2], axis=축)`을 이용하여 **기존의 축**을 따라 배열을 이어 붙일 수 있습니다.

`axis=0`은 `vstack`, `axis=1`은 `hstack`과 같은 결과를 반환하며, 결과 배열의 **차원은 증가하지 않습니다.**

반면, `stack`은 **새로운 축**을 만들어 배열을 쌓으므로 결과 배열의 차원이 하나 증가합니다.

- Tip : `hstack`, `vstack`, `dstack`은 내부적으로 `concatenate`를 사용하는 편의 함수입니다. 차원의 변화를 명확하게 표현하려면 `concatenate`와 `stack`을 사용하는 것이 좋습니다.
