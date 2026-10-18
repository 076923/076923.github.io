---
layout: post
title: "Python Numpy 강좌 : 제 5강 - 슬라이싱"
tagline: "Python Numpy slicing"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy slicing, Numpy view, Numpy copy, Fancy Indexing, Boolean Indexing
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-5/
comments: true
toc: true
---

## 슬라이싱

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(a)
print("\n")
print(a[3:])
print("\n")
print(a[1:-1]) 
print("\n")
print(a[0:3:2])

{% endhighlight %}

**결과**
:    
[1 2 3 4 5]<br>
<br>
[4 5]<br>
<br>
[2 3 4]<br>
<br>
[1 3]<br>
<br>

`배열[a:b:c]`를 이용하여  **배열의 일부를 잘라 표시할 수 있습니다.** 

`a`는 **시작 index**, `b`는 **끝 index**, `c`는 **간격**을 의미합니다. 시작 index는 결과에 **포함**되지만, 끝 index는 결과에 **포함되지 않습니다.**

`index`는 `0 ~ len-1`까지 존재하며, 아무것도 입력하지 않고 `:`로 사용할 경우, 모든 `행` 또는 `열`을 의미합니다.

`:n`으로 사용할 경우 `0 ~ n-1`까지의 원소를 의미하며 `n:`으로 사용할 경우, `n ~ len-1`까지의 원소를 의미합니다.

`음수 index`는 배열의 뒤에서부터 센 위치를 의미합니다. `-1`은 **마지막 원소(len-1)**를 의미하므로, `a[1:-1]`은 끝 index인 마지막 원소를 제외한 `1 ~ len-2`까지의 원소를 반환합니다.

<br>

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(a[::-1])
print(a[-2:])
print(a[:-2])

{% endhighlight %}

**결과**
:    
[5 4 3 2 1]<br>
[4 5]<br>
[1 2 3]<br>
<br>

`간격`에 음수를 입력할 경우, 배열을 **역순**으로 탐색합니다. `[::-1]`은 배열을 뒤집은 결과를 반환합니다.

`[-2:]`는 **마지막 두 개의 원소**를, `[:-2]`는 **마지막 두 개의 원소를 제외한** 원소를 반환합니다.

<br>

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

print(a)
print("\n")
print(a[:, 1:])
print("\n")
print(a[0:1,0:2]) 

{% endhighlight %}

**결과**
:    
[[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>
<br>
[[2 3]<br>
 [5 6]<br>
 [8 9]]<br>
<br>
[[1 2]]<br>
<br>

`배열[a:b, c:d]`를 이용하여 **배열의 일부를 잘라 표시할 수 있습니다.**

동일하게 `배열[a:b:e, c:d:f]`를 이용하여 `e`와 `f`를 **간격**으로 사용할 수 있습니다.

`a` ~ `b`는 표시할 `행`의 위치를 의미하며, `c` ~ `d`는 표시할 `열`의 위치를 의미합니다.

<br>

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]])

print(a)
print("\n")
print(a[::2, ::2])

{% endhighlight %}
**결과**
:    
[[ 1  2  3  4  5]<br>
 [ 6  7  8  9 10]<br>
 [11 12 13 14 15]]<br>
<br>
[[ 1  3  5]<br>
 [11 13 15]]<br>
<br>

배열의 슬라이싱에서 `간격`만 입력하여 배열을 출력할 수 있습니다.

`[::2, ::2]`일 경우, `행`을 `2`칸씩 띄우며, `열`도 `2`칸씩 띄워 출력합니다.

<br>
<br>

## 뷰(View)와 복사(Copy)

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a[1:4]
c = a[1:4].copy()

b[0] = 99

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[ 1 99  3  4  5]<br>
[99  3  4]<br>
[2 3 4]<br>
<br>

`list`의 슬라이싱은 새로운 `list`를 생성하지만, `numpy` 배열의 슬라이싱은 **원본 배열의 데이터를 공유하는 뷰(View)**를 반환합니다.

그러므로, 슬라이싱한 배열의 값을 변경하면 **원본 배열의 값도 함께 변경됩니다.**

원본과 별개의 배열이 필요한 경우, `배열.copy()`를 사용하여 **복사본**을 생성합니다.

- Tip : 뷰는 데이터를 복사하지 않으므로 큰 배열을 다룰 때 **메모리와 시간을 절약**할 수 있습니다. 두 배열이 메모리를 공유하는지는 `np.shares_memory(배열1, 배열2)`로 확인할 수 있습니다.

<br>
<br>

## 정수 배열 인덱싱

{% highlight Python %}

import numpy as np

a = np.array([10, 20, 30, 40, 50])
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

print(a[[0, 2, 4]])
print(b[[0, 2]])
print(b[[0, 1, 2], [2, 1, 0]])

{% endhighlight %}

**결과**
:    
[10 30 50]<br>
[[1 2 3]<br>
&nbsp;[7 8 9]]<br>
[3 5 7]<br>
<br>

`배열[[index1, index2, ...]]`처럼 **index를 담은 배열**을 사용하여 원하는 위치의 원소를 한 번에 가져올 수 있습니다. 이를 `정수 배열 인덱싱(Integer Array Indexing)` 또는 `팬시 인덱싱(Fancy Indexing)`이라 합니다.

2차원 배열에 index 배열을 하나만 사용할 경우, 해당 `행`들을 반환합니다.

`배열[[행1, 행2, ...], [열1, 열2, ...]]`처럼 두 개의 index 배열을 사용할 경우, 같은 위치의 값끼리 짝을 지어 `(0, 2)`, `(1, 1)`, `(2, 0)` 위치의 원소를 반환합니다.

- Tip : 정수 배열 인덱싱은 슬라이싱과 다르게 뷰가 아닌 **복사본**을 반환합니다.

<br>
<br>

## 불리언 인덱싱

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

print(a > 4)
print(a[a > 4])
print(a[(a > 2) & (a % 2 == 0)])

a[a > 4] = 0
print(a)

{% endhighlight %}

**결과**
:    
[[False False False]<br>
&nbsp;[False  True  True]<br>
&nbsp;[ True  True  True]]<br>
[5 6 7 8 9]<br>
[4 6 8]<br>
[[1 2 3]<br>
&nbsp;[4 0 0]<br>
&nbsp;[0 0 0]]<br>
<br>

배열에 `비교 연산`을 적용하면 각 원소의 비교 결과를 담은 **Boolean 배열**이 반환됩니다.

`배열[Boolean 배열]`을 사용할 경우, 값이 `True`인 위치의 원소만 골라 **1차원 배열**로 반환합니다. 이를 `불리언 인덱싱(Boolean Indexing)`이라 합니다.

여러 조건을 결합할 때는 `and`, `or` 대신 `&`, `|`, `~` 연산자를 사용하며, 연산자 우선순위 때문에 **각 조건을 소괄호로 감싸야 합니다.**

불리언 인덱싱에 값을 대입할 경우, 조건을 만족하는 원소의 값만 변경할 수 있습니다.

- Tip : 조건에 따라 서로 다른 값을 선택하여 새로운 배열을 만드는 방법은 [15강 - 조건 반환][15강]에서 다룹니다.

[15강]: https://076923.github.io/posts/Python-numpy-15/
