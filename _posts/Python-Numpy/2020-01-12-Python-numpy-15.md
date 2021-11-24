---
layout: post
title: "Python Numpy 강좌 : 제 15강 - 조건 반환"
tagline: "Python Numpy Where"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Where
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-15/
comments: true
toc: true
---

## 조건 반환(Where)

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

whereDefault = np.where(array > 5)
whereArray = np.where(array > 5, np.max(array), array)

print(array)
print('---------')
print(whereDefault)
print(array[whereDefault])
print(whereArray)

{% endhighlight %}

**결과**
:    
[[1 2 3]<br>
&nbsp;[4 5 6]<br>
&nbsp;[7 8 9]]<br>
---------<br>
(array([1, 2, 2, 2], dtype=int64), array([2, 0, 1, 2], dtype=int64))<br>
[6 7 8 9]<br>
[[1 2 3]<br>
&nbsp;[4 5 9]<br>
&nbsp;[9 9 9]]<br>
<br>

`조건 반환 함수(np.where)`는 배열의 요솟값이 특정 조건에 만족하는 값을 반환하는 함수입니다.

**비교 함수(np.greater, np.greater_equal)** 등으로도 특정 조건을 만족하는 배열을 반환할 수 있습니다.

하지만, `조건 반환 함수(np.where)`는 더 세밀한 조건을 통해 배열의 조건을 검색할 수 있습니다.

`np.where(조건식)` 또는 `np.where(조건식, 참 값, 거짓 값)`으로 조건식을 계산합니다.

**참 값**과 **거짓 값**을 입력하지 않은 기본 함수는 조건에 만족하는 배열의 `색인값` 튜플을 반환합니다.

예시의 `whereDefault`의 **첫 번째 값의 배열은 행**을 의미하며, **두 번째 값의 배열은 열**을 의미합니다.

즉, `array` 배열에서 `array > 5`를 만족하는 원소값의 색인은 `(1, 2)`, `(2, 0)`, `(2, 1)`, `(2, 2)`에 위치합니다.

이 색인 배열을 `array` 배열의 색인값으로 사용한다면, 조건에 만족하는 값인 `[6, 7, 8, 9]`를 반환합니다.

만약, **참 값**과 **거짓 값**에 값을 할당하면, 참 값은 `np.max(array)`인 `9`가 되며, 거짓 값은 `기본값(array)`으로 할당됩니다.

<br>
<br>

## 복수 조건

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

whereAnd = np.where((array > 3) & (array < 7), 0, array)
whereOr = np.where((array < 3) | (array > 7), 0, array)

print(whereAnd)
print(whereOr)

{% endhighlight %}

**결과**
:    
[[1 2 3]<br>
&nbsp;[0 0 0]<br>
&nbsp;[7 8 9]]<br>
[[0 0 3]<br>
&nbsp;[4 5 6]<br>
&nbsp;[7 0 0]]<br>
<br>

조건 반환 함수의 **조건식**은 **참/거짓**의 형태을 갖습니다.

그러므로, 조건식 내부에 복합 조건을 사용하기 위해서는 `&(AND)`와 `|(OR)`를 적용해 다양한 조건식을 연결할 수 있습니다.

`whereAnd`는 3보다 크고, 7보다 작은 값을 0으로 변경합니다.

`whereOr`는 3보다 작거나, 7보다 큰 값을 0으로 변경합니다.

조건문(if)처럼 `and`나 `or`처럼 사용하여 복잡한 조건식을 구성할 수 있습니다. 

<br>
<br>

## 심화 검색

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

whereSum = np.where((np.sum(array, axis = 1) > 10).reshape(3, -1), array, 0)
whereSumOdd = np.where((np.sum(array, axis = 1) > 10).reshape(3, -1), np.where( array % 2 == 1, array, 0), 0)

print(whereSum)
print(whereSumOdd)

{% endhighlight %}

**결과**
:    
[[0 0 0]<br>
&nbsp;[4 5 6]<br>
&nbsp;[7 8 9]]<br>
[[0 0 0]<br>
&nbsp;[0 5 0]<br>
&nbsp;[7 0 9]]<br>
<br>

`조건 반환 함수(np.where)`는 인수에 `Numpy 배열` 자체를 조건으로 사용하거나, `참 값`, `거짓 값`에도 **조건 반환 함수**를 활용할 수 있습니다.

`whereSum`은 `axis = 1`에 대한 합계가 **10**을 넘어가는 경우, 해당 행의 값을 모두 **0**으로 변경하는 조건으로 설정합니다.

위 조건으로 값을 반환할 경우, `[1, 2, 3]`의 합계는 `10`을 넘지 못해, 내부의 값은 모두 **0**으로 변경됩니다.

더 심화된 검색을 진행하는 경우, `조건식`을 더 복잡하게 구성해도 됩니다.

하지만, `참 값`에 `조건 반환 함수(np.where)`를 또다시 적용하는 경우, `조건식`을 만족하고 나온 값에 대해 다시 한 번 더 `조건식`을 적용합니다.

두 번째로 적용된 조건식은 `홀수` 값만 출력하게 됩니다. 그러므로, `whereSum`의 값에서 `홀수`만 출력하게됩니다.

결과는 예제처럼 `5, 7, 9`만 유지하며, 나머지 값은 모두 `0`으로 변경합니다.
