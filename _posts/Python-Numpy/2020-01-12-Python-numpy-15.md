---
layout: post
title: "Python Numpy 강좌 : 제 15강 - 조건 반환"
tagline: "Python Numpy Where"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Where, Numpy nonzero, Numpy argwhere, Numpy select
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
(array([1, 2, 2, 2]), array([2, 0, 1, 2]))<br>
[6 7 8 9]<br>
[[1 2 3]<br>
&nbsp;[4 5 9]<br>
&nbsp;[9 9 9]]<br>
<br>

`조건 반환 함수(np.where)`는 조건식의 결과에 따라 **조건을 만족하는 요소의 색인**을 반환하거나, **참과 거짓에 따라 서로 다른 값을 선택**한 배열을 반환하는 함수입니다.

**비교 함수(np.greater, np.greater_equal)**나 비교 연산자(`>`, `>=`)는 조건의 결과를 `True`, `False`로 구성된 **불(bool) 배열**로만 반환합니다.

`조건 반환 함수(np.where)`는 이 불 배열을 조건식으로 받아, 색인이나 새로운 값으로 변환할 수 있습니다.

`np.where(조건식)` 또는 `np.where(조건식, 참 값, 거짓 값)`으로 조건식을 계산합니다.

**참 값**과 **거짓 값**을 입력하지 않은 기본 함수는 조건에 만족하는 배열의 `색인값` 튜플을 반환합니다. 이 형태는 `np.nonzero(조건식)`과 동일합니다.

**참 값**과 **거짓 값**은 둘 다 입력하거나 둘 다 생략해야 하며, 하나만 입력할 수는 없습니다.

예시의 `whereDefault`의 **첫 번째 값의 배열은 행**을 의미하며, **두 번째 값의 배열은 열**을 의미합니다.

즉, `array` 배열에서 `array > 5`를 만족하는 원소값의 색인은 `(1, 2)`, `(2, 0)`, `(2, 1)`, `(2, 2)`에 위치합니다.

이 색인 배열을 `array` 배열의 색인값으로 사용한다면, 조건에 만족하는 값인 `[6, 7, 8, 9]`를 반환합니다.

만약, **참 값**과 **거짓 값**에 값을 할당하면, 조건을 만족하는 위치에는 참 값인 `np.max(array)`의 `9`가 할당되며, 만족하지 않는 위치에는 거짓 값인 `array`의 원래 값이 할당됩니다.

**참 값**과 **거짓 값**에는 스칼라나 배열을 입력할 수 있으며, 조건식과 형태가 다르다면 **브로드캐스팅**되어 계산됩니다.

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

조건문(if)의 `and`나 `or`처럼 사용하여 복잡한 조건식을 구성할 수 있습니다. 부정은 `~(NOT)`을 사용합니다.

단, 파이썬의 `and`, `or`, `not` 키워드는 배열에 사용할 수 없습니다. 배열 전체를 하나의 참/거짓으로 판단할 수 없기 때문에 `ValueError`가 발생합니다.

또한, `&`와 `|`는 비교 연산자(`>`, `<`)보다 **우선순위가 높으므로**, 각 조건식을 반드시 `소괄호(())`로 감싸야 합니다.

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

`whereSum`은 `axis = 1`에 대한 합계가 **10**을 넘어가는 행은 원래 값을 유지하고, 넘지 못하는 행의 값은 모두 **0**으로 변경하는 조건으로 설정합니다.

행별 합계는 `[6, 15, 24]`이며, `reshape(3, -1)`로 `(3, 1)` 형태로 변경하면 **브로드캐스팅**을 통해 같은 행의 모든 요소에 같은 조건이 적용됩니다.

위 조건으로 값을 반환할 경우, `[1, 2, 3]`의 합계는 `10`을 넘지 못해, 내부의 값은 모두 **0**으로 변경됩니다.

더 심화된 검색을 진행하는 경우, `조건식`을 더 복잡하게 구성해도 됩니다.

하지만, `참 값`에 `조건 반환 함수(np.where)`를 또다시 적용하는 경우, `조건식`을 만족하고 나온 값에 대해 다시 한 번 더 `조건식`을 적용합니다.

두 번째로 적용된 조건식은 `홀수` 값만 유지합니다. 그러므로, `whereSum`의 값에서 `홀수`만 남게 됩니다.

결과는 예제처럼 `5, 7, 9`만 유지하며, 나머지 값은 모두 `0`으로 변경합니다.

<br>
<br>

## 색인 반환(Nonzero & Argwhere)

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(np.nonzero(array > 5))
print(np.argwhere(array > 5))
print(array[array > 5])

{% endhighlight %}

**결과**
:    
(array([1, 2, 2, 2]), array([2, 0, 1, 2]))<br>
[[1 2]<br>
&nbsp;[2 0]<br>
&nbsp;[2 1]<br>
&nbsp;[2 2]]<br>
[6 7 8 9]<br>
<br>

`np.nonzero`는 0이 아닌(**참인**) 요소의 색인을 **축별 배열의 튜플**로 반환합니다. 조건식만 입력한 `np.where`와 같은 결과이며, 공식 문서에서도 이 용도로는 `np.nonzero`의 사용을 권장합니다.

`np.argwhere`는 같은 색인을 **요소별 좌표**로 묶어 `(요소 개수, 차원 수)` 형태의 배열로 반환합니다. 조건을 만족하는 위치를 하나씩 순회할 때 편리합니다.

조건을 만족하는 **값**만 필요하다면, 색인을 구하지 않고 불 배열을 그대로 색인으로 사용하는 `array[조건식]` 형태가 가장 간단합니다.

<br>
<br>

## 다중 선택(Select)

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

condlist = [array < 4, array < 7]
choicelist = [array * 10, array * 100]

print(np.select(condlist, choicelist, default=0))

{% endhighlight %}

**결과**
:    
[[ 10  20  30]<br>
&nbsp;[400 500 600]<br>
&nbsp;[  0   0   0]]<br>
<br>

`다중 선택 함수(np.select)`는 여러 개의 조건과 그에 대응하는 값을 목록으로 받아, 조건에 따라 서로 다른 값을 선택합니다.

`np.select(조건 목록, 값 목록, 기본값)`으로 사용하며, 조건 목록과 값 목록의 길이는 같아야 합니다.

조건 목록은 **앞에서부터 순서대로** 검사되며, 여러 조건을 동시에 만족하는 경우 **먼저 나온 조건의 값**이 선택됩니다.

예시에서 `1 ~ 3`은 두 조건을 모두 만족하지만 첫 번째 조건이 우선하므로 `10`배가 되며, `4 ~ 6`은 두 번째 조건에 따라 `100`배가 됩니다.

어떤 조건도 만족하지 않는 `7 ~ 9`는 `기본값(default)`인 `0`이 할당됩니다.

`np.where`를 여러 번 중첩하는 대신 `np.select`를 사용하면, 조건이 세 개 이상인 경우에도 읽기 쉬운 코드를 작성할 수 있습니다.
