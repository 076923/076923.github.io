---
layout: post
title: "Python Numpy 강좌 : 제 19강 - 고유값 및 집합 연산"
tagline: "Python Numpy Unique & Set Operations"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy unique, Numpy unique_counts, Numpy unique_values, Numpy isin, Numpy intersect1d, Numpy union1d, Numpy setdiff1d, Numpy setxor1d
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-19/
comments: true
toc: true
---

## 고유값(Unique)

{% highlight Python %}

import numpy as np

array = np.array([3, 1, 2, 3, 1, 3, 5])

print(np.unique(array))

values, counts = np.unique(array, return_counts=True)
print(values)
print(counts)

{% endhighlight %}

**결과**
:    
[1 2 3 5]<br>
[1 2 3 5]<br>
[2 1 3 1]<br>
<br>

`고유값 함수(np.unique)`는 배열에서 **중복을 제거한 원소**를 **오름차순으로 정렬**해 반환합니다.

`np.unique(배열, return_index=False, return_inverse=False, return_counts=False, axis=None)`으로 고유값을 계산합니다.

`return_counts=True`로 지정하면 각 고유값이 **몇 번 등장했는지**를 담은 배열을 함께 반환합니다.

예제에서 `1`은 두 번, `2`는 한 번, `3`은 세 번, `5`는 한 번 등장하므로 `[2 1 3 1]`을 반환합니다.

반환값을 추가로 요청하면 결과는 **튜플**로 반환되므로, `values, counts = ...`처럼 나누어 할당합니다.

<br>
<br>

## 고유값 색인

{% highlight Python %}

import numpy as np

array = np.array([3, 1, 2, 3, 1, 3, 5])

values, index, inverse = np.unique(array, return_index=True, return_inverse=True)

print(index)
print(inverse)
print(values[inverse])

{% endhighlight %}

**결과**
:    
[1 2 0 6]<br>
[2 0 1 2 0 2 3]<br>
[3 1 2 3 1 3 5]<br>
<br>

`return_index=True`로 지정하면 각 고유값이 원본 배열에서 **처음 나타나는 색인**을 반환합니다.

고유값 `[1 2 3 5]`는 원본 배열의 `1`, `2`, `0`, `6`번 색인에서 처음 등장합니다.

`return_inverse=True`로 지정하면 원본 배열의 각 원소가 **고유값 배열의 몇 번째 값인지**를 반환합니다.

원본의 첫 번째 원소 `3`은 고유값 배열의 `2`번 색인에 있으므로 `inverse`의 첫 번째 값은 `2`입니다.

그러므로 `values[inverse]`는 **원본 배열을 그대로 복원**합니다.

- Tip : `inverse`는 문자열이나 범주형 데이터를 **정수 번호로 변환**할 때 유용합니다. `["사과", "배", "사과"]`의 `inverse`는 `[1, 0, 1]`처럼 각 범주에 번호를 매긴 결과가 됩니다.

<br>
<br>

## 최빈값과 빈도

{% highlight Python %}

import numpy as np

array = np.array([3, 1, 2, 3, 1, 3, 5])

values, counts = np.unique(array, return_counts=True)

print(values[np.argmax(counts)])
print(dict(zip(values.tolist(), counts.tolist())))

{% endhighlight %}

**결과**
:    
3<br>
{1: 2, 2: 1, 3: 3, 5: 1}<br>
<br>

고유값과 빈도를 함께 사용하면 **가장 많이 등장한 값(최빈값)**을 계산할 수 있습니다.

`np.argmax(counts)`는 빈도가 가장 큰 위치를 반환하므로, 이 색인으로 `values`를 조회하면 최빈값인 `3`을 얻습니다.

빈도가 같은 값이 여러 개라면 `np.argmax`는 처음 나타나는 위치를 반환하므로, **가장 작은 값**이 최빈값으로 선택됩니다.

`zip` 함수로 두 배열을 묶으면 `값: 빈도` 형태의 **사전(dict)**으로 변환할 수 있습니다. `tolist()`는 numpy 자료형을 파이썬 기본 자료형으로 변환합니다.

- Tip : 배열이 `0` 이상의 정수로만 구성되어 있다면 `np.bincount(배열)`로 각 정수의 빈도를 더 빠르게 계산할 수 있습니다.

<br>
<br>

## 다차원 고유값

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2],
    [3, 4],
    [1, 2],
    [5, 6]
])

print(np.unique(array))
print(np.unique(array, axis=0))

{% endhighlight %}

**결과**
:    
[1 2 3 4 5 6]<br>
[[1 2]<br>
&nbsp;[3 4]<br>
&nbsp;[5 6]]<br>
<br>

`축(axis)`을 지정하지 않으면 다차원 배열을 **1차원으로 펼친 다음** 고유값을 계산합니다.

`axis=0`으로 지정하면 **행 전체를 하나의 단위**로 비교해, 중복된 행을 제거합니다.

예제에서 `[1, 2]` 행이 두 번 등장하므로, 한 번만 남기고 제거됩니다.

<br>
<br>

## 고유값 전용 함수

{% highlight Python %}

import numpy as np

array = np.array([3, 1, 2, 3, 1, 3, 5])

result = np.unique_counts(array)
print(result.values, result.counts)

print(np.unique_values(np.array([3, 1, 2, 3])))

{% endhighlight %}

**결과**
:    
[1 2 3 5] [2 1 3 1]<br>
[3 1 2]<br>
<br>

`NumPy 2.0`부터 반환값의 종류에 따라 나뉜 고유값 전용 함수가 추가되었습니다.

| 함수 | 반환 |
|:---:|:---|
| np.unique_values | 고유값 |
| np.unique_counts | 고유값과 빈도 (values, counts) |
| np.unique_inverse | 고유값과 역색인 (values, inverse_indices) |
| np.unique_all | 고유값, 첫 색인, 역색인, 빈도 |

<br>

전용 함수는 **이름이 있는 튜플**을 반환하므로 `result.values`, `result.counts`처럼 속성으로 접근할 수 있습니다.

단, `np.unique_values`는 `NumPy 2.3`부터 정렬을 사용하지 않는 더 빠른 알고리즘으로 바뀌어, 예제처럼 결과가 **정렬되지 않을 수 있습니다.** 정렬된 결과가 필요하다면 `np.unique`를 사용합니다.

- Tip : `np.unique`에도 `sorted` 매개변수가 추가되었습니다. 순서가 상관없다면 `np.unique(배열, sorted=False)`로 정렬을 생략해 계산 속도를 높일 수 있습니다.

<br>
<br>

## 집합 연산(Set Operations)

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7])

print(np.intersect1d(a, b))
print(np.union1d(a, b))
print(np.setdiff1d(a, b))
print(np.setdiff1d(b, a))
print(np.setxor1d(a, b))

{% endhighlight %}

**결과**
:    
[4 5]<br>
[1 2 3 4 5 6 7]<br>
[1 2 3]<br>
[6 7]<br>
[1 2 3 6 7]<br>
<br>

numpy는 두 배열을 **집합**으로 간주해 계산하는 집합 연산 함수를 제공합니다.

| 함수 | 연산 | 의미 |
|:---:|:---:|:---|
| np.intersect1d(a, b) | 교집합 | a와 b에 모두 있는 값 |
| np.union1d(a, b) | 합집합 | a나 b 중 하나라도 있는 값 |
| np.setdiff1d(a, b) | 차집합 | a에는 있지만 b에는 없는 값 |
| np.setxor1d(a, b) | 대칭 차집합 | a와 b 중 한쪽에만 있는 값 |

<br>

모든 집합 연산 함수는 결과에서 **중복을 제거하고 오름차순으로 정렬**한 1차원 배열을 반환합니다.

차집합은 **순서에 따라 결과가 달라집니다.** `np.setdiff1d(a, b)`는 `[1 2 3]`을, `np.setdiff1d(b, a)`는 `[6 7]`을 반환합니다.

함수 이름의 `1d`는 입력 배열이 다차원이라도 **1차원으로 펼친 다음** 계산한다는 의미입니다.

- Tip : 두 입력 배열에 중복이 없다는 것을 이미 알고 있다면 `assume_unique=True`를 지정해 중복 제거 과정을 생략하고 계산 속도를 높일 수 있습니다. 중복이 있는 배열에 사용하면 잘못된 결과가 반환될 수 있습니다.

<br>
<br>

## 교집합 색인

{% highlight Python %}

import numpy as np

a = np.array([3, 1, 2, 3])
b = np.array([2, 3, 3, 4])

values, indexA, indexB = np.intersect1d(a, b, return_indices=True)

print(values)
print(indexA)
print(indexB)

{% endhighlight %}

**결과**
:    
[2 3]<br>
[2 0]<br>
[0 1]<br>
<br>

`np.intersect1d`에 `return_indices=True`를 지정하면 교집합의 값이 각 배열의 **어느 색인에 있는지**를 함께 반환합니다.

교집합 `[2 3]`에서 `2`는 `a`의 `2`번 색인과 `b`의 `0`번 색인에, `3`은 `a`의 `0`번 색인과 `b`의 `1`번 색인에 있습니다.

값이 여러 번 등장한다면 **처음 나타나는 색인**을 반환합니다.

두 데이터에서 공통된 아이디를 찾고, 각 데이터의 해당 행을 가져오는 것처럼 **두 배열을 연결**할 때 사용합니다.

<br>
<br>

## 포함 여부(Isin)

{% highlight Python %}

import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
target = [2, 4, 6]

mask = np.isin(array, target)

print(mask)
print(array[mask])
print(np.isin(array, target, invert=True))

{% endhighlight %}

**결과**
:    
[[False&nbsp;&nbsp;True False]<br>
&nbsp;[&nbsp;True False&nbsp;&nbsp;True]]<br>
[2 4 6]<br>
[[&nbsp;True False&nbsp;&nbsp;True]<br>
&nbsp;[False&nbsp;&nbsp;True False]]<br>
<br>

`포함 여부 함수(np.isin)`는 배열의 각 원소가 **목록에 포함되어 있는지**를 불(bool) 배열로 반환합니다.

`np.isin(배열, 목록, invert=False)`로 포함 여부를 계산합니다.

집합 연산 함수와 달리, 반환되는 배열은 **원본 배열과 같은 형태**를 유지합니다.

그러므로 반환된 불 배열을 그대로 **불리언 인덱싱**에 사용하거나, `np.where`의 조건식으로 사용할 수 있습니다.

`invert=True`로 지정하면 목록에 **포함되지 않은** 원소가 `True`가 됩니다. 이는 `~np.isin(배열, 목록)`과 같은 결과이지만, 더 효율적으로 계산됩니다.

- Tip : 파이썬의 `in` 연산자는 배열 전체에 대해 하나의 참/거짓만 반환합니다. 원소마다 포함 여부를 확인하려면 `np.isin`을 사용합니다. 과거에 사용하던 `np.in1d`는 `NumPy 2.0`에서 사용 중단이 예고되었으며, 최신 버전에서는 제거되어 `AttributeError`가 발생합니다.

<br>
<br>

## 결측값의 고유값

{% highlight Python %}

import numpy as np

array = np.array([1.0, np.nan, 2.0, np.nan])

print(np.unique(array))
print(np.unique(array, equal_nan=False))

{% endhighlight %}

**결과**
:    
[ 1.&nbsp;&nbsp;2. nan]<br>
[ 1.&nbsp;&nbsp;2. nan nan]<br>
<br>

`NaN`은 자기 자신과도 같지 않은 특수한 값이므로, 고유값을 계산할 때 별도의 규칙이 적용됩니다.

`np.unique`의 `equal_nan` 매개변수의 기본값은 `True`이며, 여러 개의 `NaN`을 **하나의 고유값**으로 취급합니다.

`equal_nan=False`로 지정하면 모든 `NaN`을 **서로 다른 값**으로 취급해, 등장한 횟수만큼 결과에 포함됩니다.
