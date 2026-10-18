---
layout: post
title: "Python Numpy 강좌 : 제 18강 - 정렬"
tagline: "Python Numpy Sort"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy sort, Numpy argsort, Numpy lexsort, Numpy partition, Numpy argpartition, Numpy searchsorted
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-18/
comments: true
toc: true
---

## 정렬(Sort)

{% highlight Python %}

import numpy as np

array = np.array([
    [5, 1, 4],
    [2, 6, 3]
])

print(np.sort(array))
print(np.sort(array, axis=0))
print(np.sort(array, axis=None))
print(np.sort(array)[:, ::-1])

{% endhighlight %}

**결과**
:    
[[1 4 5]<br>
&nbsp;[2 3 6]]<br>
[[2 1 3]<br>
&nbsp;[5 6 4]]<br>
[1 2 3 4 5 6]<br>
[[5 4 1]<br>
&nbsp;[6 3 2]]<br>
<br>

`정렬 함수(np.sort)`는 배열의 원소를 **오름차순**으로 정렬한 **새로운 배열**을 반환합니다.

`np.sort(배열, axis=축, kind=알고리즘)`으로 배열을 정렬합니다.

`축(axis)`의 기본값은 `-1`이며, **마지막 축을 따라** 정렬합니다. 2차원 배열에서는 각 행의 원소를 정렬합니다.

`axis=0`으로 지정하면 각 열의 원소를 정렬하며, `axis=None`으로 지정하면 배열을 **1차원으로 펼친 다음** 정렬합니다.

축을 따라 정렬할 때는 행이나 열이 **독립적으로** 정렬됩니다. 그러므로 `axis=0`의 결과처럼 원래 같은 행에 있던 원소들이 서로 다른 행으로 흩어질 수 있습니다.

`np.sort` 함수에는 **내림차순** 매개변수가 없습니다. 내림차순이 필요하다면 오름차순으로 정렬한 다음 `[:, ::-1]`처럼 **역순 슬라이싱**을 적용합니다.

- Tip : 역순 슬라이싱은 데이터를 복사하지 않는 **뷰(View)**를 반환합니다. 독립적인 배열이 필요하다면 `.copy()`를 함께 사용합니다.

<br>
<br>

## 제자리 정렬(In-place Sort)

{% highlight Python %}

import numpy as np

array = np.array([5, 1, 4, 2, 3])
array.sort()
print(array)

other = np.array([5, 1, 4, 2, 3])
print(np.sort(other))
print(other)

{% endhighlight %}

**결과**
:    
[1 2 3 4 5]<br>
[1 2 3 4 5]<br>
[5 1 4 2 3]<br>
<br>

`배열.sort()` 메서드는 새로운 배열을 만들지 않고, **원본 배열 자체를 정렬**합니다.

메서드의 반환값은 `None`이므로, `array = array.sort()`처럼 사용하면 배열이 사라지게 됩니다.

반면, `np.sort(배열)` 함수는 정렬된 **사본**을 반환하며 원본 배열은 변경하지 않습니다.

큰 배열을 정렬한 다음 원본이 더 이상 필요하지 않다면, 메모리를 추가로 사용하지 않는 `배열.sort()`가 효율적입니다.

<br>
<br>

## 정렬 색인 반환(Argsort)

{% highlight Python %}

import numpy as np

score = np.array([70, 95, 80, 60])
name = np.array(["A", "B", "C", "D"])

index = np.argsort(score)

print(index)
print(score[index])
print(name[index])
print(name[np.argsort(-score)])

{% endhighlight %}

**결과**
:    
[3 0 2 1]<br>
[60 70 80 95]<br>
['D' 'A' 'C' 'B']<br>
['B' 'C' 'A' 'D']<br>
<br>

`정렬 색인 함수(np.argsort)`는 정렬된 값 대신, **정렬했을 때의 원소 순서를 나타내는 색인**을 반환합니다.

`index`의 `[3 0 2 1]`은 **가장 작은 값이 3번 색인**에, 그다음 값이 0번 색인에 있다는 의미입니다.

이 색인 배열을 다시 원본의 색인값으로 사용하면 `score[index]`처럼 정렬된 배열을 얻을 수 있습니다.

`np.argsort`의 가장 큰 장점은 **같은 순서를 다른 배열에도 적용**할 수 있다는 점입니다. 점수를 기준으로 구한 색인으로 `name` 배열을 정렬하면, 점수가 낮은 순서의 이름을 얻을 수 있습니다.

내림차순 색인이 필요하다면 `np.argsort(-score)`처럼 **부호를 반전**한 배열의 색인을 구합니다.

- Tip : `np.argsort(score)[::-1]`도 내림차순 색인을 반환하지만, 값이 같은 원소들의 순서까지 뒤집힙니다. 동점자의 원래 순서를 유지해야 한다면 부호를 반전하고 `kind='stable'`을 함께 사용합니다.

<br>
<br>

## 다차원 정렬 색인

{% highlight Python %}

import numpy as np

array = np.array([
    [5, 1, 4],
    [2, 6, 3]
])

print(np.argsort(array, axis=1))
print(np.argsort(array, axis=0))

{% endhighlight %}

**결과**
:    
[[1 2 0]<br>
&nbsp;[0 2 1]]<br>
[[1 0 1]<br>
&nbsp;[0 1 0]]<br>
<br>

다차원 배열에서도 `축(axis)`을 지정해 정렬 색인을 반환할 수 있습니다.

`axis=1`의 결과에서 첫 번째 행 `[1 2 0]`은 `[5, 1, 4]`를 정렬하려면 **1번, 2번, 0번 열의 순서**로 읽어야 한다는 의미입니다.

`axis=0`의 결과에서 첫 번째 열 `[1 0]`은 `[5, 2]`를 정렬하려면 **1번 행, 0번 행의 순서**로 읽어야 한다는 의미입니다.

다차원 정렬 색인으로 원본 배열의 값을 가져오려면 `np.take_along_axis(배열, 색인, axis=축)`을 사용합니다. `np.take_along_axis(array, np.argsort(array, axis=1), axis=1)`은 `np.sort(array, axis=1)`과 같은 결과를 반환합니다.

<br>
<br>

## 다중 키 정렬(Lexsort)

{% highlight Python %}

import numpy as np

grade = np.array([2, 1, 2, 1, 3])
score = np.array([80, 90, 70, 60, 85])

index = np.lexsort((score, grade))

print(index)
print(grade[index])
print(score[index])

{% endhighlight %}

**결과**
:    
[3 1 2 0 4]<br>
[1 1 2 2 3]<br>
[60 90 70 80 85]<br>
<br>

`다중 키 정렬 함수(np.lexsort)`는 **여러 개의 기준**으로 정렬했을 때의 색인을 반환합니다.

`np.lexsort((보조 키, 주 키))`로 정렬 색인을 계산합니다.

`np.lexsort`는 **마지막에 입력한 키를 주 키**로 사용합니다. 주 키의 값이 같은 원소끼리는 그 앞의 키로 순서를 정합니다.

예제는 `grade`를 기준으로 먼저 정렬하고, 학년이 같은 학생끼리는 `score`를 기준으로 정렬합니다.

결과를 확인하면 학년이 `1`인 학생은 `60`, `90`의 순서로, 학년이 `2`인 학생은 `70`, `80`의 순서로 정렬된 것을 확인할 수 있습니다.

- Tip : 엑셀의 "정렬 기준 추가"와 같은 기능입니다. 키를 입력하는 순서가 일반적인 생각과 **반대**라는 점에 주의합니다.

<br>
<br>

## 부분 정렬(Partition)

{% highlight Python %}

import numpy as np

array = np.array([7, 2, 9, 1, 5, 3, 8])

p = np.partition(array, 3)
print(p[3])
print(np.all(p[:3] <= p[3]), np.all(p[4:] >= p[3]))

top3 = np.partition(array, -3)[-3:]
print(np.sort(top3))

index = np.argpartition(array, 3)[:3]
print(np.sort(array[index]))

{% endhighlight %}

**결과**
:    
5<br>
True True<br>
[7 8 9]<br>
[1 2 3]<br>
<br>

`부분 정렬 함수(np.partition)`는 배열 전체를 정렬하지 않고, **지정한 위치의 원소만 올바른 자리**에 놓습니다.

`np.partition(배열, k)`는 정렬했을 때 `k`번째에 올 원소를 `k`번 색인에 배치합니다. 그리고 그보다 **작거나 같은 원소는 앞쪽**에, **크거나 같은 원소는 뒤쪽**에 배치합니다.

예제에서 `p[3]`은 정렬했을 때 네 번째로 작은 값인 `5`이며, 앞쪽의 세 원소는 모두 `5` 이하, 뒤쪽의 세 원소는 모두 `5` 이상입니다.

단, 앞쪽과 뒤쪽 그룹 **내부의 순서는 보장되지 않습니다.** 작은 배열에서는 우연히 정렬된 것처럼 보일 수 있지만, 이에 의존해서는 안 됩니다.

`k`에 음수를 입력하면 뒤에서부터 위치를 계산합니다. `np.partition(array, -3)[-3:]`은 **가장 큰 세 개의 값**을 반환하며, 필요하다면 그 세 개만 다시 정렬합니다.

`np.argpartition`은 부분 정렬의 **색인**을 반환합니다. `np.argsort`와 `np.argpartition`의 관계는 `np.sort`와 `np.partition`의 관계와 같습니다.

- Tip : 전체 정렬은 원소 개수가 `n`일 때 `O(n log n)`의 시간이 걸리지만, 부분 정렬은 평균 `O(n)`의 시간이 걸립니다. 수백만 개의 점수 중 상위 10개만 필요한 경우처럼, **일부 순위만 필요할 때** 부분 정렬이 훨씬 빠릅니다.

<br>
<br>

## 삽입 위치 탐색(Searchsorted)

{% highlight Python %}

import numpy as np

array = np.array([1, 3, 5, 7, 9])

print(np.searchsorted(array, 6))
print(np.searchsorted(array, 5))
print(np.searchsorted(array, 5, side='right'))
print(np.searchsorted(array, [0, 4, 10]))

{% endhighlight %}

**결과**
:    
3<br>
2<br>
3<br>
[0 2 5]<br>
<br>

`삽입 위치 탐색 함수(np.searchsorted)`는 **정렬된 배열**에 값을 삽입했을 때, 정렬 순서가 유지되는 **색인**을 반환합니다.

`np.searchsorted(정렬된 배열, 값, side=방향)`으로 삽입 위치를 계산합니다.

`6`은 `5`와 `7` 사이에 들어가야 하므로 `3`을 반환합니다.

배열에 같은 값이 이미 있는 경우, `side='left'`(기본값)는 **기존 값의 앞쪽** 색인을, `side='right'`는 **기존 값의 뒤쪽** 색인을 반환합니다.

여러 개의 값을 배열로 입력하면, 각 값의 삽입 위치를 한 번에 반환합니다. 배열의 모든 값보다 크다면 배열의 길이인 `5`를 반환합니다.

`np.searchsorted`는 **이진 탐색(Binary Search)**으로 동작하므로, 정렬된 배열에서 매우 빠르게 위치를 찾을 수 있습니다. 점수를 구간별 등급으로 나누는 것처럼, 값이 어느 구간에 속하는지 계산할 때 유용합니다.

- Tip : 입력 배열이 정렬되어 있지 않으면 오류 없이 **잘못된 결과**를 반환합니다. 정렬되지 않은 배열이라면 `np.sort`로 먼저 정렬하거나, `sorter` 매개변수에 `np.argsort`의 결과를 전달합니다.

<br>
<br>

## 정렬 알고리즘과 특수한 값

{% highlight Python %}

import numpy as np

text = np.array(["banana", "apple", "cherry"])
value = np.array([3.0, np.nan, 1.0, 2.0])

print(np.sort(text))
print(np.sort(value))

{% endhighlight %}

**결과**
:    
['apple' 'banana' 'cherry']<br>
[ 1.  2.  3. nan]<br>
<br>

`kind` 매개변수로 정렬 알고리즘을 선택할 수 있으며, `'quicksort'`(기본값), `'mergesort'`, `'heapsort'`, `'stable'`을 사용할 수 있습니다.

실제 구현은 자료형과 하드웨어에 따라 더 빠른 알고리즘으로 대체될 수 있으므로, 알고리즘의 이름보다는 **안정성(Stability)** 여부가 중요합니다.

`안정 정렬(Stable Sort)`은 값이 같은 원소들의 **원래 순서를 유지**하는 정렬입니다. `kind='stable'` 또는 `kind='mergesort'`를 지정하면 안정 정렬이 보장되며, 기본값인 `'quicksort'`는 안정성을 보장하지 않습니다.

문자열 배열은 **사전 순서**로 정렬됩니다. 대문자는 소문자보다 앞에 정렬된다는 점에 주의합니다.

실수 배열에 `NaN`이 포함되어 있다면, `NaN`은 **항상 배열의 끝**으로 정렬됩니다. `NaN`은 어떤 값과 비교해도 참/거짓이 정해지지 않으므로, 별도의 규칙으로 처리됩니다.

<br>
<br>

## 정렬 함수 정리

| 함수 | 반환 | 설명 |
|:---:|:---:|:---|
| np.sort | 정렬된 사본 | 축을 따라 오름차순 정렬 |
| 배열.sort | None | 원본 배열을 직접 정렬 |
| np.argsort | 색인 | 정렬했을 때의 원소 순서 |
| np.lexsort | 색인 | 여러 키를 기준으로 정렬, 마지막 키가 주 키 |
| np.partition | 부분 정렬된 사본 | k번째 원소만 올바른 위치에 배치 |
| np.argpartition | 색인 | 부분 정렬의 색인 |
| np.searchsorted | 색인 | 정렬된 배열에서 삽입 위치 탐색 |

<br>

- Tip : 가장 큰 값이나 작은 값 하나의 **위치**만 필요하다면 정렬할 필요 없이 `np.argmax`, `np.argmin`을 사용합니다. 최댓값이 여러 개라면 **처음 나타나는 색인**을 반환합니다.
