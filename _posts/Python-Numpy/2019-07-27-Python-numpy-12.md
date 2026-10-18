---
layout: post
title: "Python Numpy 강좌 : 제 12강 - 브로드캐스팅"
tagline: "Python Numpy Broadcasting"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Broadcasting, Numpy broadcast_to, Numpy broadcast_shapes, Numpy Type Promotion
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-12/
comments: true
toc: true
---

## 브로드캐스팅(Broadcasting)

`브로드캐스팅`이란 Numpy 배열에서 차원의 크기가 서로 다른 배열에서도 **산술 연산을 가능하게 하는 원리입니다.**

두 배열 간 차원의 크기가 **(4, 2)**, **(2, )** 일 때 `산술 연산`을 실행한다면 **(2, )**의 배열이 **(4, 2)** 행렬의 각 행에 대해 요소별 연산을 실행할 수 있습니다.

두 배열 간의 차원의 크기가 달라도 차원의 크기가 더 큰 배열에 대해 작은 배열을 **여러 번 반복하지 않아도 되는 것을 의미합니다.**

- Tip : (2, )은 1차원 형태의 배열을 의미합니다. 즉, [a b]의 값을 지닙니다.

<br>
<br>

## 브로드캐스팅 허용 규칙

1. 두 배열의 차원(ndim)이 같지 않다면 차원이 더 낮은 배열의 형태 **앞쪽에 1을 추가**하여 차원의 수를 맞춥니다.
2.	두 배열의 형태를 **뒤쪽(마지막 축)부터** 한 축씩 비교합니다.
3.	비교한 축의 크기가 **같거나, 둘 중 하나가 1**이라면 브로드캐스팅이 가능합니다. 그렇지 않다면 `ValueError`가 발생합니다.
4.	반환된 배열의 각 축의 크기는 두 배열의 해당 축 크기 중 **더 큰 값**이 되며, 크기가 1인 축은 상대 배열의 크기만큼 **반복된 것처럼** 계산됩니다.

<br>

예를 들어 **(2, 2)** 형태와 **(2, )** 형태의 배열을 연산하는 경우, **(2, )**는 1번 규칙에 의해 **(1, 2)**로 인식됩니다.

마지막 축은 `2`와 `2`로 같고, 첫 번째 축은 `2`와 `1`이므로 브로드캐스팅이 가능하며, 결과는 **(2, 2)** 형태가 됩니다.

<br>

{% highlight Python %}

import numpy as np 

array1 = np.array([1, 2, 3, 4]).reshape(2, 2)
array2 = np.array([1.5, 2.5])

add = array1 + array2

print(add)

{% endhighlight %}

**결과**
:    
[[2.5 4.5]<br>
 [4.5 6.5]]<br>

<br>

각 배열의 차원이 달라도 브로드캐스팅 **허용 규칙에 위반되지 않는다면 연산이 가능합니다.**

또한, `형식 승격(type promotion)`을 지원해 **두 배열의 자료형(dtype)이 다르더라도 연산이 가능합니다.** 예제에서는 `int64`와 `float64` 배열을 연산했으므로 결과는 `float64`가 됩니다.

- Tip : `형식 승격` : 두 자료형의 값을 모두 표현할 수 있는 자료형을 선택하는 것을 의미합니다. `numpy.result_type(배열1, 배열2)`로 연산 결과의 자료형을 미리 확인할 수 있습니다.

<br>
<br>

## 양쪽 배열의 브로드캐스팅

{% highlight Python %}

import numpy as np

a = np.array([[0], [10], [20]])
b = np.array([1, 2, 3, 4])

result = a + b

print(a.shape, b.shape)
print(result.shape)
print(result)

{% endhighlight %}

**결과**
:    
(3, 1) (4,)<br>
(3, 4)<br>
[[ 1  2  3  4]<br>
 [11 12 13 14]<br>
 [21 22 23 24]]<br>
<br>

브로드캐스팅은 작은 배열 한쪽만 확장되는 것이 아니라, **두 배열이 동시에 확장**될 수도 있습니다.

`(4,)` 형태의 배열은 1번 규칙에 의해 `(1, 4)`로 인식됩니다. 이후 뒤쪽 축부터 비교하면 다음과 같습니다.

| 축 | a | b | 결과 |
|:---:|:---:|:---:|:---:|
| 마지막 축 | 1 | 4 | 4 |
| 첫 번째 축 | 3 | 1 | 3 |

`a`는 열 방향으로 4번, `b`는 행 방향으로 3번 반복된 것처럼 계산되어 `(3, 4)` 형태의 결과를 반환합니다.

- Tip : 브로드캐스팅은 실제로 배열을 복사하여 반복하지 않으므로, **메모리를 추가로 사용하지 않고** 연산할 수 있습니다.

<br>
<br>

## 브로드캐스팅 실패

{% highlight Python %}

import numpy as np

a = np.ones((2, 3))
b = np.array([1, 2])

print(a + b[:, np.newaxis])

a + b

{% endhighlight %}

**결과**
:    
[[2. 2. 2.]<br>
 [3. 3. 3.]]<br>
ValueError: operands could not be broadcast together with shapes (2,3) (2,)<br>
<br>

`(2, 3)` 형태와 `(2,)` 형태의 배열은 마지막 축의 크기가 `3`과 `2`로 다르고, 둘 다 1이 아니므로 **브로드캐스팅이 불가능합니다.**

브로드캐스팅은 **앞쪽 축이 아닌 뒤쪽 축부터** 맞추기 때문에, 행의 개수가 같더라도 자동으로 행 방향으로 연산되지 않습니다.

이 경우, `b[:, np.newaxis]`를 이용하여 `(2, 1)` 형태로 변경하면 마지막 축이 `1`이 되므로 브로드캐스팅이 가능해집니다.

<br>
<br>

## 브로드캐스팅 확인

{% highlight Python %}

import numpy as np

print(np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5)))

arr = np.array([1, 2, 3])
b = np.broadcast_to(arr, (2, 3))

print(b)
print(b.flags.writeable)

{% endhighlight %}

**결과**
:    
(8, 7, 6, 5)<br>
[[1 2 3]<br>
 [1 2 3]]<br>
False<br>
<br>

`numpy.broadcast_shapes(형태1, 형태2, ...)`를 이용하여 실제 연산을 수행하지 않고 **브로드캐스팅 결과의 형태**를 확인할 수 있습니다.

`numpy.broadcast_to(배열, 형태)`를 이용하여 배열을 지정한 형태로 브로드캐스팅한 배열을 반환합니다.

`broadcast_to`가 반환하는 배열은 원본 데이터를 공유하는 **읽기 전용 뷰(view)**이므로, 값을 수정할 수 없습니다.

- Tip : 브로드캐스팅된 배열의 값을 수정해야 한다면 `.copy()`를 이용하여 별도의 배열로 복제한 뒤 사용합니다.
