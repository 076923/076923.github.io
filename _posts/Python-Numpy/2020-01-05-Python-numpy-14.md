---
layout: post
title: "Python Numpy 강좌 : 제 14강 - 이어 붙이기"
tagline: "Python Numpy Append"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Append, Numpy delete, Numpy insert, Numpy concatenate
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-14/
comments: true
toc: true
---

## 이어 붙이기(Append)

{% highlight Python %}

import numpy as np

arr = np.array([
    [
        [1, 1],
        [2, 2]
    ],
    [
        [3, 3],
        [4, 4]
    ]
])


item = np.array([
    [5, 5],
    [6, 6]
])

print(arr.shape)
print(item.shape)

append = np.append(arr, item.reshape(1, 2, 2), axis=0)

print(append)

{% endhighlight %}

**결과**
:    
(2, 2, 2)<br>
(2, 2)<br>
[[[1 1]<br>
&nbsp;&nbsp;[2 2]]<br>
<br>
&nbsp;[[3 3]<br>
&nbsp;&nbsp;[4 4]]<br>
<br>
&nbsp;[[5 5]<br>
&nbsp;&nbsp;[6 6]]]<br>
<br>

`이어 붙이기 함수(np.append)`는 **내장 이어 붙이기 함수(append)**와 다르게, `축(axis)`을 지정할 경우 두 배열의 **차원 수가 같아야** 붙일 수 있습니다.

또한, 연결하는 축을 제외한 나머지 축의 크기도 같아야 합니다.

만약, `Numpy 배열`이 아닌 `List 배열`이라면 내장 이어 붙이기 함수를 통해 큰 문제 없이 이어 붙일 수 있습니다.

그 이유는 `List`는 `Container 형식`의 단순 값을 저장하는 자료형이기 때문입니다.

`Numpy 배열`은 연산을 위한 라이브러리입니다.

`arr` 변수의 차원 형태는 **(2, 2, 2)**를 가지며, `item` 변수는 **(2, 2)**를 갖습니다.

이 값을 이어 붙일 때, `(2, 2)` 배열이 앞쪽 두 축(**2, 2**, 2)에 대응하는지, 뒤쪽 두 축(2, **2, 2**)에 대응하는지 알 수 없습니다.

그러므로, `item` 배열의 차원 크기를 `arr` 배열의 차원 크기와 동일하게 구성한다음 명시적으로 `어떤 축(axis)`에 연결할지 설정합니다.

이어 붙이기 함수는 `결과 = np.append(배열1, 배열2, 축)`을 통해 배열을 이어 붙일 수 있습니다.

위의 예제는 `item`을 `(1, 2, 2)`로 변경해 `첫 번째 축(axis=0)`에 연결하는 예시입니다. 결과의 형태는 `(3, 2, 2)`가 됩니다.

만약, `마지막 축`에 연결한다면, 차원 형태를 `(2, 2, 1)`로 설정하고 `마지막 축(axis=-1)`에 연결해 사용할 수 있습니다. 결과의 형태는 `(2, 2, 3)`이 됩니다.

<br>

{% highlight Python %}

# 위 예제의 arr, item 변수를 그대로 사용합니다.
append = np.append(arr, item.reshape(2, 2, 1), axis=-1)

print(append)

{% endhighlight %}

**결과**
:    
[[[1 1 5]<br>
&nbsp;&nbsp;[2 2 5]]<br>
<br>
&nbsp;[[3 3 6]<br>
&nbsp;&nbsp;[4 4 6]]]<br>
<br>

`np.append`는 원본 배열을 변경하지 않고 **항상 새로운 배열을 반환**합니다. 그러므로 반환값을 변수에 다시 할당해야 합니다.

<br>
<br>

## 축을 지정하지 않은 이어 붙이기

{% highlight Python %}

import numpy as np

arr = np.array([[1, 2], [3, 4]])

print(np.append(arr, [5, 6]))
print(np.append(arr, [[5, 6]], axis=0))

{% endhighlight %}

**결과**
:    
[1 2 3 4 5 6]<br>
[[1 2]<br>
&nbsp;[3 4]<br>
&nbsp;[5 6]]<br>
<br>

`축(axis)`을 입력하지 않는 경우, `np.append`는 두 배열을 모두 **1차원으로 평탄화**한 다음 이어 붙입니다.

그러므로 2차원 배열에 값을 추가하더라도 결과는 1차원 배열이 됩니다.

배열의 형태를 유지하려면 `축(axis)`을 명시하고, 추가할 배열도 `[[5, 6]]`처럼 **같은 차원 수**로 구성해야 합니다.

<br>
<br>

## 빈 배열에서 이어 붙이기

{% highlight Python %}

import numpy as np

arr = np.empty((1, 2), dtype=int)

for i in range(5):
    item = np.array([[i, i]])
    arr = np.append(arr, item, axis=0) 

arr = np.delete(arr, [0, 0], axis=0)
print(arr)

{% endhighlight %}

**결과**
:    
[[0 0]<br>
&nbsp;[1 1]<br>
&nbsp;[2 2]<br>
&nbsp;[3 3]<br>
&nbsp;[4 4]]<br>
<br>

`List` 자료형에서는 `L = []`로 선언 후 **내장 이어 붙이기 함수(append)**를 활용해 자유롭게 배열의 형태를 생성하곤 했습니다.

하지만, `Numpy 배열`에서는 앞서 설명한 이유처럼 어떤 형태로 이어 붙이기를 실행할지 알 수 없습니다.

그러므로, **이어 붙일 배열의 형태를 미리 선언(np.empty)**합니다.

반복문을 통해 이어 붙이기가 완료되면, **첫 번째 행에 있는 더미 값을 제거**해야 합니다.

`제거 함수(np.delete)`를 통해 특정 요소값을 삭제합니다.

`결과 = np.delete(배열, 색인, 축)`을 통해 배열의 요소를 제거할 수 있습니다.

제거 함수는 `색인`과 `축` 값에 따라 제거할 대상을 설정합니다.

`색인`은 범위가 아닌 **제거할 위치의 목록**입니다. 정수 하나, 목록, `slice` 객체 등을 입력할 수 있습니다.

예시에서는 `첫 번째 축(axis=0)`을 대상으로 `[0, 0]`을 입력했으며, 중복된 색인은 한 번만 적용되므로 첫 번째 행만 제거됩니다. `np.delete(arr, 0, axis=0)`과 같은 결과입니다.

즉, 더미 값인 `np.empty` 값이 제거되고 이어 붙인 값만 남게됩니다.

- Tip : `np.empty`는 메모리를 초기화하지 않으므로 더미 행에는 **임의의 값**이 들어 있습니다. 반드시 제거한 다음 사용해야 합니다.

<br>
<br>

## 크기가 0인 배열에서 이어 붙이기

{% highlight Python %}

import numpy as np

arr = np.empty((0, 2), dtype=int)

for i in range(5):
    item = np.array([[i, i]])
    arr = np.append(arr, item, axis=0)

print(arr.shape)
print(arr)

{% endhighlight %}

**결과**
:    
(5, 2)<br>
[[0 0]<br>
&nbsp;[1 1]<br>
&nbsp;[2 2]<br>
&nbsp;[3 3]<br>
&nbsp;[4 4]]<br>
<br>

`np.empty((0, 2))`처럼 **첫 번째 축의 크기가 0**인 배열을 선언하면, 열의 개수는 `2`로 정해져 있지만 행은 하나도 없는 배열이 생성됩니다.

이 배열에 이어 붙이면 더미 값이 생기지 않으므로, `np.delete`로 첫 번째 행을 제거하는 과정이 필요하지 않습니다.

<br>
<br>

## 목록으로 모은 다음 변환하기

{% highlight Python %}

import numpy as np

items = []

for i in range(5):
    items.append([i, i])

arr = np.array(items)
print(arr)

{% endhighlight %}

**결과**
:    
[[0 0]<br>
&nbsp;[1 1]<br>
&nbsp;[2 2]<br>
&nbsp;[3 3]<br>
&nbsp;[4 4]]<br>
<br>

`np.append`는 호출할 때마다 **배열 전체를 새로운 메모리에 복사**합니다.

그러므로 반복문 안에서 `np.append`를 여러 번 호출하면, 요소의 개수가 늘어날수록 복사량이 계속 커져 속도가 크게 느려집니다.

반복문으로 값을 모으는 경우에는 `List`의 **내장 이어 붙이기 함수(append)**로 값을 모은 다음, 마지막에 한 번만 `np.array`로 변환하는 것이 효율적입니다.

여러 개의 Numpy 배열을 모았다면 `np.concatenate(목록, axis)`나 `np.stack(목록, axis)`으로 한 번에 연결할 수 있습니다.

- Tip : 최종 크기를 미리 알고 있다면 `np.zeros((5, 2))`처럼 배열을 먼저 할당한 다음 `arr[i] = [i, i]`로 값을 채우는 방법이 가장 빠릅니다.

<br>
<br>

## 삽입(Insert)

{% highlight Python %}

import numpy as np

arr = np.array([[1, 1], [2, 2], [3, 3]])

print(np.insert(arr, 1, [9, 9], axis=0))
print(np.insert(arr, 1, 0, axis=1))
print(np.delete(arr, [0, 2], axis=0))

{% endhighlight %}

**결과**
:    
[[1 1]<br>
&nbsp;[9 9]<br>
&nbsp;[2 2]<br>
&nbsp;[3 3]]<br>
[[1 0 1]<br>
&nbsp;[2 0 2]<br>
&nbsp;[3 0 3]]<br>
[[2 2]]<br>
<br>

`삽입 함수(np.insert)`는 배열의 **끝이 아닌 특정 위치**에 값을 추가합니다.

`결과 = np.insert(배열, 색인, 값, 축)`을 통해 `색인` 위치의 **앞에** 값을 삽입합니다.

첫 번째 예시는 `첫 번째 축(axis=0)`의 `1`번 색인 앞에 `[9, 9]` 행을 삽입합니다.

두 번째 예시는 `두 번째 축(axis=1)`의 `1`번 색인 앞에 `0`을 삽입합니다. 스칼라 값은 브로드캐스팅되어 모든 행에 적용됩니다.

세 번째 예시처럼 `np.delete`에 여러 색인을 목록으로 입력하면, 해당 위치의 요소를 한 번에 제거합니다.

`np.insert`와 `np.delete`도 `np.append`와 마찬가지로 원본을 변경하지 않고 새로운 배열을 반환합니다.
