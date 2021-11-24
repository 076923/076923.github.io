---
layout: post
title: "Python Numpy 강좌 : 제 14강 - 이어 붙이기"
tagline: "Python Numpy Append"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Append
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

`이어 붙이기 함수(np.append)`는 **내장 이어 붙이기 함수(append)**와 다르게 차원이 같아야 붙일 수 있습니다.

만약, `Numpy 배열`이 아닌 `List 배열`이라면 내장 이어 붙이기 함수를 통해 큰 문제 없이 이어 붙일 수 있습니다.

그 이유는 `List`는 `Container 형식`의 단순 값을 저장하는 자료형이기 때문입니다.

`Numpy 배열`은 연산을 위한 라이브러리입니다.

`arr` 변수의 차원 형태는 **(2, 2, 2)**를 갖으며, `item` 변수는 **(2, 2)**를 갖습니다.

이 값을 이어 붙일 때, 앞 부분(**2, 2**, 2)에 붙여야하는지, 뒷 부분(2, **2, 2**)에 붙여야하는지 알 수 없습니다.

그러므로, `item` 배열의 차원 크기를 `arr` 배열의 차원 크기와 동일하게 구성한다음 명시적으로 `어떤 축(axis)`에 연결할지 설정합니다.

이어 붙이기 함수는 `결과 = np.append(배열1, 배열2, 축)`을 통해 배열을 이어 붙일 수 있습니다.

위의 예제는 `뒷 부분`에 연결하는 예시입니다.

만약, `앞 부분`에 연결한다면, 차원 형태를 `(2, 2, 1)`로 설정하고 `마지막 축(axis=2)`에 연결해 사용할 수 있습니다.

<br>

{% highlight Python %}

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

반복문을 통해 이어 붙이기가 완료되면, **첫 번째 행과 열에 있는 더미 값을 제거**해야 합니다.

`제거 함수(np.delete)`를 통해 특정 요소값을 삭제합니다.

`결과 = np.delete(배열, [시작, 끝], 축)`을 통해 배열을 제거할 수 있습니다.

제거 함수는 `[시작, 끝]`과 `축` 값에 따라 제거할 대상을 설정합니다.

**시작 축** 부터 **끝 축**까지 제거합니다.

예시에서는 `첫 번째 축(axis=0)`을 대상으로 `[0 ~ 0]`을 삭제하므로 첫 번째 행이 제거됩니다.

즉, 더미 값인 `np.empty` 값이 제거되고 이어 붙인 값만 남게됩니다.
