---
layout: post
title: "Python Numpy 강좌 : 제 9강 - 차원 확장 및 축소"
tagline: "Python Numpy Dimension Change"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy newaxis, Numpy Expansion, Numpy Reduction, Numpy Dimension
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-9/
comments: true
toc: true
---

## 축 추가

{% highlight Python %}

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr)
print(arr[np.newaxis])
print(arr[:, np.newaxis])

{% endhighlight %}

**결과**
:    
[1 2 3 4]<br>
[[1 2 3 4]]<br>
[[1]<br>
&nbsp;[2]<br>
&nbsp;[3]<br>
&nbsp;[4]]<br>
<br>

`index` 중 `np.newaxis`를 이용하여 차원을 확장할 수 있습니다.

`행` 부분에 `np.newaxis`를 입력시, **차원을 한 단계 추가합니다.**

`열` 부분에 `np.newaxis`를 입력시, **차원을 분해한 후 한 단계 추가합니다.**

<br>

{% highlight Python %}

import numpy as np

arr = np.array([[1, 2],
              [3, 4]], dtype=int)

new_arr = arr[:, np.newaxis]

print(arr)

print(new_arr)
print(new_arr[1][0])
print(new_arr[1][0][1])

{% endhighlight %}

**결과**
:    
[[1 2]<br>
&nbsp;[3 4]]<br>
[[[1 2]]<br>
<br>
&nbsp;[[3 4]]]<br>
[3 4]<br>
4<br>
<br>

`배열[:, :, :, ... , np.newaxis]`를 이용하여 차원을 확장시킬 수 있습니다.

차원이 증가함에 따라 `index`의 표시법이 같이 증가합니다.

주로, **슬라이싱를 통한 연산에 사용됩니다.**

<br>

{% highlight Python %}

import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

print(arr1[np.newaxis] * arr2)
print(arr1[:, np.newaxis] * arr2)
print("\n")
print(arr1[np.newaxis] + arr2)
print(arr1[:, np.newaxis] + arr2)

{% endhighlight %}

**결과**
:    
[[1 4 9]]<br>
[[1 2 3]<br>
&nbsp;[2 4 6]<br>
&nbsp;[3 6 9]]<br>
<br>
<br>
[[2 4 6]]<br>
[[2 3 4]<br>
&nbsp;[3 4 5]<br>
&nbsp;1[4 5 6]]<br>
<br>

차원을 어떻게 나누느냐에 따라, 결과가 상이하게 달라집니다.

주로, `배열[:, np.newaxis]` 형태로 계산을 진행합니다.

<br>
<br>

## 차원 확장 및 축소

{% highlight Python %}

import numpy as np

arr = np.array([[1], [2], [3]])

expansion = np.expand_dims(arr, axis=0)
reduction = np.squeeze(arr, axis=1)

print(arr)
print(expansion)
print(reduction)

{% endhighlight %}

**결과**
:    
[[1]<br>
&nbsp;[2]<br>
&nbsp;[3]]<br>
[[[1]<br>
&nbsp;&nbsp;[2]<br>
&nbsp;&nbsp;[3]]]<br>
[1 2 3]<br>
<br>

`Numpy` 함수를 통해서도 차원을 확장하거나 축소할 수 있습니다.

차원을 확장하는 경우, 지정된 축을 대상으로 차원을 확장합니다.

`np.expand_dims(배열, 축)`을 통해 지정된 **축의 차원을 확장**할 수 있습니다.

차원을 축소하는 경우, 지정된 축을 대상으로 차원을 축소합니다.

`np.squeeze(배열, 축)`을 통해 지정된 **축의 차원을 축소**할 수 있습니다.

만약, 차원 축소 함수에 축을 입력하지 않으면, `1차원 배열`로 축소합니다.

<br>

{% highlight Python %}

import numpy as np

arr = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

expand_dims = np.reshape(arr, (1, 1, 2, 4))
reduction = np.reshape(arr, (4, -1))

print(arr)
print(expand_dims)
print(reduction)

{% endhighlight %}

**결과**
:    
[[[1 2]<br>
&nbsp;&nbsp;[3 4]]<br>
<br>
&nbsp;[[5 6]<br>
&nbsp;&nbsp;[7 8]]]<br>
[[[[1 2 3 4]<br>
&nbsp;&nbsp;&nbsp;[5 6 7 8]]]]<br>
[[1 2]<br>
&nbsp;[3 4]<br>
&nbsp;[5 6]<br>
&nbsp;[7 8]]<br>
<br>

만약, 임의의 차원으로 변경하는 경우, `reshape`를 통해 명시적으로 차원의 형태를 변경할 수 있습니다.

동일 차원을 사용하더라도, `reshape` 함수의 요소값으로 형태를 변경할 수 있습니다.

reshape 함수는 `브로드캐스팅(Broadcasting)` 조건을 만족해야 사용할 수 있습니다.
