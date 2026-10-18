---
layout: post
title: "Python Numpy 강좌 : 제 9강 - 차원 확장 및 축소"
tagline: "Python Numpy Dimension Change"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy newaxis, Numpy Expansion, Numpy Reduction, Numpy Dimension, Numpy expand_dims, Numpy squeeze, Numpy reshape, Numpy Ellipsis
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

`np.newaxis`를 입력한 위치에 **길이가 1인 새로운 축**이 추가됩니다.

`arr[np.newaxis]`는 맨 앞에 축을 추가하므로 `(4,)` 형태의 배열이 `(1, 4)` 형태의 **행 벡터**가 됩니다.

`arr[:, np.newaxis]`는 기존 축 뒤에 축을 추가하므로 `(4,)` 형태의 배열이 `(4, 1)` 형태의 **열 벡터**가 됩니다.

- Tip : `np.newaxis`는 `None`의 별칭이므로, `arr[None]`이나 `arr[:, None]`으로 작성해도 동일한 결과를 반환합니다.

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

`배열[:, :, :, ... , np.newaxis]`를 이용하여 원하는 위치에 차원을 확장시킬 수 있습니다.

`(2, 2)` 형태의 배열에서 두 번째 위치에 축을 추가했으므로, `new_arr`는 `(2, 1, 2)` 형태가 됩니다.

차원이 증가함에 따라 `index`의 표시법이 같이 증가합니다.

주로, **브로드캐스팅을 이용한 연산에 사용됩니다.**

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
&nbsp;[4 5 6]]<br>
<br>

차원을 어떻게 나누느냐에 따라, 결과가 상이하게 달라집니다.

`(1, 3)` 형태와 `(3,)` 형태를 연산하면 `(1, 3)` 형태의 결과가 반환되며, `(3, 1)` 형태와 `(3,)` 형태를 연산하면 `(3, 3)` 형태의 결과가 반환됩니다.

이는 [브로드캐스팅][12강] 규칙에 따라 길이가 1인 축이 상대 배열의 길이만큼 확장되기 때문입니다.

주로, `배열[:, np.newaxis]` 형태로 **모든 원소 쌍의 연산 결과(외적 형태)**를 계산할 때 사용합니다.

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

만약, 차원 축소 함수에 축을 입력하지 않으면, **길이가 1인 모든 축을 제거합니다.**

- Tip : `np.squeeze`는 길이가 1인 축만 제거할 수 있습니다. 결과가 항상 `1차원 배열`이 되는 것은 아니며, 길이가 1이 아닌 축은 그대로 유지됩니다.

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

`reshape` 함수는 변경 전과 변경 후의 **전체 원소의 개수가 같아야** 사용할 수 있습니다. 예제의 배열은 원소가 8개이므로, `(1, 1, 2, 4)`나 `(4, 2)`처럼 곱이 8인 형태로만 변경할 수 있습니다.

`-1`을 입력한 축은 전체 원소의 개수에 맞게 **길이가 자동으로 계산**되며, 한 번만 사용할 수 있습니다.

<br>
<br>

## 축 위치 지정

{% highlight Python %}

import numpy as np

arr = np.zeros((2, 3))

print(np.expand_dims(arr, axis=0).shape)
print(np.expand_dims(arr, axis=1).shape)
print(np.expand_dims(arr, axis=-1).shape)
print(np.expand_dims(arr, axis=(0, 3)).shape)
print(arr[np.newaxis, :, :, np.newaxis].shape)
print(arr[..., np.newaxis].shape)

{% endhighlight %}

**결과**
:    
(1, 2, 3)<br>
(2, 1, 3)<br>
(2, 3, 1)<br>
(1, 2, 3, 1)<br>
(1, 2, 3, 1)<br>
(2, 3, 1)<br>
<br>

`np.expand_dims`의 `axis`는 **결과 배열에서 새로운 축이 놓일 위치**를 의미합니다.

`axis=-1`을 입력할 경우, 맨 뒤에 새로운 축을 추가합니다.

`axis`에 `튜플`을 입력할 경우, **여러 개의 축을 한 번에 추가**할 수 있습니다.

`np.newaxis`를 여러 번 사용하여도 동일한 결과를 얻을 수 있으며, `...(Ellipsis)`는 **나머지 모든 축**을 의미하므로 차원의 수를 몰라도 맨 뒤에 축을 추가할 수 있습니다.

<br>

{% highlight Python %}

import numpy as np

arr = np.zeros((1, 3, 1, 2))

print(np.squeeze(arr).shape)
print(np.squeeze(arr, axis=0).shape)
print(np.squeeze(arr, axis=(0, 2)).shape)

np.squeeze(arr, axis=1)

{% endhighlight %}

**결과**
:    
(3, 2)<br>
(3, 1, 2)<br>
(3, 2)<br>
ValueError: cannot select an axis to squeeze out which has size not equal to one<br>
<br>

`np.squeeze`에 축을 입력하지 않으면 길이가 1인 `0번 축`과 `2번 축`이 모두 제거되어 `(3, 2)` 형태가 됩니다.

축을 지정할 경우, **지정한 축만 제거**되며 `튜플`로 여러 개의 축을 지정할 수 있습니다.

길이가 1이 아닌 축을 지정할 경우, `ValueError`가 발생합니다.

- Tip : 길이가 1인 축이 의도치 않게 모두 제거되는 것을 방지하려면, `np.squeeze`를 사용할 때 **축을 명시적으로 지정**하는 것이 안전합니다.

[12강]: https://076923.github.io/posts/Python-numpy-12/
