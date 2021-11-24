---
layout: post
title: "Python Numpy 강좌 : 제 8강 - 매트릭스"
tagline: "Python Numpy matrix"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy matrix
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-8/
comments: true
toc: true
---

## 매트릭스

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 3], [2, 4]])

print(a*b)

ma = np.mat(a)
mb = np.mat(b)

print(ma*mb)

{% endhighlight %}

**결과**
:    
[[ 1  6]<br>
 [ 6 16]]<br>
[[ 5 11]<br>
 [11 25]]<br>
<br>

`array * array`의 경우, 결과는 **각각의 원소에 대한 곱을 반환합니다.**

행렬의 곱을 반환해야하는 경우, `mat` 형식으로 변환 후, **곱 연산을 실행합니다.**

<br>
<br>

## 매트릭스 클래스

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1-1j, 2-1j], [3, 4]])
c = np.array([[1, 3], [2, 4]])
d = np.array([[1, 3], [2, 4]])

ma = np.mat(a).T
mb = np.mat(b).H
mc = np.mat(c).I
md = np.mat(d).A

print(ma)
print(mb)
print(mc)
print(md)

{% endhighlight %}

**결과**
:    
[[1 3]<br>
 [2 4]]<br>
[[1.+1.j 3.-0.j]<br>
 [2.+1.j 4.-0.j]]<br>
[[-2.   1.5]<br>
 [ 1.  -0.5]]<br>
[[1 3]<br>
 [2 4]]<br>
<br>

`mat(배열).T`의 경우, 매트릭스의 `전치`값을 반환합니다.

`mat(배열).H`의 경우, 매트릭스의 `공액복소수의 전치`값을 반환합니다.

`mat(배열).I`의 경우, 매트릭스의 `곱의 역함수`값을 반환합니다.

`mat(배열).A`의 경우, 매트릭스 형식을 다시 `array` 형식으로 변환하여 반환합니다.

- Tip : `*.T`의 경우, `array` 배열의 **전치** 값으로도 사용이 가능합니다.

- Tip : `*.A`의 경우, **곱 연산**을 실행시 `원소 곱`의 값으로 반환합니다.
