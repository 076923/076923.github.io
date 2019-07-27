---
layout: post
title: "Python Numpy 강좌 : 제 12강 - 브로드캐스팅"
tagline: "Python Numpy Broadcasting"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy Broadcasting
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-12/
comments: true
---

## 브로드캐스팅(Broadcasting) ##
----------

`브로드캐스팅`이란 Numpy 배열에서 차원의 크기가 서로 다른 배열에서도 **산술 연산을 가능하게 하는 원리입니다.**

두 배열 간 차원의 크기가 **(4, 2)**, **(2, )** 일 때 `산술 연산`을 실행한다면 **(2, )**의 배열이 **(4, 2)** 행렬의 각 행에 대해 요소별 연산을 실행할 수 있습니다.

두 배열 간의 차원의 크기가 달라도 차원의 크기가 더 큰 배열에 대해 작은 배열을 **여러 번 반복하지 않아도 되는 것을 의미합니다.**

* Tip : (2, )은 1차원 형태의 배열을 의미합니다. 즉, [a b]의 값을 지닙니다.

<br>
<br>

## 브로드캐스팅 허용 규칙 ##

1. 두 배열의 차원(ndim)이 같지 않다면 차원이 더 낮은 배열이 차원이 더 높은 배열과 **같은 차원의 배열로 인식**됩니다.
2.	반환된 배열은 연산을 수행한 배열 중 **차원의 수(ndim)**가 가장 큰 배열이 됩니다.
3.	연산에 사용된 배열과 반환된 배열의 **차원의 크기(shape)**가 같거나 1일 경우 브로드캐스팅이 가능합니다.
4.	브로드캐스팅이 적용된 **배열의 차원 크기(shape)**는 연산에 사용된 배열들의 차원의 크기에 대한 `최소 공배수 값`으로 사용합니다.

<br>
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

또한, `형식 캐스팅(type casting)`을 지원해 **두 배열의 자료형(dtype)이 다르더라도 연산이 가능합니다.**

* Tip : `형식 캐스팅` : 자료형을 비교해 표현 범위가 더 넓은 자료형을 선택하는 것을 의미합니다.