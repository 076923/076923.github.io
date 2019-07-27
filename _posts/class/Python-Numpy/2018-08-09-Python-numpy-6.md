---
layout: post
title: "Python Numpy 강좌 : 제 6강 - 리쉐이프"
tagline: "Python Numpy reshape"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy reshape
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-6/
comments: true
---

## 리쉐이프 ##
----------

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]])

print(a)
print(np.shape(a))
a.shape = (6,2)
print(a)

{% endhighlight %}

**결과**
:    
[[ 1  2  3  4]<br>
 [ 5  6  7  8]<br>
 [ 9 10 11 12]]<br>
(3, 4)<br>
[[ 1  2]<br>
 [ 3  4]<br>
 [ 5  6]<br>
 [ 7  8]<br>
 [ 9 10]<br>
 [11 12]]<br>
<br>

`배열.shape = (행, 열)`을 통하여 배열의 형태를 변환할 수 있습니다.

`행렬`의 값은 총 길이의 `곱(개수)`과 같아야합니다.

원본 배열의 크기가 `3x4=12`일 경우, 리쉐이프 할때 `6x2=12`로 **총 개수와 동일해야합니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]],
              [[13, 14, 15], [16, 17, 18]]])

print(a)
print(np.shape(a))
a.shape = (2, 3, 3)
print(a)

{% endhighlight %}

**결과**
:    
[[[ 1  2  3]<br>
  [ 4  5  6]]<br>
<br>
 [[ 7  8  9]<br>
  [10 11 12]]<br>
<br>
 [[13 14 15]<br>
  [16 17 18]]]<br>
(3, 2, 3)<br>
[[[ 1  2  3]<br>
  [ 4  5  6]<br>
  [ 7  8  9]]<br>
<br> 
 [[10 11 12]<br>
  [13 14 15]<br>
  [16 17 18]]]<br>
<br>

**3차원 이상의 배열** 또한 형태를 변환할 수 있습니다.

`배열.shape= (페이지, 행, 열)`의 순서이며, 역시 **총 길이의 개수와 동일해야합니다.**