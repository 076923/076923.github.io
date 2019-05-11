---
bg: "numpy.png"
layout: post
comments: true
title: "Python numpy 강좌 : 제 11강 - 병합"
crawlertitle: "Python numpy 강좌 : 제 11강 - 병합"
summary: "Python numpy stack"
date: 2018-08-12
categories: posts
tags: ['Python-numpy']
author: 윤대희
star: true
---

### 병합 ###
----------

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.hstack([a, b])
d = np.vstack([a, b])

print(a)
print(b)
print("--------")
print(c)
print(d)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
--------<br>
[[1 2 5 6]<br>
 [3 4 7 8]]<br>
[[1 2]<br>
 [3 4]<br>
 [5 6]<br>
 [7 8]]<br>
<br>

`numpy.hstack([배열1, 배열2])`를 이용하여 `배열1` **우측**에 `배열2`를 **이어 붙일 수 있습니다.**

`numpy.vstack([배열1, 배열2])`를 이용하여 `배열1` **하단**에 `배열2`를 **이어 붙일 수 있습니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.stack([a, b], axis=0)
d = np.stack([a, b], axis=1)

print(a)
print(b)
print("--------")
print(c)
print("--------")
print(d)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
--------<br>
[[[1 2]<br>
  [3 4]]<br>
<br>
 [[5 6]<br>
  [7 8]]]<br>
--------<br>
[[[1 2]<br>
  [5 6]]<br>
<br>
 [[3 4]<br>
  [7 8]]]<br>
<br>

`numpy.stack([배열1, 배열2, axis=축])`를 이용하여 지정한 `축`으로 `배열1`과 `배열2`를 **이어 붙일 수 있습니다.**

`축`은 **이어 붙일 차원의 범위를 넘어갈 수 없습니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.dstack([a, b])

print(a)
print(b)
print("--------")
print(c)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
--------<br>
[[[1 5]<br>
  [2 6]]<br>
<br>
 [[3 7]<br>
  [4 8]]]<br>
<br>

`numpy.dstack([배열1, 배열2])`를 이용하여 `새로운 축`으로 `배열1`과 `배열2`를 **이어 붙일 수 있습니다.**

`numpy.stack([a, b], axis=2)`과 **동일한 결과를 반환합니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.tile(a, 2)

print(a)
print(b)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[1 2 1 2]<br>
 [3 4 3 4]]<br>
<br>

`numpy.tile(배열, 반복 횟수)`를 이용하여 `배열`을 `반복 횟수`만큼 **이어 붙일 수 있습니다.**

<br>

### 특수 병합 메소드 ###
----------

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.c_[a, b]
d = np.r_[a, b]

print(a)
print(b)
print("--------")
print(c)
print(d)

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[5 6]<br>
 [7 8]]<br>
--------<br>
[[1 2 5 6]<br>
 [3 4 7 8]]<br>
[[1 2]<br>
 [3 4]<br>
 [5 6]<br>
 [7 8]]<br>
<br>

`numpy.c_[배열1, 배열2]`를 이용하여 `배열1` **우측**에 `배열2`를 **이어 붙일 수 있습니다.**

`numpy.r_[배열1, 배열2]`를 이용하여 `배열1` **하단**에 `배열2`를 **이어 붙일 수 있습니다.**

`소괄호 ()`를 사용하지 않고 `대괄호 []`를 사용하여 메소드를 생성합니다.
