---
bg: "numpy.png"
layout: post
comments: true
title: "Python numpy 강좌 : 제 7강 - 연산"
crawlertitle: "Python numpy 강좌 : 제 7강 - 연산"
summary: "Python numpy calculate"
date: 2018-08-09
categories: posts
tags: ['Python-numpy']
author: 윤대희
star: true
---

### 연산 ###
----------

{% highlight Python %}

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.ones(3, dtype=int)

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(np.dot(a, b))
print(np.cross(a, b))

{% endhighlight %}

**결과**
:    
[[ 2  3  4]<br>
 [ 5  6  7]<br>
 [ 8  9 10]]<br>
[[0 1 2]<br>
 [3 4 5]<br>
 [6 7 8]]<br>
[[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>
[[1. 2. 3.]<br>
 [4. 5. 6.]<br>
 [7. 8. 9.]]<br>
[ 6 15 24]<br>
[[-1  2 -1]<br>
 [-1  2 -1]<br>
 [-1  2 -1]]<br>

사칙연산과 관련된 연산은 `array`와 `array` 사이에 **연산 기호를 포함하여 계산할 수 있습니다.**

내적의 경우, `np.dot(a, b)`를 이용하여 계산할 수 있습니다.

외적의 경우, `np.cross(a, b)`를 이용하여 계산할 수 있습니다.

<br>

이외에도 `sin()`, `cos()`, `tan()`, `ceil()`, `floor()`, `exp()`, `mod()`, `sqrt()`, `maximum()`, `minimum()`등 다양한 수학 함수도 지원합니다.
