---
layout: post
title: "Python numpy 강좌 : 제 5강 - 슬라이싱"
tagline: "Python Numpy slicing"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy slicing
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-5/
comments: true
---

## 슬라이싱 ##
----------

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(a)
print("\n")
print(a[3:])
print("\n")
print(a[1:-1]) 
print("\n")
print(a[0:3:2])

{% endhighlight %}

**결과**
:    
[1 2 3 4 5]<br>
<br>
[4 5]<br>
<br>
[2 3 4]<br>
<br>
[1 3]
<br>

`배열[a:b:c]`를 이용하여  **배열의 일부를 잘라 표시할 수 있습니다.** 

`a`는 **시작값**, `b`는 **도착값**, `c`는 **간격**을 의미합니다.

`index`는 `0 ~ len-1`까지 존재하며, 아무것도 입력하지 않고 `:`로 사용할 경우, 모든 `행` 또는 `열`을 의미합니다.

`:n`으로 사용할 경우 `0 ~ n`까지의 길이를 의미하며 `n:`으로 사용할 경우, `n ~ len-1`까지의 길이를 의미합니다.

`-1`을 입력할 경우, `마지막 index-1 (len-2)`를 의미합니다.

<br>

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

print(a)
print("\n")
print(a[:, 1:])
print("\n")
print(a[0:1,0:2]) 

{% endhighlight %}

**결과**
:    
[[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>
<br>
[[2 3]<br>
 [5 6]<br>
 [8 9]]<br>
<br>
[[1 2]]<br>
<br>

`배열[a:b, c:d]`를 이용하여 **배열의 일부를 잘라 표시할 수 있습니다.**

동일하게 `배열[a:b:e, c:d:f]`를 이용하여 `e`와 `f`를 **간격**으로 사용할 수 있습니다.

`a` ~ `b`는 표시할 `행`의 위치를 의미하며, `c` ~ `d`는 표시할 `열`의 위치를 의미합니다.

<br>

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]])

print(a)
print("\n")
print(a[::2, ::2])

{% endhighlight %}
**결과**
:    
[[ 1  2  3  4  5]<br>
 [ 6  7  8  9 10]<br>
 [11 12 13 14 15]]<br>
<br>
[[ 1  3  5]<br>
 [11 13 15]]<br>
<br>

배열의 슬라이싱에서 `간격`만 입력하여 배열을 출력할 수 있습니다.

`[::2, ::2]`일 경우, `행`을 `2`칸씩 띄우며, `열`도 `2`칸씩 띄워 출력합니다.