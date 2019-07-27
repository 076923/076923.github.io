---
layout: post
title: "Python Numpy 강좌 : 제 9강 - 차원 확장"
tagline: "Python Numpy newaxis"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy newaxis
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-9/
comments: true
---

## 차원 확장 ##
----------

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3, 4])

print(a)
print(a[np.newaxis])
print(a[:, np.newaxis])

{% endhighlight %}

**결과**
:    
[1 2 3 4]<br>
[[1 2 3 4]]<br>
[[1]<br>
 [2]<br>
 [3]<br>
 [4]]<br>
<br>

`index` 중 `np.newaxis`를 이용하여 차원을 확장할 수 있습니다.

`행` 부분에 `np.newaxis`를 입력시, **차원을 한 단계 추가합니다.**

`열` 부분에 `np.newaxis`를 입력시, **차원을 분해한 후 한 단계 추가합니다.**

<br>

{% highlight Python %}

import numpy as np

b = np.array([[1, 2],
              [3, 4]], dtype=int)

c = b[:, np.newaxis]

print(b)

print(c)
print(c[1][0])
print(c[1][0][1])

{% endhighlight %}

**결과**
:    
[[1 2]<br>
 [3 4]]<br>
[[[1 2]]<br>
<br>
 [[3 4]]]<br>
[3 4]<br>
4<br>

`배열[:, :, :, ... , np.newaxis]`를 이용하여 차원을 확장시킬 수 있습니다.

차원이 증가함에 따라 `index`의 표시법이 같이 증가합니다.

주로, **슬라이싱를 통한 연산에 사용됩니다.**

<br>

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

print(a[np.newaxis] * b)
print(a[:, np.newaxis] * b)
print("\n")
print(a[np.newaxis] + b)
print(a[:, np.newaxis] + b)

{% endhighlight %}

**결과**
:    
[[1 4 9]]<br>
[[1 2 3]<br>
 [2 4 6]<br>
 [3 6 9]]<br>
<br>
<br>
[[2 4 6]]<br>
[[2 3 4]<br>
 [3 4 5]<br>
 [4 5 6]]<br>
<br>

차원을 어떻게 나누느냐에 따라, 결과가 상이하게 달라집니다.

주로, `배열[:, np.newaxis]` 형태로 계산을 진행합니다.

