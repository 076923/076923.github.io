---
layout: post
title: "Python numpy 강좌 : 제 10강 - 난수"
tagline: "Python Numpy random"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy random
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-10/
comments: true
---

## 무작위 선택 ##
----------

{% highlight Python %}

import numpy as np

np.random.seed(76923)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.random.choice(a, 5, replace=False, p=[0, 0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.8])

print (b)

{% endhighlight %}

**결과**
:    
[9 5 6 7 8]<br>
<br>

`numpy.random.seed(n)`을 이용하여 **임의의 시드**를 생성할 수 있습니다. 시드 값에 따라 난수와 흡사하지만 **항상 같은 결과를 반환합니다.**

`numpy.random.choice(배열, n, replace=True, p=None)`을 이용하여 `배열`에서 `n`개의 값을 선택하여 반환할 수 있습니다.

`replace`를 `True`로 사용할 경우, 값이 중복되어 선택되 반환될 수 있습니다. `False`로 사용할 경우, 값이 중복되지 않습니다.

`p`를 이용하여 **각 데이터가 선택될 확률**을 설정할 수 있습니다. `p` 배열의 길이는 항상 `배열`의 길이와 같아야합니다. 

`p`값들의 총합은 항상 `1`이여야 하며, `replace`를 `False`로 사용할 경우, 값이 중복되지 않기 때문에 **n개 이상 0의 값과 달라야합니다.**

<br>
<br>

## 난수 발생 ##
----------

{% highlight Python %}

import numpy as np

np.random.seed(76923)

a = np.random.rand(2, 2)
b = np.random.randn(2, 2)
c = np.random.randint(1, 3, (2, 2), dtype=int)

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[[0.76367992 0.87641303]<br>
 [0.53095238 0.38451373]]<br>
[[-0.15929049 -0.07981017]<br>
 [ 1.73777738 -0.47496771]]<br>
[[1 1]<br>
 [2 1]]<br>
<br>

`numpy.random.rand(n, m, ...)`을 이용하여 **다차원** 무작위 배열을 생성할 수 있습니다.

`numpy.random.randn(n, m)`을 이용하여 **표준 정규 분포**에서 무작위 배열을 생성할 수 있습니다.

`numpy.random.randint(low, high, (n, m), dtype=None)`을 이용하여 `low` ~ `high-1` 사이의 무작위  `(n, m)` 크기정수 배열을 반환합니다.

<br>

{% highlight Python %}

import numpy as np

np.random.seed(76923)

a = np.random.random((2, 3))
b = np.random.sample((2, 3))

print(a)
print(b)

{% endhighlight %}

**결과**
:    
[[0.76367992 0.87641303 0.53095238]<br>
 [0.38451373 0.2777934  0.05650517]]<br>
[[0.44143693 0.7142663  0.54434277]<br>
 [0.74534435 0.89561778 0.36096285]]<br>
<br>

`numpy.random.random((n, m))`과 `numpy.random.sample((n, m))`를 이용하여 **0.0 ~ 1.0** 사이의 무작위 `(n, m)` 크기 배열을 반환합니다.

<br>
<br>

## 난수 발생 ##
----------

{% highlight Python %}

import numpy as np

np.random.seed(76923)

a = np.random.uniform(1, 2, (2, 2))
b = np.random.lognormal(3, 1, (2, 2))
c = np.random.laplace(0, 1, (2, 2))

print(a)
print(b)
print(c)

{% endhighlight %}

**결과**
:    
[[1.76367992 1.87641303]<br>
 [1.53095238 1.38451373]]<br>
[[ 17.12791367  18.54480749]<br>
 [114.18014024  12.4912986 ]]<br>
[[ 0.09286726  0.67469586]<br>
 [ 1.56654873 -0.32583306]]<br>
<br>

`numpy.random.uniform(low, high, size)`를 이용하여  `low`~`high` 사이의 **균일한 분포의 무작위 배열**을 반환합니다.

`numpy.random.lognormal(mean, sigma, size)`를 이용하여 `평균`과 `시그마`를 대입한 **로그 정규 분포의 무작위 배열**을 반환합니다.

`numpy.random.laplace(loc, scale, size)` : `μ`와 `λ`를 대입한 **라플라스 분포의 무작위 배열**을 반환합니다.
