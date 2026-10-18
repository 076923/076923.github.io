---
layout: post
title: "Python Numpy 강좌 : 제 10강 - 난수"
tagline: "Python Numpy random"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy random, Numpy seed, Numpy choice, Numpy default_rng, Numpy Generator, Numpy permutation, Numpy shuffle
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-10/
comments: true
toc: true
---

## 무작위 선택

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

`numpy.random.seed(n)`을 이용하여 난수 생성기의 **시드(seed)**를 설정할 수 있습니다. 같은 시드 값을 사용하면 난수와 흡사하지만 **항상 같은 결과를 반환합니다.**

`numpy.random.choice(배열, n, replace=True, p=None)`을 이용하여 `배열`에서 `n`개의 값을 선택하여 반환할 수 있습니다.

`replace`를 `True`로 사용할 경우, 값이 중복되어 선택될 수 있습니다. `False`로 사용할 경우, 값이 중복되지 않습니다.

`p`를 이용하여 **각 데이터가 선택될 확률**을 설정할 수 있습니다. `p` 배열의 길이는 항상 `배열`의 길이와 같아야합니다. 

`p`값들의 총합은 항상 `1`이어야 하며, `replace`를 `False`로 사용할 경우 값이 중복되지 않기 때문에 **확률이 0이 아닌 원소가 n개 이상**이어야 합니다.

- Tip : 이 강좌의 결과는 시드를 `76923`으로 고정했을 때의 값입니다. 시드를 설정하지 않으면 **실행할 때마다 다른 결과**가 반환됩니다.

<br>
<br>

## 난수 발생

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

`numpy.random.rand(n, m, ...)`을 이용하여 **0.0 이상 1.0 미만**의 균일 분포에서 **다차원** 무작위 배열을 생성할 수 있습니다.

`numpy.random.randn(n, m, ...)`을 이용하여 평균이 0, 표준 편차가 1인 **표준 정규 분포**에서 무작위 배열을 생성할 수 있습니다.

`numpy.random.randint(low, high, (n, m), dtype=int)`을 이용하여 `low` ~ `high-1` 사이의 무작위 `(n, m)` 크기 정수 배열을 반환합니다. `high`는 **결과에 포함되지 않습니다.**

- Tip : `rand`와 `randn`은 크기를 **개별 인수**로 전달하지만, `randint`를 비롯한 다른 함수는 크기를 `튜플`로 전달합니다.

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

`numpy.random.random((n, m))`과 `numpy.random.sample((n, m))`를 이용하여 **0.0 이상 1.0 미만** 사이의 무작위 `(n, m)` 크기 배열을 반환합니다.

두 함수는 `numpy.random.random_sample`과 **동일한 함수**입니다. 같은 난수 생성기를 이어서 사용하므로 `b`는 `a` 다음 순서의 난수를 반환합니다.

<br>
<br>

## 확률 분포 난수

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

`numpy.random.uniform(low, high, size)`를 이용하여 `low` 이상 `high` 미만 사이의 **균일한 분포의 무작위 배열**을 반환합니다.

`numpy.random.lognormal(mean, sigma, size)`를 이용하여 `평균`과 `시그마`를 대입한 **로그 정규 분포의 무작위 배열**을 반환합니다. 이때 `mean`과 `sigma`는 결과값이 아닌, **로그를 취한 값이 따르는 정규 분포**의 평균과 표준 편차입니다.

`numpy.random.laplace(loc, scale, size)`를 이용하여 위치 `μ`와 척도 `λ`를 대입한 **라플라스 분포의 무작위 배열**을 반환합니다.

<br>
<br>

## 난수 생성기(Generator)

{% highlight Python %}

import numpy as np

rng = np.random.default_rng(76923)

a = rng.random((2, 2))
b = rng.normal(0, 1, (2, 2))
c = rng.integers(1, 3, (2, 2))
d = rng.choice([1, 2, 3, 4, 5], 3, replace=False)

print(a)
print(b)
print(c)
print(d)

{% endhighlight %}

**결과**
:    
[[0.36793328 0.34801885]<br>
 [0.93239592 0.74132094]]<br>
[[-0.00144017  0.41942466]<br>
 [ 0.5210108  -0.05317788]]<br>
[[1 2]<br>
 [2 2]]<br>
[4 3 1]<br>
<br>

`numpy.random.default_rng(시드)`를 이용하여 **난수 생성기(Generator)** 객체를 생성할 수 있습니다.

NumPy 1.17 버전부터는 `numpy.random.seed`와 같이 **전역 상태를 공유하는 방식** 대신, 난수 생성기 객체를 만들어 사용하는 방식이 권장됩니다.

난수 생성기 객체는 각각 독립적인 상태를 가지므로, 다른 코드에서 난수를 사용하더라도 **결과의 재현성에 영향을 주지 않습니다.**

`rng.random`, `rng.normal`, `rng.integers`, `rng.choice` 등 앞에서 다룬 함수와 대응하는 메서드를 제공합니다.

`rng.integers(low, high, size)`는 `randint`와 동일하게 `high`를 포함하지 않으며, `endpoint=True`를 입력하면 `high`까지 포함합니다.

- Tip : 기존의 `numpy.random.*` 함수는 제거되지 않았으며 그대로 사용할 수 있습니다. 다만 호환성을 위해 **결과가 바뀌지 않도록 고정된 레거시 방식**이며, 같은 시드라도 `default_rng`와는 **다른 난수 알고리즘**을 사용하므로 결과가 다릅니다.

<br>
<br>

## 배열 섞기

{% highlight Python %}

import numpy as np

rng = np.random.default_rng(76923)

arr = np.array([1, 2, 3, 4, 5])
p = rng.permutation(arr)

print(p)
print(arr)

rng.shuffle(arr)

print(arr)

{% endhighlight %}

**결과**
:    
[2 1 3 5 4]<br>
[1 2 3 4 5]<br>
[1 2 3 5 4]<br>
<br>

`rng.permutation(배열)`을 이용하여 배열의 순서를 무작위로 섞은 **새로운 배열**을 반환합니다. 원본 배열은 변경되지 않습니다.

`rng.shuffle(배열)`을 이용하여 **원본 배열의 순서를 직접 섞습니다.** 반환값은 `None`입니다.

다차원 배열의 경우, 기본적으로 **첫 번째 축(행)의 순서만** 섞으며 각 행 내부의 원소 순서는 유지됩니다.

- Tip : 레거시 방식의 `numpy.random.permutation`과 `numpy.random.shuffle`도 동일한 동작을 수행합니다.
