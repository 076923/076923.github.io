---
layout: post
title: "Python Numpy 강좌 : 제 8강 - 매트릭스"
tagline: "Python Numpy matrix"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy matrix, Numpy matmul, Numpy linalg, Matrix multiplication
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-8/
comments: true
toc: true
---

## 행렬 곱

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 3], [2, 4]])

print(a * b)
print(a @ b)
print(np.matmul(a, b))

{% endhighlight %}

**결과**
:    
[[ 1  6]<br>
&nbsp;[ 6 16]]<br>
[[ 5 11]<br>
&nbsp;[11 25]]<br>
[[ 5 11]<br>
&nbsp;[11 25]]<br>
<br>

`array * array`의 경우, 결과는 **각각의 원소에 대한 곱을 반환합니다.**

행렬의 곱을 계산해야 하는 경우, `@` 연산자 또는 `np.matmul(배열1, 배열2)` 함수를 사용합니다.

`a @ b`의 첫 번째 원소는 `a`의 첫 번째 행 `[1, 2]`와 `b`의 첫 번째 열 `[1, 2]`의 곱의 합인 `1×1 + 2×2 = 5`로 계산됩니다.

- Tip : `np.dot(배열1, 배열2)`도 2차원 배열에서는 행렬 곱과 같은 결과를 반환합니다. 다만 3차원 이상의 배열에서는 `np.matmul`과 계산 방식이 다르므로, 행렬 곱에는 `@` 연산자를 사용하는 것을 권장합니다.

<br>
<br>

## 매트릭스 클래스

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1-1j, 2-1j], [3, 4]])
c = np.array([[1, 3], [2, 4]])
d = np.array([[1, 3], [2, 4]])

ma = np.asmatrix(a).T
mb = np.asmatrix(b).H
mc = np.asmatrix(c).I
md = np.asmatrix(d).A

print(ma)
print(mb)
print(mc)
print(md)
print(np.asmatrix(a) * np.asmatrix(c))

{% endhighlight %}

**결과**
:    
[[1 3]<br>
&nbsp;[2 4]]<br>
[[1.+1.j 3.-0.j]<br>
&nbsp;[2.+1.j 4.-0.j]]<br>
[[-2.   1.5]<br>
&nbsp;[ 1.  -0.5]]<br>
[[1 3]<br>
&nbsp;[2 4]]<br>
[[ 5 11]<br>
&nbsp;[11 25]]<br>
<br>

`np.asmatrix(배열)`을 사용하여 `array`를 `matrix` 형식으로 변환할 수 있습니다. `matrix` 형식은 항상 **2차원**이며, `*` 연산자가 **행렬 곱**을 계산합니다.

`matrix.T`의 경우, 매트릭스의 `전치`값을 반환합니다.

`matrix.H`의 경우, 매트릭스의 `켤레 전치(공액 전치)`값을 반환합니다.

`matrix.I`의 경우, 매트릭스의 `역행렬`값을 반환합니다.

`matrix.A`의 경우, 매트릭스 형식을 다시 `array` 형식으로 변환하여 반환합니다.

- Tip : `NumPy 2.0`부터 `np.mat` 함수는 제거되었으며, `np.asmatrix`를 사용해야 합니다.

- Tip : Numpy 공식 문서는 `matrix` 클래스의 사용을 **권장하지 않습니다.** `array`와 `@` 연산자로 모든 행렬 연산을 수행할 수 있으며, `matrix` 클래스는 향후 제거될 수 있습니다.

<br>
<br>

## 배열로 행렬 연산하기

{% highlight Python %}

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1-1j, 2-1j], [3, 4]])
c = np.array([[1, 3], [2, 4]])

print(a.T)
print(b.conj().T)
print(np.linalg.inv(c))

{% endhighlight %}

**결과**
:    
[[1 3]<br>
&nbsp;[2 4]]<br>
[[1.+1.j 3.-0.j]<br>
&nbsp;[2.+1.j 4.-0.j]]<br>
[[-2.   1.5]<br>
&nbsp;[ 1.  -0.5]]<br>
<br>

`matrix` 클래스의 속성은 `array`에서 다음과 같이 대체할 수 있습니다.

| matrix |         array          |      의미      |
|:------:|:----------------------:|:--------------:|
| m.T    | a.T                    | 전치           |
| m.H    | a.conj().T             | 켤레 전치      |
| m.I    | np.linalg.inv(a)       | 역행렬         |
| m * n  | a @ b                  | 행렬 곱        |

<br>
<br>

## 선형대수 함수

{% highlight Python %}

import numpy as np

a = np.array([[3, 1], [1, 2]])
y = np.array([9, 8])

x = np.linalg.solve(a, y)

print(np.linalg.det(a))
print(np.linalg.inv(a))
print(x)
print(a @ x)

{% endhighlight %}

**결과**
:    
5.000000000000001<br>
[[ 0.4 -0.2]<br>
&nbsp;[-0.2  0.6]]<br>
[2. 3.]<br>
[9. 8.]<br>
<br>

`np.linalg` 모듈은 행렬식, 역행렬, 연립방정식 등 **선형대수 연산**을 제공합니다.

`np.linalg.det(배열)`은 `행렬식`을, `np.linalg.inv(배열)`은 `역행렬`을 반환합니다. 행렬식은 부동소수점 연산으로 계산되므로 `5`가 아닌 `5.000000000000001`처럼 **미세한 오차**가 발생할 수 있습니다.

`np.linalg.solve(A, y)`는 연립방정식 `Ax = y`의 해 `x`를 반환합니다. 위 예제는 `3x₀ + x₁ = 9`, `x₀ + 2x₁ = 8`의 해인 `x₀ = 2`, `x₁ = 3`을 계산합니다.

- Tip : 연립방정식을 풀 때는 `np.linalg.inv(A) @ y`보다 `np.linalg.solve(A, y)`가 더 빠르고 수치적으로 안정적입니다. 역행렬이 존재하지 않는 행렬을 입력하면 `LinAlgError`가 발생합니다.

|             함수            |                 설명                |
|:---------------------------:|:-----------------------------------:|
|    np.linalg.det( array )   |               행렬식                |
|    np.linalg.inv( array )   |               역행렬                |
| np.linalg.solve( array, y ) |         연립방정식의 해          |
|   np.linalg.eig( array )    |       고윳값과 고유벡터       |
|   np.linalg.norm( array )   |         노름(벡터의 크기)         |
| np.linalg.matrix_rank( array ) |             행렬의 계수             |
|    np.trace( array )        |        대각합(대각 원소의 합)        |
