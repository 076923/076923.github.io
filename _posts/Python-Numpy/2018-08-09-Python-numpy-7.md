---
layout: post
title: "Python Numpy 강좌 : 제 7강 - 연산"
tagline: "Python Numpy calculate"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy calculate, Numpy ufunc, Numpy axis, Numpy floor divide
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-7/
comments: true
toc: true
---

## 기본 연산

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])
b = np.ones(3, dtype = int)

print(a + b)
print(a - b)
print(a * b)
print(a / b)

{% endhighlight %}

**결과**
:    
[[ 2  3  4]<br>
&nbsp;[ 5  6  7]<br>
&nbsp;[ 8  9 10]]<br>
[[0 1 2]<br>
&nbsp;[3 4 5]<br>
&nbsp;[6 7 8]]<br>
[[1 2 3]<br>
&nbsp;[4 5 6]<br>
&nbsp;[7 8 9]]<br>
[[1. 2. 3.]<br>
&nbsp;[4. 5. 6.]<br>
&nbsp;[7. 8. 9.]]<br>
<br>

사칙연산과 관련된 연산은 `array`와 `array` 사이에 **연산 기호(+, -, *, /)를 포함하여 계산할 수 있습니다.**

연산은 **같은 위치의 원소끼리** 수행됩니다. `*` 연산자도 행렬 곱이 아닌 **원소별 곱**을 계산합니다.

`나누기(/)`는 정수 배열끼리 연산하더라도 결과가 항상 **실수(float)** 배열로 반환됩니다.

사칙연산의 혼합계산은 일반 수식과 동일하게 `곱하기(*)`와 `나누기(/)`를 우선적으로 연산합니다.

배열의 차원구조가 동일하지 않더라도, `브로드캐스팅(Broadcasting)` 조건에 포함된다면 연산이 가능합니다. 위 예제에서 `(3, 3)` 형태의 `a`와 `(3,)` 형태의 `b`는 `b`가 각 행에 반복 적용되어 계산됩니다. 브로드캐스팅은 [12강 - 브로드캐스팅][12강]에서 자세히 다룹니다.

<br>
<br>

## 나눗셈 연산

{% highlight Python %}

import numpy as np

a = np.array([7, -7, 8])
b = np.array([2, 2, 3])

print(a / b)
print(a // b)
print(a % b)

{% endhighlight %}

**결과**
:    
[ 3.5        -3.5         2.66666667]<br>
[ 3 -4  2]<br>
[1 1 2]<br>
<br>

`/`는 실수 나눗셈, `//`는 **몫(내림 나눗셈)**, `%`는 **나머지**를 계산합니다.

`//`는 소수점 아래를 버리는 것이 아니라 **음의 무한대 방향으로 내림**하므로, `-7 // 2`는 `-3`이 아닌 `-4`가 됩니다.

`%`의 결과는 **나누는 수와 같은 부호**를 가지므로, `-7 % 2`는 `1`이 됩니다. 항상 `a == (a // b) * b + (a % b)`가 성립합니다.

<br>
<br>

## 배열 연산

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])
b = np.ones(3, dtype = int)

print(np.cross(a, b))
print(np.dot(a, b))

{% endhighlight %}

**결과**
:    
[[-1  2 -1]<br>
&nbsp;[-1  2 -1]<br>
&nbsp;[-1  2 -1]]<br>
[ 6 15 24]<br>
<br>

`Numpy 라이브러리`는 배열 연산에 특화된 라이브러리입니다.

그러므로, 배열 연산에 활용되는 `점곱(dot product)`이나 `벡터곱(cross product)` 등을 계산할 수 있습니다.

`np.cross(a, b)`는 2차원 배열이 입력될 경우, **각 행을 3차원 벡터로 간주**하여 `b`와의 벡터곱을 계산합니다. `np.dot(a, b)`는 2차원 배열과 1차원 배열이 입력될 경우, **행렬과 벡터의 곱**을 계산합니다.

주의사항으로는 `벡터곱(cross product)`과 `외적(outer product)`은 Numpy 함수에서는 다른 연산입니다.

연산시 의도하는 연산이 `벡터곱(cross product)`인지 `외적(outer product)`인지 명확하게 인지 후 활용해야 합니다.

<br>
<br>

## 심화 연산

{% highlight Python %}

import numpy as np

arr = np.array([-90, 0, np.radians(90)])

sin = np.sin(arr)
tanh = np.tanh(arr)
isclose = np.isclose(arr, -90)
gcd = np.gcd(arr.astype(int) , 3)

print(sin)
print(tanh)
print(isclose)
print(gcd)

{% endhighlight %}

**결과**
:   
[-0.89399666  0.          1.        ]<br>
[-1.          0.          0.91715234]<br>
[ True False False]<br>
[3 3 1]<br>
<br>

`Numpy 라이브러리`는 `Math 라이브러리`에서 사용할 수 있는 수학 함수들을 사용할 수 있습니다.

삼각 함수는 입력값을 **라디안(radian)** 단위로 계산합니다. 위 예제의 `-90`은 `-90도`가 아닌 `-90 라디안`으로 계산되며, 각도를 사용하려면 `np.radians()` 또는 `np.deg2rad()`로 변환해야 합니다.

`np.isclose(arr, -90)`은 각 원소가 `-90`과 **허용 오차 내에서 같은지** 판단합니다.

모든 배열에 대해 일괄로 적용할 수 있으며, 특정 함수는 `자료형(dtype)`이 일치해야 사용할 수 있습니다.

예를 들어, **x 배열과 y 배열의 최대공약수**를 계산하는 `gcd` 함수는 **정수형(int)** 배열에만 연산이 가능합니다.

<br>
<br>

## 범용 함수

|                    함수                   |                                       설명                                      |
|:-----------------------------------------:|:-------------------------------------------------------------------------------:|
|         np.add( array1, array2 )        |                                   요소별 덧셈                                   |
|      np.subtract( array1, array2 )      |                                   요소별 뺄셈                                   |
|      np.multiply( array1, array2 )      |                                   요소별 곱셈                                   |
|       np.divide( array1, array2 )       |                                  요소별 나눗셈                                  |
|        np.power( array1, array2 )       |                                  요소별 거듭제곱                                 |
|             np.sqrt( array )            |                                  요소별 제곱근                                  |
|         np.mod( array1, array2 )        |                              요소별 나눗셈의 나머지                             |
|    np.floor_divide( array1, array2 )    |                            요소별 나눗셈 내림 처리                            |
|      np.logaddexp( array1, array2 )     |          요소별 지수의 합을 로그 처리<br>log(exp(array1)+exp(array2))          |
|     np.logaddexp2( array1, array2 )     |    요소별 2의 제곱의 합을 밑이 2인 로그 처리<br>log2(2\*\*array1 + 2\*\*array2)    |
|            np.gcd( array1, array2 )     |                              요소별 최대공약수                              |
|           np.positive( array )          |                          요소별 +array (값을 그대로 반환)                         |
|           np.negative( array )          |                            요소별 -array (부호 반전)                            |
|             np.abs( array )             |                                  요소별 절댓값                                  |
|            np.round( array )            |                    요소별 반올림<br>(0.5는 가장 가까운 짝수로 반올림)                    |
|             np.ceil( array )            |                                   요소별 올림                                   |
|            np.floor( array )            |                                   요소별 내림                                   |
|            np.trunc( array )            |                                   요소별 절사                                   |
|       np.maximum( array1, array2 )      |                                  요소별 최댓값                                  |
|       np.minimum( array1, array2 )      |                                  요소별 최솟값                                  |
|             np.max( array )             |                                  배열의 최댓값                                  |
|             np.min( array )             |                                  배열의 최솟값                                  |
|             np.argmax( array )          |                                  배열의 최댓값의 색인                            |
|             np.argmin( array )          |                                  배열의 최솟값의 색인                           |
|             np.exp( array )             |                                   요소별 지수                                   |
|             np.log( array )             |                               요소별 밑이 e인 로그                              |
|             np.log2( array )            |                               요소별 밑이 2인 로그                              |
|            np.log10( array )            |                              요소별 밑이 10인 로그                              |

<br>
<br>

## 삼각 함수

|                 함수                 |                     설명                     |
|:------------------------------------:|:--------------------------------------------:|
|           np.sin( array )          |                  요소별 사인                 |
|           np.cos( array )          |                 요소별 코사인                |
|           np.tan( array )          |                 요소별 탄젠트                |
|         np.arcsin( array )         |               요소별 아크 사인               |
|         np.arccos( array )         |              요소별 아크 코사인              |
|         np.arctan( array )         |              요소별 아크 탄젠트              |
|    np.arctan2( array1, array2 )    |    요소별 아크 탄젠트<br>array1 / array2    |
|          np.sinh( array )          |            요소별 하이퍼볼릭 사인            |
|          np.cosh( array )          |           요소별 하이퍼볼릭 코사인           |
|          np.tanh( array )          |           요소별 하이퍼볼릭 탄젠트           |
|         np.arcsinh( array )        |         요소별 하이퍼볼릭 아크 사인        |
|         np.arccosh( array )        |        요소별 하이퍼볼릭 아크 코사인       |
|         np.arctanh( array )        |        요소별 하이퍼볼릭 아크 탄젠트       |
|         np.deg2rad( array )        |         요소별 각도에서 라디안 변환        |
|         np.rad2deg( array )        |         요소별 라디안에서 각도 변환        |
|     np.hypot( array1, array2 )     |          요소별 유클리드 거리 계산         |

<br>
<br>

## 비트 연산 함수

|                   함수                   |              설명             |
|:----------------------------------------:|:-----------------------------:|
|    np.bitwise_and( array1, array2 )    |        요소별 AND 연산        |
|     np.bitwise_or( array1, array2 )    |         요소별 OR 연산        |
|    np.bitwise_xor( array1, array2 )    |        요소별 XOR 연산        |
|         np.bitwise_not( array )        |        요소별 NOT 연산        |
|     np.left_shift( array1, array2 )    |     요소별 LEFT SHIFT 연산    |
|    np.right_shift( array1, array2 )    |    요소별 RIGHT SHIFT 연산    |

<br>
<br>

## 비교 함수

|                    함수                    |                 설명                 |
|:------------------------------------------:|:------------------------------------:|
|       np.greater( array1, array2 )       |     요소별 array1 > array2 연산    |
|    np.greater_equal( array1, array2 )    |    요소별 array1 >= array2 연산    |
|         np.less( array1, array2 )        |     요소별 array1 < array2 연산    |
|      np.less_equal( array1, array2 )     |    요소별 array1 <= array2 연산    |
|        np.equal( array1, array2 )        |    요소별 array1 == array2 연산    |
|      np.not_equal( array1, array2 )      |    요소별 array1 != array2 연산    |
| np.isclose( array1, array2, rtol=r, atol=t ) | \|array1 - array2\| <= t + r * \|array2\| 이면 True, 아니면 False<br>(기본값 rtol=1e-05, atol=1e-08) |
|           np.isinf(array)           |           array가 inf이면 True, 아니면 False          |
|          np.isfinite(array)         | array가 inf, nan이면 False, 아니면 True               |
|           np.isnan(array)           | array가 nan이면 True, 아니면 False                    |

<br>
<br>

## 논리 함수

|                   함수                   |                    설명                   |
|:----------------------------------------:|:-----------------------------------------:|
|    np.logical_and( array1, array2 )    |    요소별 Boolean 자료형 논리 AND 연산    |
|     np.logical_or( array1, array2 )    |     요소별 Boolean 자료형 논리 OR 연산    |
|    np.logical_xor( array1, array2 )    |    요소별 Boolean 자료형 논리 XOR 연산    |
|         np.logical_not( array )        |    요소별 Boolean 자료형 논리 NOT 연산    |

<br>
<br>

## 배열 연산 함수

|                  함수                  |                 설명                |
|:--------------------------------------:|:-----------------------------------:|
|       np.dot( array1, array2 )       |       배열의 점곱(dot product)      |
|      np.cross( array1, array2 )      |     배열의 벡터곱(cross product)    |
|      np.inner( array1, array2 )      |      배열의 내적(inner product)     |
|      np.outer( array1, array2 )      |      배열의 외적(outer product)     |
|    np.tensordot( array1, array2, axes=2 )    |    배열의 텐서 축약(tensor dot product)    |
|            np.sum( array )           |            배열 원소의 합           |
|           np.mean( array )           |           배열 원소의 평균          |
|            np.std( array )           |        배열 원소의 표준편차        |
|            np.var( array )           |          배열 원소의 분산          |
|           np.prod( array )           |            배열 원소의 곱           |
|          np.cumsum( array )          |         배열 원소의 누적 합         |
|          np.cumprod( array )         |         배열 원소의 누적 곱         |
|           np.diff( array )           |           배열 원소별 차분          |
|         np.gradient( array )         |          배열 원소별 기울기         |
|      np.matmul( array1, array2 )     |            배열의 행렬 곱           |

<br>
<br>

## 상수

| 연산 |     의미     |
|:----:|:------------:|
| newaxis | 새로운 축 지정 |
|   e  |       e      |
|  pi  |       π      |
|  inf |       ∞      |
|  nan | Not a Number |
| euler_gamma | 오일러-마스케로니 상수 |

상수는 `np.pi`, `np.inf`처럼 `np.*` 형태로 사용합니다.

- Tip : `NumPy 2.0`부터 `np.PZERO`, `np.NZERO` 상수는 제거되었습니다. 양의 0과 음의 0은 `0.0`, `-0.0`을 직접 사용합니다. 또한 `τ(2π)`는 Numpy 상수로 제공되지 않으므로 `2 * np.pi` 또는 `math.tau`를 사용합니다.

<br>
<br>

## 축(axis) 기준 연산

{% highlight Python %}

import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]])

print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.mean(a, axis=0))
print(np.argmax(a, axis=1))

{% endhighlight %}

**결과**
:    
21<br>
[5 7 9]<br>
[ 6 15]<br>
[2.5 3.5 4.5]<br>
[2 2]<br>
<br>

`np.sum`, `np.mean`, `np.max`, `np.argmax` 등의 집계 함수는 `axis` 매개변수를 생략할 경우 **배열 전체**를 하나의 값으로 계산합니다.

`axis=0`을 입력할 경우 **행 방향(위에서 아래)**으로 계산하여 `열`마다 하나의 값을 반환하고, `axis=1`을 입력할 경우 **열 방향(왼쪽에서 오른쪽)**으로 계산하여 `행`마다 하나의 값을 반환합니다.

즉, `axis`로 지정한 축이 **계산 후 사라지는 축**입니다. `(2, 3)` 형태의 배열에 `axis=0`을 적용하면 `(3,)` 형태가, `axis=1`을 적용하면 `(2,)` 형태가 반환됩니다.

- Tip : `keepdims=True`를 함께 입력하면 사라지는 축을 크기 `1`로 유지하여, `(2, 3)` 배열의 `axis=1` 합계가 `(2, 1)` 형태로 반환됩니다. 원본 배열과 다시 연산할 때 유용합니다.

[12강]: https://076923.github.io/posts/Python-numpy-12/
