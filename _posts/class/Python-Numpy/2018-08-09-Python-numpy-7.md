---
layout: post
title: "Python Numpy 강좌 : 제 7강 - 연산"
tagline: "Python Numpy calculate"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Numpy']
keywords: Python, Python Numpy, Numpy calculate
ref: Python-Numpy
category: posts
permalink: /posts/Python-numpy-7/
comments: true
---

## 기본 연산 ##
----------

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

사칙연산과 관련된 연산은 `array`와 `array` 사이에 **연산 기호(+. -, *, /)를 포함하여 계산할 수 있습니다.**

사칙연산의 혼합계산은 일반 수식과 동일하게 `곱하기(*)`와 `나누기(/)`를 우선적으로 연산합니다.

배열의 차원구조가 동일하지 않더라도, `브로드캐스팅(Broadcasting)` 조건에 포함된다면 연산이 가능합니다.

<br>
<br>

## 배열 연산 ##
----------

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

주의사항으로는 `벡터곱(cross product)`과 `외적(outer product)`은 Numpy 함수에서는 다른 연산입니다.

연산시 의도하는 연산이 `벡터곱(cross product)`인지 `외적(outer product)`인지 명확하게 인지 후 활용해야 합니다.

<br>
<br>

## 심화 연산 ##
----------

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

모든 배열에 대해 일괄로 적용할 수 있으며, 특정 함수는 `자료형(dtype)`이 일치해야 사용할 수 있습니다.

예를 들어, **x 배열과 y 배열의 최대공약수**를 계산하는 `gcd` 함수는 **정수형(int)** 배열에만 연산이 가능합니다.

<br>
<br>

## 범용 함수 ##
----------

|                    함수                   |                                       설명                                      |
|:-----------------------------------------:|:-------------------------------------------------------------------------------:|
|         np.add( array1, array2 )        |                                   요소별 덧셈                                   |
|      np.subtract( array1, array2 )      |                                   요소별 뺄셈                                   |
|      np.multiply( array1, array2 )      |                                   요소별 곱셈                                   |
|       np.divide( array1, array2 )       |                                  요소별 나눗셈                                  |
|        np.power( array1, array2 )       |                                   요소별 제곱                                   |
|             np.sqrt( array )            |                                  요소별 제곱근                                  |
|         np.mod( array1, array2 )        |                              요소별 나눗셈의 나머지                             |
|    np.floor_divide( array1, array2 )    |                            요소별 나눗셈 내림 처리                            |
|      np.logaddexp( array1, array2 )     |          요소별 지수의 합을 로그 처리<br>log(exp(array1)+exp(array2))          |
|     np.logaddexp2( array1, array2 )     |    요소별 2의 제곱의 합을 밑이 2인 로그 처리<br>log2(2\*\*array1 + 2\*\*array2)    |
|            np.gcd( array1, array2 )     |                              요소별 최대공약수                              |
|           np.positive( array )          |                                  요소별 양수 곱                                 |
|           np.negative( array )          |                                  요소별 음수 곱                                 |
|             np.abs( array )             |                                  요소별 절댓값                                  |
|            np.round( array )            |                                  요소별 반올림                                  |
|             np.ceil( array )            |                                   요소별 올림                                   |
|            np.floor( array )            |                                   요소별 내림                                   |
|            np.trunc( array )            |                                   요소별 절사                                   |
|       np.maximum( array1, array2 )      |                                  요소별 최댓값                                  |
|       np.minimum( array1, array2 )      |                                  요소별 최솟값                                  |
|             np.max( array )             |                                  배열의 최댓값                                  |
|             np.min( array )             |                                  배열의 최솟값                                  |
|             np.argmax( array )          |                                  배열의 최댓값의 색인                            |
|             np.argmin( array )          |                                  배열의 최솟값의 색안                           |
|             np.exp( array )             |                                   요소별 지수                                   |
|             np.log( array )             |                               요소별 밑이 e인 로그                              |
|             np.log2( array )            |                               요소별 밑이 2인 로그                              |
|            np.log10( array )            |                              요소별 밑이 10인 로그                              |

<br>
<br>

## 삼각 함수 ##
----------

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

## 비트 연산 함수 ##
----------

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

## 비교 함수 ##
----------

|                    함수                    |                 설명                 |
|:------------------------------------------:|:------------------------------------:|
|       np.greater( array1, array2 )       |     요소별 array1 > array2 연산    |
|    np.greater_equal( array1, array2 )    |    요소별 array1 >= array2 연산    |
|         np.less( array1, array2 )        |     요소별 array1 < array2 연산    |
|      np.less_equal( array1, array2 )     |    요소별 array1 <= array2 연산    |
|        np.equal( array1, array2 )        |    요소별 array1 == array2 연산    |
|      np.not_equal( array1, array2 )      |    요소별 array1 != array2 연산    |
| np.isclose( array1, array2, rel_tol=z ) | arra1와 array2가 (z*1e+02)% 내외로 가까우면 True, 아니면 False |
|           np.isinf(array)           |           array가 inf이면 True, 아니면 False          |
|          np.isfinite(array)         | array가 inf, nan이면 False, 아니면 True               |
|           np.isnan(array)           | array가 nan이면 True, 아니면 False                    |

<br>
<br>

## 논리 함수 ##
----------

|                   함수                   |                    설명                   |
|:----------------------------------------:|:-----------------------------------------:|
|    np.logical_and( array1, array2 )    |    요소별 Boolean 자료형 논리 AND 연산    |
|     np.logical_or( array1, array2 )    |     요소별 Boolean 자료형 논리 OR 연산    |
|    np.logical_xor( array1, array2 )    |    요소별 Boolean 자료형 논리 XOR 연산    |
|         np.logical_not( array )        |    요소별 Boolean 자료형 논리 NOT 연산    |

<br>
<br>

## 논리 함수 ##
----------

|                  함수                  |                 설명                |
|:--------------------------------------:|:-----------------------------------:|
|       np.dot( array1, array2 )       |       배열의 점곱(dot product)      |
|      np.cross( array1, array2 )      |     배열의 벡터곱(cross product)    |
|      np.inner( array1, array2 )      |      배열의 내적(inner product)     |
|      np.outer( array1, array2 )      |      배열의 외적(outer product)     |
|    np.tensordot( array1, array2 )    |    배열의 텐서곱(tensor product)    |
|            np.sum( array )           |            배열 원소의 합           |
|           np.prod( array )           |            배열 원소의 곱           |
|          np.cumsum( array )          |         배열 원소의 누적 합         |
|          np.cumprod( array )         |         배열 원소의 누적 곱         |
|           np.diff( array )           |           배열 원소별 차분          |
|         np.gradient( array )         |          배열 원소별 기울기         |
|      np.matmul( array1, array2 )     |            배열의 행렬 곱           |

<br>
<br>

## 상수 ##
----------

| 연산 |     의미     |
|:----:|:------------:|
| newaxis | 새로운 축 지정 |
|   e  |       e      |
|  pi  |       π      |
|  tau |       τ      |
|  inf |       ∞      |
| PZERO | 양의 0.0 |
| NZERO | 음의 0.0 |
|  nan | Not a Number |
| euler_gamma | 오일러 감마 |

<br>
<br>

