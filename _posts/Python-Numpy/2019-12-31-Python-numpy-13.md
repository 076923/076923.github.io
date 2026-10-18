---
layout: post
title: "Python Numpy 강좌 : 제 13강 - 메모리 레이아웃"
tagline: "Python Numpy Memory Layout"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Memory Layout, Row-Major, Column-Major, strides, flags, C_CONTIGUOUS, F_CONTIGUOUS, ascontiguousarray
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-13/
comments: true
toc: true
---

## 메모리 레이아웃(Memory Layout)

`메모리 레이아웃`이란 **Numpy 배열의 요솟값이 메모리에 정렬되는 방식**을 의미합니다.

컴퓨터의 메모리는 1차원으로 이어진 공간이므로, 2차원 이상의 배열을 저장하려면 요소들을 **한 줄로 나열하는 순서**를 정해야 합니다.

다차원 배열의 요솟값 정렬 방식은 크게, `행 우선(Row-Major)` 방식과 `열 우선(Column-Major)` 방식이 있습니다.

행 우선 방식은 첫 번째 행을 메모리에 넣은 다음 두 번째 행을 메모리에 넣습니다.

열 우선 방식은 첫 번째 열을 메모리에 넣은 다음 두 번째 열을 메모리에 넣습니다.

이런 방식으로 **행 또는 열의 순서대로 메모리 레이아웃의 구조가 형성**됩니다.

<br>
<br>

## 행 우선(Row-Major)

<img data-src="{{ site.images }}/assets/posts/Python/Numpy/lecture-13/1.webp" class="lazyload" width="100%" height="100%"/>

행 우선 방식은 위 그림과 같이 **행 인덱스를 고정한 채, 열 인덱스가 높아지는 순서**로 정렬됩니다.

즉, **마지막 색인이 가장 빠르게 변화**하는 순서로 할당됩니다. 행 우선 방식은 `C 언어 스타일`의 메모리 순서이며, Numpy 배열의 **기본값**입니다.

만약, 3차원 배열일 때 `[i][j][k]` 형태로 색인이 구성돼 있다면 `k`의 값부터 **순차적으로 증가**하고, `k` 색인이 **최댓값에 도달**하면 `j` 색인이 **증가**하는 구조입니다. 

메모리 레이아웃 구조를 볼 때, 같은 행의 요소들이 메모리에 **연속적으로** 놓입니다.

<br>
<br>

## 열 우선(Column-Major)

<img data-src="{{ site.images }}/assets/posts/Python/Numpy/lecture-13/2.webp" class="lazyload" width="100%" height="100%"/>

열 우선 방식은 위 그림과 같이 **열 인덱스를 고정한 채, 행 인덱스가 높아지는 순서**로 정렬됩니다.

즉, **첫 번째 색인이 가장 빠르게 변화**하는 순서로 할당됩니다. 열 우선 방식은 `Fortran 언어 스타일`의 메모리 순서이며, `MATLAB`이나 `R` 등도 이 방식을 사용합니다.

만약, 3차원 배열일 때 `[i][j][k]` 형태로 색인이 구성돼 있다면 `i`의 값부터 **순차적으로 증가**하고, `i` 색인이 **최댓값에 도달**하면 `j` 색인이 **증가**하는 구조입니다. 

메모리 레이아웃 구조를 볼 때, 같은 열의 요소들이 메모리에 **연속적으로** 놓입니다.

<br>
<br>

## Numpy의 메모리 레이아웃

{% highlight Python %}

import numpy as np

array = np.arange(10)

print(array)

arrayC = np.reshape(array, (2, -1), order='C')
arrayF = np.reshape(array, (2, -1), order='F')

print(arrayC)
print(arrayF)

{% endhighlight %}

**결과**
:    
[0 1 2 3 4 5 6 7 8 9]<br>
[[0 1 2 3 4]<br>
&nbsp;[5 6 7 8 9]]<br>
[[0 2 4 6 8]<br>
&nbsp;[1 3 5 7 9]]<br>
<br>

Numpy 배열에서는 `order` 매개변수의 인수를 통해 요소를 읽고 쓰는 순서를 설정할 수 있습니다.

`arrayC`는 `C 언어 스타일`의 순서로 요소를 채우며, `arrayF`는 `Fortran 언어 스타일`의 순서로 요소를 채웁니다.

`arrayC`는 `0 ~ 4`를 첫 번째 **행**에 먼저 채우고, `arrayF`는 `0, 1`을 첫 번째 **열**에 먼저 채우는 것을 확인할 수 있습니다.

`np.reshape`의 `order`는 **색인을 읽고 쓰는 순서**를 의미하며, 결과 배열이 실제로 어떤 메모리 레이아웃을 갖는지와는 별개의 개념입니다.

`order` 매개변수는 `C 언어 스타일`과 `Fortran 언어 스타일` 외에도 **원본 배열의 레이아웃을 따르는** 두 종류의 값을 추가로 사용할 수 있습니다.

단, 함수마다 허용하는 값이 다릅니다. `np.reshape`는 `'C'`, `'F'`, `'A'`를 허용하며, `np.array`, `ndarray.copy`, `np.ravel` 등은 `'K'`까지 허용합니다.

<br>
<br>

## 메모리 레이아웃 정렬 플래그

| order |                    의미                   |
|:-----:|:-----------------------------------------:|
|  'K'  |      원본 배열의 레이아웃에 최대한 일치      |
|  'A'  | 원본이 Fortran 연속 배열이면 'F', 아닐 경우 'C' |
|  'C'  |               C 언어 스타일               |
|  'F'  |            Fortran 언어 스타일            |

<br>
<br>

## 스트라이드(Strides)

{% highlight Python %}

import numpy as np

array = np.arange(6, dtype=np.int32).reshape(2, 3)

print(array.strides)
print(np.asfortranarray(array).strides)
print(array.T.strides)

{% endhighlight %}

**결과**
:    
(12, 4)<br>
(4, 8)<br>
(4, 12)<br>
<br>

`스트라이드(strides)`는 각 축의 색인이 **1 증가할 때 메모리에서 몇 바이트를 건너뛰어야 하는지**를 나타내는 튜플입니다.

Numpy는 이 값을 이용해 1차원 메모리 공간의 데이터를 다차원 배열처럼 해석합니다.

`int32` 자료형은 요소 하나가 `4` 바이트이므로, `(2, 3)` 형태의 `C 언어 스타일` 배열은 다음 열로 갈 때 `4` 바이트, 다음 행으로 갈 때 한 행 전체인 `3 × 4 = 12` 바이트를 건너뜁니다.

`np.asfortranarray`로 `Fortran 언어 스타일` 배열을 만들면 다음 행으로 갈 때 `4` 바이트, 다음 열로 갈 때 한 열 전체인 `2 × 4 = 8` 바이트를 건너뜁니다.

`전치(T)`는 데이터를 복사하지 않고 **스트라이드의 순서만 뒤바꾼** 배열을 반환합니다. 그러므로 `(12, 4)`가 `(4, 12)`로 바뀝니다.

- Tip : 메모리에서 연속된 요소를 순서대로 읽는 연산이 더 빠릅니다. `C 언어 스타일` 배열은 **마지막 축 방향**의 반복이, `Fortran 언어 스타일` 배열은 **첫 번째 축 방향**의 반복이 더 효율적입니다.

<br>
<br>

## 메모리 연속성(Contiguous)

{% highlight Python %}

import numpy as np

array = np.arange(6).reshape(2, 3)
transpose = array.T
contiguous = np.ascontiguousarray(transpose)

print(array.flags['C_CONTIGUOUS'], array.flags['F_CONTIGUOUS'])
print(transpose.flags['C_CONTIGUOUS'], transpose.flags['F_CONTIGUOUS'])
print(contiguous.flags['C_CONTIGUOUS'], contiguous.flags['F_CONTIGUOUS'])
print(np.shares_memory(array, transpose), np.shares_memory(array, contiguous))

{% endhighlight %}

**결과**
:    
True False<br>
False True<br>
True False<br>
True False<br>
<br>

`flags` 속성으로 배열의 메모리 레이아웃 상태를 확인할 수 있습니다.

`C_CONTIGUOUS`는 배열이 **C 언어 스타일로 연속된 메모리**에 저장돼 있는지, `F_CONTIGUOUS`는 **Fortran 언어 스타일로 연속된 메모리**에 저장돼 있는지를 의미합니다.

`array`는 기본값인 `C 언어 스타일` 배열이며, 이를 전치한 `transpose`는 같은 메모리를 공유하므로 `Fortran 언어 스타일`의 연속 배열이 됩니다.

`np.ascontiguousarray`는 입력 배열을 `C 언어 스타일`의 연속 배열로 반환합니다. 이미 연속 배열이라면 복사하지 않지만, 그렇지 않다면 **새로운 메모리에 복사**합니다.

`np.shares_memory`로 두 배열이 같은 메모리를 공유하는지 확인할 수 있으며, `transpose`는 원본과 메모리를 공유하지만 `contiguous`는 별도의 메모리를 사용하는 것을 확인할 수 있습니다.

- Tip : 슬라이싱(`array[:, ::2]`)이나 전치처럼 **뷰(View)**를 반환하는 연산은 연속성이 깨질 수 있습니다. C 언어 확장 모듈이나 일부 라이브러리는 연속 배열을 요구하므로, 이때 `np.ascontiguousarray`를 사용합니다.

<br>
<br>

## 레이아웃에 따른 평탄화

{% highlight Python %}

import numpy as np

array = np.arange(6).reshape(2, 3)

print(array.ravel(order='C'))
print(array.ravel(order='F'))
print(array.T.ravel(order='K'))

{% endhighlight %}

**결과**
:    
[0 1 2 3 4 5]<br>
[0 3 1 4 2 5]<br>
[0 1 2 3 4 5]<br>
<br>

`np.ravel`은 다차원 배열을 1차원 배열로 평탄화하며, `order` 매개변수에 따라 요소를 읽는 순서가 달라집니다.

`order='C'`는 행 방향으로, `order='F'`는 열 방향으로 요소를 읽습니다.

`order='K'`는 **실제 메모리에 저장된 순서**대로 요소를 읽습니다. `array.T`는 `[[0, 3], [1, 4], [2, 5]]`의 형태이지만, 메모리에는 원본의 순서인 `0 ~ 5`로 저장돼 있으므로 `[0 1 2 3 4 5]`를 반환합니다.

메모리 순서대로 읽을 수 있다면 `np.ravel`은 복사 없이 **뷰**를 반환하므로, `order='K'`는 불필요한 복사를 피할 때 유용합니다.
