---
layout: post
title: "Python Numpy 강좌 : 제 13강 - 메모리 레이아웃"
tagline: "Python Numpy Memory Layout"
image: /assets/images/numpy.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy Memory Layout
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-13/
comments: true
toc: true
---

## 메모리 레이아웃(Memory Layout)

`메모리 레이아웃`이란 **Numpy 배열의 요솟값 정렬 방식**을 의미합니다.

다차원 배열의 요솟값 정렬 방식은 크게, `행 우선(Column-Major)` 방식과 `열 우선(Row-Major)` 방식이 있습니다.

행 우선 방식은 첫 번째 행을 메모리에 넣은 다음 두 번째 행을 메모리에 넣습니다.

열 우선 방식은 첫 번째 열을 메모리에 넣은 다음 두 번째 열을 메모리에 넣습니다.

이런 방식으로 **행 또는 열의 순서대로 메모리 레이아웃의 구조가 형성**됩니다.

<br>
<br>

## 행 우선(Column-Major)

![1]({{ site.images }}/assets/posts/Python/Numpy/lecture-13/1.webp){: width="100%" height="100%"}

행 우선 방식은 위 그림과 같이 **행 인덱스가 가장 낮으면서, 열 인덱스가 높아지는 순서**로 정렬됩니다.

즉, **가장 빠르게 변화하는 색인의 순서로 할당**됩니다. 행 우선 방식은 `C 언어 스타일`의 메모리 순서입니다.

만약, 3차원 배열일 때 `[i][j][k]` 형태로 색인이 구성돼 있다면 `k`의 값부터 **순차적으로 증가**하고, `k` 색인이 **최댓값에 도달**하면 `j` 색인이 **증가**하는 구조입니다. 

메모리 레이아웃 구조를 볼 때, 행 요소는 한 번씩만 변화합니다.

<br>
<br>

## 열 우선(Row-Major)

![2]({{ site.images }}/assets/posts/Python/Numpy/lecture-13/2.webp){: width="100%" height="100%"}

열 우선 방식은 위 그림과 같이 **열 인덱스가 가장 낮으면서, 행 인덱스가 높아지는 순서**로 정렬됩니다.

즉, **가장 느리게 변화하는 색인의 순서로 할당**됩니다. 열 우선 방식은 `Fortran 언어 스타일`의 메모리 순서입니다.

만약, 3차원 배열일 때 `[i][j][k]` 형태로 색인이 구성돼 있다면 `i`의 값부터 **순차적으로 증가**하고, `i` 색인이 **최댓값에 도달**하면 `j` 색인이 **증가**하는 구조입니다. 

메모리 레이아웃 구조를 볼 때, 열 요소는 한 번씩만 변화합니다.

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

Numpy 배열에서는 `order` 매개변수의 인수를 통해 메모리 레이아웃을 설정할 수 있습니다.

`arrayC`는 `C 언어 스타일`의 정렬 방식을 가지며, `arrayF`는 `Fortran 언어 스타일`의 정렬 방식을 갖습니다.

Numpy 배열은 `C 언어 스타일`과 `Fortran 언어 스타일`로 정렬할 수 있습니다.

하지만, 크게 본다면 **네 종류**의 값으로 메모리 레이아웃을 정렬할 수 있습니다.

추가로 포함되는 두 종류의 값은 자동으로 `C 언어 스타일`에 맞출지, `Fortran 언어 스타일`에 맞출지 정렬하는 옵션입니다.

위의 예제 코드에서 확인할 수 있듯이, 정렬 방식에 따라 반환되는 값이 달라지는 것을 확인할 수 있습니다.

<br>
<br>

## 메모리 레이아웃 정렬 플래그

| order |                    의미                   |
|:-----:|:-----------------------------------------:|
|  'K'  |           레이아웃에 최대한 일치          |
|  'A'  | Fortran에 근접한 경우 'F', 아닐 경우 'C' |
|  'C'  |               C 언어 스타일               |
|  'F'  |            Fortran 언어 스타일            |
