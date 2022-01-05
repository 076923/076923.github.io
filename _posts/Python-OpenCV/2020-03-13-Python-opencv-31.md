---
layout: post
title: "Python OpenCV 강좌 : 제 31강 - 이미지 연산 (2)"
tagline: "Python OpenCV Image Calculation (2)"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Calculation, OpenCV max, OpenCV min, OpenCV absdiff, OpenCV compare
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-31/
comments: true
toc: true
---

## 이미지 연산(Image Calculation) ##

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-31/1.webp){:class="lazyload" width="100%" height="100%"}

이미지 연산은 하나 또는 둘 이상의 이미지에 대해 수학적인 연산을 수행합니다.

Numpy 클래스의 배열 연산과 동일하거나 비슷한 의미와 결과를 갖습니다.

또한, `대수적 표현(+, - 등)`을 통해 Mat 클래스 간의 연산을 수행할 수 있습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import numpy as np
import cv2

src = cv2.imread("pencils.jpg")
number = np.ones_like(src) * 127

_max = cv2.max(src, number)
_min = cv2.min(src, number)
_abs = cv2.absdiff(src, number)
compare = cv2.compare(src, number, cv2.CMP_GT)

src = np.concatenate((src, src, src, src), axis = 1)
number = np.concatenate((number, number, number, number), axis = 1)
dst = np.concatenate((_max, _min, _abs, compare), axis = 1)

dst = np.concatenate((src, number, dst), axis = 0)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.imread("pencils.jpg")
number = np.ones_like(src) * 127

{% endhighlight %}

`원본 이미지(src)`와 `연산 값(number)`을 선언합니다.

연산 이미지는 `회색 이미지(127, 127, 127)`를 사용합니다.

<br>

{% highlight Python %}

_max = cv2.max(src, number)
_min = cv2.min(src, number)
_abs = cv2.absdiff(src, number)
compare = cv2.compare(src, number, cv2.CMP_GT)

{% endhighlight %}

`cv2.Calc(연산 이미지1, 연산 이미지2)`를 이용하여 **이미지 연산을 진행합니다.**

`최댓값(max)`, `최솟값(min)`, `절댓값 차이(absdiff)`, `비교(compare)` 등으로 연산이 가능합니다.

`최댓값 함수`는 두 이미지의 요소별 최댓값을 계산합니다.

`최솟값 함수`는 두 이미지의 요소별 최솟값을 계산합니다.

최댓값 함수와 최솟값 함수는 `정밀도`에 따라 요소의 최댓값과 최솟값이 있으며, **최댓값을 넘어가거나 최솟값보다 낮아질 수 없습니다.**

`절댓값 차이 함수`는 두 이미지의 요소별 절댓값 차이를 계산합니다.

덧셈 함수나 뺄셈 함수에서는 두 배열의 요소를 서로 뺄셈했을 때 음수가 발생하면 0을 반환했지만 **절댓값 차이 함수는 이 값을 절댓값으로 변경해서 양수 형태로 반환합니다.**

`비교 함수`는 요소별 두 이미지의 요소별 비교 연산을 수행합니다.

비교 결과가 `True`일 경우 요소의 값을 **255**로 변경하며, 비교 결과가 `False`일 경우 요소의 값을 **0**으로 변경합니다.

<br>

{% highlight Python %}

src = np.concatenate((src, src, src, src), axis = 1)
number = np.concatenate((number, number, number, number), axis = 1)
dst = np.concatenate((_max, _min, _abs, compare), axis = 1)

dst = np.concatenate((src, number, dst), axis = 0)

{% endhighlight %}

`연결 함수(np.concatenate)`로 이미지들을 연결합니다.

결과 이미지는 다음과 같이 구성됩니다.

<br>

| src | src | src | src |
| number1 | number1 | number2 | number2 |
| _max | _min | _abs | compare |

<br>
<br>

## 추가 정보

### 비교 함수 플래그 ###

|   플래그   |               설명               |
|:----------:|:--------------------------------:|
| cv2.CMP_EQ |     src1와 src2의 요소가 같음    |
| cv2.CMP_NE |  src1와 src2의 요소가 같지 않음  |
| cv2.CMP_GT |      src1와 src2의 요소가 큼     |
| cv2.CMP_GE | src1와 src2의 요소가 크거나 같음 |
| cv2.CMP_LT | src1와 src2의 요소가 작음        |
| cv2.CMP_LE | src1와 src2의 요소가 작거나 같음 |

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-31/2.webp){:class="lazyload" width="100%" height="100%"}
