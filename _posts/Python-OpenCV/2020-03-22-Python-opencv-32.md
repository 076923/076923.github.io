---
layout: post
title: "Python OpenCV 강좌 : 제 32강 - 비트 연산"
tagline: "Python OpenCV Bitwise"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Bitwise, OpenCV AND, OpenCV OR, OpenCV XOR, OpenCV NOT
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-32/
comments: true
toc: true
---

## 비트 연산(Bitwise)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-32/1.jpg)

비트 연산은 하나 또는 두 이미지에 대해 비트 연산을 수행합니다.

Numpy 클래스의 비트 연산과 동일한 의미와 결과를 갖습니다.

또한, `비트 연산 표현(&, | 등)`을 통해 Mat 클래스 간의 연산을 수행할 수 있습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import numpy as np
import cv2

src = cv2.imread("analysis.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

_and = cv2.bitwise_and(gray, binary)
_or = cv2.bitwise_or(gray, binary)
_xor = cv2.bitwise_xor(gray, binary)
_not = cv2.bitwise_not(gray)

src = np.concatenate((np.zeros_like(gray), gray, binary, np.zeros_like(gray)), axis = 1)
dst = np.concatenate((_and, _or, _xor, _not), axis = 1)
dst = np.concatenate((src, dst), axis = 0)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.imread("analysis.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

{% endhighlight %}

`원본 이미지(src)`와 `그레이스케일(gray)`, `이진화(binary)`을 선언합니다.

연산 이미지는 `그레이스케일` 이미지와 **127** 임곗값을 갖는 `이진화` 이미지를 사용합니다.

<br>

{% highlight Python %}

_and = cv2.bitwise_and(gray, binary)
_or = cv2.bitwise_or(gray, binary)
_xor = cv2.bitwise_xor(gray, binary)
_not = cv2.bitwise_not(gray)

{% endhighlight %}

`cv2.bitwise(연산 이미지1, 연산 이미지2)`를 이용하여 **비트 연산을 진행합니다.**

`논리곱(bitwise_and)`, `논리합(bitwise_or)`, `배타적 논리합(bitwise_xor)`, `부정(bitwise_not)` 등으로 연산이 가능합니다.

`논리곱 함수`는 두 이미지의 요소별 논리곱을 계산합니다.

**연산 이미지1**과 **연산 이미지2**의 값을 비트 단위로 파악하며, 해당 비트에 대해 `AND` 연산을 진행합니다.

`논리합 함수`는 두 이미지의 요소별 논리합을 계산합니다. 

**연산 이미지1**과 **연산 이미지2**의 값을 비트 단위로 파악하며, 해당 비트에 대해 `OR` 연산을 진행합니다.

`배타적 논리합 함수`는 두 이미지의 요소별 배타적 논리합을 계산합니다.

**연산 이미지1**과 **연산 이미지2**의 값을 비트 단위로 파악하며, 해당 비트에 대해 `XOR` 연산을 진행합니다.

`논리합 함수`는 두 이미지의 요소별 논리합을 계산합니다.

**연산 이미지1**의 값을 비트 단위로 파악하며, 해당 비트에 대해 `NOT` 연산을 진행합니다.

<br>

요소의 값이 각각 `198`, `255`인 이미지를 **배타적 논리합 비트 연산**을 진행한다면 다음과 같습니다.

`198`은 **1100 0110**이 되며, `255`는 **1111 1111**이 됩니다.

`XOR` 연산은 비트 값이 같으면 `0`, 다르다면 `1`이 됩니다.

각 자리수 마다 값을 비교한다면 `0011 1001`이 됩니다.

이 값을 10진수로 변경한다면, `57`이 됩니다.

그러므로, 이미지 요소 값은 `57`의 값으로 할당됩니다.

<br>

{% highlight Python %}

src = np.concatenate((np.zeros_like(gray), gray, binary, np.zeros_like(gray)), axis = 1)
dst = np.concatenate((_and, _or, _xor, _not), axis = 1)
dst = np.concatenate((src, dst), axis = 0)

dst = np.concatenate((src, number, dst), axis = 0)

{% endhighlight %}

`연결 함수(np.concatenate)`로 이미지들을 연결합니다.

결과 이미지는 다음과 같이 구성됩니다.

<br>

| None | gray | binary | None |
| _and | _or | _xor | _not |

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-32/2.jpg)
