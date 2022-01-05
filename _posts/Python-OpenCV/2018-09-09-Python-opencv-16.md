---
layout: post
title: "Python OpenCV 강좌 : 제 16강 - 배열 병합"
tagline: "Python OpenCV addWeighted"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV addWeighted
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-16/
comments: true
toc: true
---

## 배열 병합(addWeighted)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-16/1.webp){:class="lazyload" width="100%" height="100%"}

영상이나 이미지에서 `색상`을 검출 할 때, `배열 요소의 범위 설정 함수(cv2.inRange)`의 영역이 한정되어 색상을 설정하는 부분이 제한되어 있습니다.

예를 들어, 빨간색 영역을 검출하려 할 때, 빨간색 영역이 약 **0 ~ 5**와 약 **170 ~ 179**으로 범위가 두 가지로 나눠져 있습니다.

이 문제를 해결하려면 배열 요소의 범위 설정 함수를 두 개의 범위로 설정하고 검출한 두 요소의 배열을 병합해서 하나의 공간으로 만들어야 합니다.

이때 배열 병합 함수를 사용하며, **서로 다른 두 범위의 배열을 병합할 때 사용합니다.**

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
added_red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)

red = cv2.bitwise_and(hsv, hsv, mask = added_red)
red = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)

cv2.imshow("red", red)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
added_red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)

{% endhighlight %}

`빨간색` 영역은 `0 ~ 5`, `170 ~ 179`의 범위로 두 부분으로 나뉘어 있습니다.

이때, 두 부분을 합쳐서 한 번에 출력하기 위해서 사용합니다.

`배열 요소의 범위 설정 함수(cv2.inRange)`를 사용하여 빨간색 영역의 범위를 검출합니다.

배열 요소 범위 설정 함수는 다채널 이미지도 한 번에 범위를 설정할 수 있습니다.

색상을 분리한 두 배열을 `배열 병합 함수(cv2.addWeighted)`로 입력된 두 배열의 하나로 병합합니다.

`dst = cv2.addWeighted(src1, alpha, src2, beta, gamma, dtype = None)`은 `입력 이미지1(src1)`에 대한 `가중치1(alpha)` 곱과 `입력 이미지2(src2)`에 대한 `가중치2(beta)` 곱의 합에 `추가 합(gamma)`을 더해서 계산합니다.

`정밀도(dtype)`은 `출력 이미지(dst)`의 정밀도를 설정하며, 할당하지 않을 경우, **입력 이미지1**과 같은 정밀도로 할당됩니다.

두 이미지를 그대로 합칠 예정이므로, **가중치1**과 **가중치2**의 값은 **1.0**으로 사용하고, **추가 합**은 사용하지 않으므로 **0.0**을 할당합니다.

배열 병합 함수는 다음과 같은 수식으로 나타낼 수 있습니다.

<br>

$$ dst = src1 \times alpha + src2 \times beta + gamma $$

<br>

- Tip : 배열 병합 함수는 `알파 블렌딩(alpha blending)`을 구현할 수 있어 서로 다른 이미지를 불투명하게 혼합해서 표시할 수 있습니다.

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-16/2.webp){:class="lazyload" width="100%" height="100%"}
