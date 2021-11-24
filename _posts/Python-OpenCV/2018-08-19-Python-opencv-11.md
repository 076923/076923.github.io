---
layout: post
title: "Python OpenCV 강좌 : 제 11강 - 역상"
tagline: "Python OpenCV Reverse Image"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Reverse Image, OpenCV Bitwise Not
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-11/
comments: true
toc: true
---

## 역상(Reverse Image)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-11/1.jpg)

`역상(Reverse Image)`은 영상이나 이미지를 **반전 된 색상**으로 변환하기 위해서 사용합니다.

픽셀 단위마다 `비트 연산(Bitwise Operation)`을 적용하는데, 그중 **NOT 연산**을 적용합니다.

NOT 연산은 **각 자릿수의 값을 반대로 바꾸는 연산입니다.**

만약 **153**의 값을 갖는 픽셀에 NOT 연산을 적용한다면 **102**의 값으로 변경됩니다.

**153**은 `0b10011001`의 값을 가지며, **102**는 `0b01100110`의 값을 갖습니다.

즉, 10 진수의 픽셀값을 2 진수의 값으로 변경한 다음, 각 자릿수의 값을 반대로 바꾸게 됩니다.

**1**은 **0**이 되며, **0**은 **1**로 변경됩니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/whitebutterfly.jpg", cv2.IMREAD_COLOR)
dst = cv2.bitwise_not(src)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

dst = cv2.bitwise_not(src)

{% endhighlight %}

`NOT 연산 함수(cv2.bitwise_not)`로 이미지에 NOT 연산을 적용할 수 있습니다.

`dst = cv2.bitwise_not(src, mask)`는 `입력 이미지(src)`, `마스크(mask)`로 `출력 이미지(dst)`을 생성합니다.

`마스크`는 NOT 연산을 적용할 특정 영역을 의미합니다. 마스크 배열이 포함되어 있다면, 해당 영역만 반전 연산을 적용합니다. 

- Tip : `not` 연산 이외에도 `and`, `or`, `xor` 연산이 존재합니다.

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-11/2.png)
