---
layout: post
title: "Python OpenCV 강좌 : 제 11강 - 역상"
tagline: "Python OpenCV Reverse Image"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Reverse Image
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-11/
comments: true
---

## 역상(Reverse Image) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch11/1.jpg)
영상이나 이미지를 `반전 된 색상`으로 변환하기 위해서 사용합니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/whitebutterfly.jpg", cv2.IMREAD_COLOR)

dst = cv2.bitwise_not(src)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

dst = cv2.bitwise_not(src)

{% endhighlight %}

`cv2.bitwise_not(원본 이미지)`를 이용하여 **이미지의 색상을 반전할 수 있습니다.**

`비트 연산`을 이용하여 **색상을 반전시킵니다.**

* Tip : `not` 연산 이외에도 `and`, `or`, `xor` 연산이 존재합니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch11/2.png)

