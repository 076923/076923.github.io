---
layout: post
title: "Python OpenCV 강좌 : 제 10강 - 그레이스케일"
tagline: "Python OpenCV Grayscale"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Grayscale
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-10/
comments: true
---

## 그레이스케일(Grayscale) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch10/1.jpg)
영상이나 이미지의 색상을 `흑백`색상으로 변환하기 위해서 사용합니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/crow.jpg", cv2.IMREAD_COLOR)

dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

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

dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

{% endhighlight %}

`cv2.cvtcolor(원본 이미지, 색상 변환 코드)`를 이용하여 **이미지의 색상 공간을 변경할 수 있습니다.**

`색상 변환 코드`는 `원본 이미지 색상 공간`**2**`결과 이미지 색상 공간`을 의미합니다.

`원본 이미지 색상 공간`은 `원본 이미지`와 일치해야합니다.

* Tip : `BGR`은 `RGB` 색상 채널을 의미합니다. (Byte 역순)

<br>
<br>

## Additional Information ##
----------

## 채널 범위 ##

`CV_8U 이미지 값 범위` : 0 ~ 255

`CV_16U 이미지의 값 범위` : 0 ~ 65535

`CV_32F 이미지의 값 범위` : 0 ~ 1

<br>
<br>

## 색상 공간 코드 ##

|    속성    |               의미              |           비고          |
|:----------:|:-------------------------------:|:-----------------------:|
|     BGR    |      Blue, Green, Red 채널      |            -            |
|    BGRA    |   Blue, Green, Red, Alpha 채널  |            -            |
|     RGB    |      Red, Green, Blue 채널      |            -            |
|    RGBA    |   Red, Green, Blue, Alpha 채널  |            -            |
|    GRAY    |            단일 채널            |       그레이스케일      |
|   BGR565   |      Blue, Green, Red 채널      |      16 비트 이미지     |
|     XYZ    |           X, Y, Z 채널          |     CIE 1931 색 공간    |
|    YCrCb   |          Y, Cr, Cb 채널         |       YCC (크로마)      |
|     HSV    |   Hue, Saturation, Value 채널   |     색상, 채도, 명도    |
|     Lab    |           L, a, b 채널          |   반사율, 색도1, 색도2  |
|     Luv    |           L, u, v 채널          |         CIE Luv         |
|     HLS    | Hue, Lightness, Saturation 채널 |     색상, 밝기, 채도    |
|     YUV    |           Y, U, V 채널          |    밝기, 색상1, 색상2   |
| BG, GB, RG |            디모자이킹           | 단일 색상 공간으로 변경 |
|     _EA    |            디모자이킹           |      가장자리 인식      |
|    _VNG    |            디모자이킹           |     그라데이션 사용     |

<br>

`원본 이미지 색상 공간`**2**`결과 이미지 색상 공간`에 `색상 공간 코드`를 조합하여 사용할 수 있습니다.

`예)` `BGR2GRAY`는 `Blue, Green, Red 채널` 이미지를 `단일 채널, 그레이스케일` 이미지로 변경합니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch10/2.png)

