---
layout: post
title: "Python OpenCV 강좌 : 제 10강 - 색상 공간 변환"
tagline: "Python OpenCV Convert Color"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Grayscale, OpenCV Convert Color
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-10/
comments: true
toc: true
---

## 색상 공간 변환(Convert Color)

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-10/1.webp" class="lazyload" width="100%" height="100%"/>

`색상 공간 변환(Convert Color)`은 본래의 색상 공간에서 다른 색상 공간으로 변환할 때 사용합니다.

색상 공간 변환 함수는 **데이터 타입을 같게 유지하고 채널을 변환합니다.**

입력된 이미지는 `8 비트`, `16 비트`, `32 비트`의 정밀도를 갖는 배열을 사용할 수 있습니다.

출력된 이미지는 입력된 이미지의 이미지 크기와 정밀도가 동일한 배열이 됩니다.

채널의 수가 감소하게 되어 이미지 내부의 데이터는 설정한 색상 공간과 일치하는 값으로 변환되며, **데이터 값이 변경되거나 채널 순서가 변경될 수 있습니다.**

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/crow.jpg", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

{% endhighlight %}

`색상 공간 변환 함수(cv2.cvtcolor)`로 이미지의 색상 공간을 변경할 수 있습니다.

`dst = cv2.cvtcolor(src, code, dstCn)`는 `입력 이미지(src)`, `색상 변환 코드(code)`, `출력 채널(dstCn)`으로 `출력 이미지(dst)`을 생성합니다.

`색상 변환 코드`는 `원본 이미지 색상 공간`**2**`결과 이미지 색상 공간`을 의미합니다.

`원본 이미지 색상 공간`은 `원본 이미지`와 일치해야합니다.

`출력 채널`은 출력 이미지에 필요한 채널의 수를 설정합니다.

- Tip : `BGR`은 `RGB` 색상 채널을 의미합니다. (Byte 역순)
- Tip : 출력 채널은 기본값을 사용하여 자동으로 채널의 수를 결정하게 합니다.

<br>
<br>

## 추가 정보

### 채널 범위

| 형식 | 범위 |
|:-----:|:------:|
| CV_8U | 0 ~ 255 |
| CV_16U | 0 ~ 65535 |
| CV_32F | 0 ~ 1 |

<br>

### 색상 공간 코드

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

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-10/2.webp" class="lazyload" width="100%" height="100%"/>

