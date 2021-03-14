---
layout: post
title: "Python OpenCV 강좌 : 제 15강 - HSV"
tagline: "Python OpenCV HSV"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV HSV
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-15/
comments: true
---

## HSV(Hue, Saturation, Value) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch15/1.jpg)

`HSV(Hue, Saturation, Value)` 공간은 색상을 표현하기에 간편한 색상 공간입니다.

이미지에서 색상을 검출한다고 가정할 때 BGR이나 RGB 패턴으로는 **인간이 인지하는 영역의 색상을 구별하기에는 매우 어렵고 복잡합니다.**

하지만 HSV 색상 공간을 활용한다면 간편하고 빠르게 특정 색상을 검출하고 분리할 수 있습니다.

`색상(Hue)`은 빨간색, 노란색, 파란색 등으로 인식되는 색상 중 하나 또는 둘의 조합과 유사한 것처럼 보이는 **시각적 감각의 속성**을 의미합니다.

0°에서 360°의 범위로 표현되며, 파란색은 220°에서 260° 사이에 있습니다. OpenCV에서는 0 ~ 179의 범위로 표현됩니다.

`채도(Saturation)`는 이미지의 색상 깊이로, 색상이 얼마나 선명한(수수한) 색인지를 의미합니다.

아무것도 섞지 않아 맑고 깨끗하며 원색에 가까운 것을 채도가 높다고 표현합니다. 

0%에서 100%의 비율로 표현되며, 0%에 가까울수록 무채색, 100%에 가까울수록 가장 **선명한(순수한)색**이 됩니다. OpenCV에서는 0 ~ 255의 범위로 표현됩니다.

`명도(Value)`는 색의 밝고 어두운 정도를 의미합니다. 명도가 높을수록 색상이 밝아지며, 명도가 낮을수록 색상이 어두워집니다.

0%에서 100%의 비율로 표현되며, 0%에 가까울수록 검은색, 100%에 가까울수록 **가장 맑은색**이 됩니다. OpenCV에서는 0 ~ 255의 범위로 표현됩니다.

<br>

* Tip : 0 ~ 360의 범위는 **1 Byte(uint8)**의 범위를 벗어나게 되므로 불필요한 메모리 사용을 줄이기 위해, 절반의 값인 0 ~ 179의 범위로 표현합니다.

<br>
<br>

## Main Code (1) ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

{% endhighlight %}

초기 속성은 `BGR`이므로, `cv2.cvtColor()`를 이용하여 `HSV`채널로 변경합니다.

각각의 속성으로 분할하기 위해서 `cv2.split()`을 이용하여 채널을 분리합니다.

* Tip : 분리된 채널들은 `단일 채널`이므로 흑백의 색상으로만 표현됩니다.

<br>
<br>

## Result ##
----------

## <center>Hue</center> ##

![2]({{ site.images }}/assets/images/Python/opencv/ch15/2.png)

<br>

## <center>Saturation</center> ##

![3]({{ site.images }}/assets/images/Python/opencv/ch15/3.png)

<br>

## <center>Value</center> ##

![4]({{ site.images }}/assets/images/Python/opencv/ch15/4.png)

<br>
<br>

## Main Code (2) ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

h = cv2.inRange(h, 8, 20)
orange = cv2.bitwise_and(hsv, hsv, mask = h)
orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

cv2.imshow("orange", orange)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

h = cv2.inRange(h, 8, 20)
orange = cv2.bitwise_and(hsv, hsv, mask = h)
orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

{% endhighlight %}

`Hue`의 범위를 조정하여 **특정 색상만 출력**할 수 있습니다.

`cv2.inRange(단일 채널 이미지, 최솟값, 최댓값)`을 이용하여 범위를 설정합니다.

`주황색`은 약 `8~20` 범위를 갖습니다.

이 후, 해당 `마스크`를 **이미지 위에 덧씌워 해당 부분만 출력합니다.**

`cv2.bitwise_and(원본, 원본, mask = 단일 채널 이미지)`를 이용하여 `마스크`만 덧씌웁니다.

이 후, 다시 `HSV` 속성에서 `BGR` 속성으로 변경합니다.

* 색상 (Hue) : 0 ~ 180의 값을 지닙니다.
* 채도 (Saturation) : 0 ~ 255의 값을 지닙니다.
* 명도 (Value) : 0 ~ 255의 값을 지닙니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch15/5.png)