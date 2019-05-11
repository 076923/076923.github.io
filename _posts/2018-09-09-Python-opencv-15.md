---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 15강 - HSV"
crawlertitle: "Python OpenCV 강좌 : 제 15강 - HSV"
summary: "Python OpenCV HSV"
date: 2018-09-09
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### HSV (Hue, Saturation, Value) ###
----------
[![1]({{ site.images }}/Python/opencv/ch15/1.jpg)]({{ site.images }}/Python/opencv/ch15/1.jpg)
영상이나 이미지를 `색상`을 검출 하기 위해 사용합니다. 채널을 `Hue`, `Saturation`, `Value`로 분리하여 변환할 수 있습니다. 

<br>

* `색상 (Hue)` : `색의 질`입니다. 빨강, 노랑, 파랑이라고 하는 표현으로 나타내는 성질입니다.
* `채도 (Saturation)` : `색의 선명도`입니다. 아무것도 섞지 않아 맑고 깨끗하며 원색에 가까운 것을 채도가 높다고 표현합니다.
* `명도 (Value)` : `색의 밝기`입니다. 명도가 높을수록 백색에, 명도가 낮을수록 흑색에 가까워집니다.

<br>
<br>

### Main Code (1) ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

### Detailed Code ###
----------

{% highlight Python %}

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

{% endhighlight %}

초기 속성은 `BGR`이므로, `cv2.cvtColor()`를 이용하여 `HSV`채널로 변경합니다.

각각의 속성으로 분할하기 위해서 `cv2.split()`을 이용하여 채널을 분리합니다.

<br>

* Tip : 분리된 채널들은 `단일 채널`이므로 흑백의 색상으로만 표현됩니다.

<br>
<br>

### Result ###
----------

### Hue ###

[![2]({{ site.images }}/Python/opencv/ch15/2.png)]({{ site.images }}/Python/opencv/ch15/2.png)

<br>

### Saturation ###

[![2]({{ site.images }}/Python/opencv/ch15/3.png)]({{ site.images }}/Python/opencv/ch15/3.png)

<br>

### Value ###

[![2]({{ site.images }}/Python/opencv/ch15/4.png)]({{ site.images }}/Python/opencv/ch15/4.png)


<br>
<br>


### Main Code (2) ###
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

### Detailed Code ###
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

<br>

* 색상 (Hue) : 0 ~ 180의 값을 지닙니다.
* 채도 (Saturation) : 0 ~ 255의 값을 지닙니다.
* 명도 (Value) : 0 ~ 255의 값을 지닙니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch15/5.png)]({{ site.images }}/Python/opencv/ch15/5.png)


