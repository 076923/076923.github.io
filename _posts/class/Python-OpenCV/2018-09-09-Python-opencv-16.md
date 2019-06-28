---
layout: post
title: "Python OpenCV 강좌 : 제 16강 - 채널 범위 병합"
tagline: "Python OpenCV addWeighted"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV addWeighted
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-16/
comments: true
---

## 채널 범위 병합(addWeighted) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch16/1.jpg)

영상이나 이미지를 `색상`을 검출 할 때, `cv2.inRange()`의 영역이 한정되어 색상을 설정하는 부분이 한정되어 있습니다.

이 때 **특정 범위들을 병합할 때 사용합니다.**

<br>
<br>

## Main Code  ##
----------

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
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
added_red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)

{% endhighlight %}

`빨간색` 영역은 `0 ~ 5`, `170 ~ 180`의 범위로 두부분으로 나뉘어 있습니다.

이 때, 두 부분을 합쳐서 한 번에 출력하기 위해서 사용합니다.

`cv2.inRange(다채널 이미지, (채널1 최솟값, 채널2 최솟값, 채널3 최솟값), (채널1 최댓값, 채널2 최댓값, 채널3 최댓값))`을 통하여 다채널 이미지도 한 번에 범위를 설정할 수 있습니다.

`HSV` 형식이므로 각각의 `h`, `s`, `v` 범위를 한 번에 설정합니다.

분리된 채널을 `cv2.addWeighted(이미지1, 이미지1 비율, 이미지2, 이미지2 비율, 가중치)`를 이용하여 채널을 하나로 합칠 수 있습니다.

두 이미지의 채널을 그대로 합칠 예정이므로 각각의 비율은 `1.0`으로 사용하고, 가중치는 사용하지 않으므로 `0.0`을 할당합니다.

`cv2.inRange()`를 사용할 때, 단일 채널 이미지의 범위만 할당하여 병합할 수 도 있습니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch16/2.png)
