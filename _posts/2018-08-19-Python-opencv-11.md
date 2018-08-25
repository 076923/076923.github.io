---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 11강 - 역상"
crawlertitle: "Python OpenCV 강좌 : 제 11강 - 역상"
summary: "Python OpenCV Reverse Image"
date: 2018-08-19
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 역상 (Reverse Image) ###
----------
[![1]({{ site.images }}/Python/opencv/ch11/1.jpg)]({{ site.images }}/Python/opencv/ch11/1.jpg)
영상이나 이미지를 `반전 된 색상`으로 변환하기 위해서 사용합니다.


<br>
<br>

### Main Code ###
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

### Detailed Code ###
----------

{% highlight Python %}

dst = cv2.bitwise_not(src)

{% endhighlight %}

`cv2.bitwise_not(원본 이미지)`를 이용하여 **이미지의 색상을 반전할 수 있습니다.**

`비트 연산`을 이용하여 **색상을 반전시킵니다.**

<br>

* Tip : `not` 연산 이외에도 `and`, `or`, `xor` 연산이 존재합니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch11/2.png)]({{ site.images }}/Python/opencv/ch11/2.png)


