---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 17강 - 채널 분리 & 병합"
crawlertitle: "Python OpenCV 강좌 : 제 17강 - 채널 분리 & 병합"
summary: "Python OpenCV Split & Merge"
date: 2018-09-15
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 채널 분리 & 병합  (Split & Merge) ###
----------
[![1]({{ site.images }}/Python/opencv/ch17/1.jpg)]({{ site.images }}/Python/opencv/ch17/1.jpg)
영상이나 이미지를 `채널`을 나누고 합치기 위해 사용합니다. 채널을 `B(Blue)`, `G(Green)`, `R(Red)`로 분리하여 채널을 변환할 수 있습니다. 

* Tip : OpenCV의 가산혼합의 삼원색 **기본 배열순서**는 `BGR`입니다.

<br>
<br>

### Main Code (1) ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(src)
inversebgr = cv2.merge((r, g, b))

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.imshow("inverse", inversebgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

### Detailed Code ###
----------

{% highlight Python %}

b, g, r = cv2.split(src)

{% endhighlight %}

`b, g, r = cv2.split(이미지)`를 이용하여 채널을 분리합니다.

채널에 순서의 맞게 각 변수에 대입됩니다.

분리된 채널들은 `단일 채널`이므로 흑백의 색상으로만 표현됩니다.


<br>
<br>

{% highlight Python %}

inversebgr = cv2.merge((r, g, b))

{% endhighlight %}

`cv2.merge((채널1, 채널2, 채널3))`을 이용하여 나눠진 채널을 **다시 병합**할 수 있습니다.

채널을 변형한 뒤에 **다시 합치거나 순서를 변경**하여 병합할 수 있습니다.

순서가 변경될 경우, 원본 이미지와 다른 색상으로 표현됩니다.

<br>
<br>

### Additional Information ###
----------

#### numpy 형식 채널 분리 ###

{% highlight Python %}

b = src[:,:,0]
g = src[:,:,1]
r = src[:,:,2]

{% endhighlight %}

`이미지[높이, 너비, 채널]`을 이용하여 **특정 영역**의 **특정 채널**만 불러올 수 있습니다.

`:, :,  n`을 입력할 경우, 이미지 `높이와 너비`를 그대로 반환하고 `n`번째 채널만 반환하여 적용합니다.

<br>

<br>

{% highlight Python %}

height, width, channel = src.shape
zero = np.zeros((height, width, 1), dtype = np.uint8)
bgz = cv2.merge((b, g, zero))

{% endhighlight %}

검은색 빈 공간 이미지가 필요할 때는 `np.zeros((높이, 너비, 채널), dtype=정밀도)`을 이용하여 빈 이미지를 생성할 수 있습니다.

`Blue, Green, Zero`이미지를 병합할 경우, `Red` 채널 영역이 모두 `흑백`이미지로 변경됩니다.

<br>

* Tip : `import numpy as np`가 포함된 상태여야합니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch17/2.png)]({{ site.images }}/Python/opencv/ch17/2.png)

[![3]({{ site.images }}/Python/opencv/ch17/3.png)]({{ site.images }}/Python/opencv/ch17/3.png)

[![4]({{ site.images }}/Python/opencv/ch17/4.png)]({{ site.images }}/Python/opencv/ch17/4.png)

[![5]({{ site.images }}/Python/opencv/ch17/5.png)]({{ site.images }}/Python/opencv/ch17/5.png)

[![6]({{ site.images }}/Python/opencv/ch17/6.png)]({{ site.images }}/Python/opencv/ch17/6.png)


