---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 6강 - 회전"
crawlertitle: "Python OpenCV 강좌 : 제 6강 - 회전"
summary: "Python OpenCV Rotate"
date: 2018-08-16
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 회전 (Rotate) ###
----------
[![1]({{ site.images }}/Python/opencv/ch6/1.jpg)]({{ site.images }}/Python/opencv/ch6/1.jpg)
영상이나 이미지를 `회전`시켜 띄울 수 있습니다. `90°`, `45°`, `-45°` 등 다양한 각도로 회전이 가능합니다. 

<br>
<br>

### Main Code ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/ara.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
dst = cv2.warpAffine(src, matrix, (width, height))

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

src = cv2.imread("Image/ara.jpg", cv2.IMREAD_COLOR)

{% endhighlight %}

원본 이미지로 사용할 `src`를 선언하고 이미지를 불러옵니다.

<br>
<br>

{% highlight Python %}

height, width, channel = src.shape

{% endhighlight %}

`height, width, channel = src.shape`를 이용하여 해당 이미지의 `높이`, `너비`, `채널`의 값을 저장합니다.

`높이`와 `너비`를 이용하여 **회전 중심점**을 설정합니다.

<br>
<br>

{% highlight Python %}

matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

{% endhighlight %}

`matrix`에 `회전 배열`을 생성하여 저장합니다.

`cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)`을 설정합니다.

`중심점`은 `Tuple`형태로 사용하며 회전할 **기준점**을 설정합니다.

`각도`는 **회전할 각도**를 설정합니다.

`스케일`은 이미지의 **확대 비율**을 설정합니다.

<br>
<br>

{% highlight Python %}

dst = cv2.warpAffine(src, matrix, (width, height))

{% endhighlight %}

결과 이미지로 사용할 `dst`를 선언하고 회전 함수를 적용합니다.

`cv2.warpAffine(원본 이미지, 배열, (결과 이미지 너비, 결과 이미지 높이))`을 의미합니다.

`결과 이미지의 너비와 높이`로 크기가 선언되며 `배열`에 따라 이미지가 `회전`합니다.

<br>
<br>

### Additional Information ###
----------

`matrix`를 `numpy`형식으로 선언하여 `warpAffine`을 적용하여 변환할 수 있습니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch6/2.png)]({{ site.images }}/Python/opencv/ch6/2.png)
