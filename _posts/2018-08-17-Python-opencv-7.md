---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 7강 - 확대 & 축소"
crawlertitle: "Python OpenCV 강좌 : 제 7강 - 확대 & 축소"
summary: "Python OpenCV ZoomIn & ZoomOut"
date: 2018-08-17
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 이미지 피라미드 (Image Pyramid) ###
----------
[![1]({{ site.images }}/Python/opencv/ch7/1.jpg)]({{ site.images }}/Python/opencv/ch7/1.jpg)
`이미지 피라미드 (Image Pyramid)`란 이미지의 크기를 변화시켜 `원하는 단계까지 샘플링`하는 작업입니다. 영상이나 이미지를 `확대`, `축소`시켜 띄울 수 있습니다. 
 
<br>
<br>

### Main Code ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/fruits.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
dst = cv2.pyrUp(src, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT);
dst2 = cv2.pyrDown(src);

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

### Detailed Code ###
----------


{% highlight Python %}

width, height, channel = src.shape

{% endhighlight %}

`height, width, channel = src.shape`를 이용하여 해당 이미지의 `높이`, `너비`, `채널`의 값을 저장합니다.

`너비`와 `높이`를 이용하여 **dstsize (결과 이미지 크기)**을 설정합니다.

<br>
<br>

{% highlight Python %}

dst = cv2.pyrUp(src, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT);

{% endhighlight %}

`cv2.pyrUp(원본 이미지)`로 이미지를 `2배`로 **확대**할 수 있습니다.

`cv2.pyrUp(원본 이미지, 결과 이미지 크기, 픽셀 외삽법)`을 의미합니다.

`결과 이미지 크기`는 `pyrUp()`함수일 경우, `이미지 크기의 2배`로 사용합니다.

`픽셀 외삽법`은 이미지를 `확대` 또는 `축소`할 경우, 영역 밖의 픽셀은 `추정`해서 값을 할당해야합니다.

이미지 밖의 픽셀을 외삽하는데 사용되는 **테두리 모드**입니다. `외삽 방식`을 설정합니다.

<br>
<br>

{% highlight Python %}

dst2 = cv2.pyrDown(src);

{% endhighlight %}

`cv2.pyrDown(원본 이미지)`로 이미지를 `1/2배`로 **축소**할 수 있습니다.

`cv2.pyrUp()` 함수와 **동일한 매개변수를** 가집니다.

`결과 이미지 크기`는 `(width/2, height/2)`를 사용해야합니다.

<br>
<br>

### Additional Information ###
----------

`pyrUp()`과 `pyrDown()` 함수에서 `결과 이미지 크기`와 `픽셀 외삽법`은 기본값으로 설정된 인수를 할당해야하므로 `생략`하여 사용합니다.

피라미드 함수에서 `픽셀 외삽법`은 `cv2.BORDER_DEFAULT`만 사용할 수 있습니다.

이미지를 `1/8배`, `1/4배` ,`4배`, `8배` 등의 배율을 사용해야하는 경우, `반복문`을 이용하여 적용할 수 있습니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch7/2.png)]({{ site.images }}/Python/opencv/ch7/2.png)
