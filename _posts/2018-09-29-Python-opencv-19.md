---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 19강 - 기하학적 변환"
crawlertitle: "Python OpenCV 강좌 : 제 19강 - 기하학적 변환"
summary: "Python OpenCV WarpPerspective"
date: 2018-09-29
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 기하학적 변환  (Warp Perspective) ###
----------
[![1]({{ site.images }}/Python/opencv/ch19/1.jpg)]({{ site.images }}/Python/opencv/ch19/1.jpg)
영상이나 이미지 위에 `기하학적으로 변환`하기 위해 사용합니다. 영상이나 이미지를 **펼치거나 좁힐 수 있습니다.**

<br>

* Tip : `WarpPerspective`의 경우 4개의 점을 매핑합니다. (4개의 점을 이용한 변환)
* Tip : `WarpAffine`의 경우 3개의 점을 매핑합니다. (3개의 점을 이용한 변환)

<br>
<br>

### Main Code ###
----------

{% highlight Python %}

import numpy as np
import cv2

src = cv2.imread("Image/harvest.jpg", cv2.IMREAD_COLOR)
height, width, channel = src.shape

srcPoint=np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint=np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)

dst = cv2.warpPerspective(src, matrix, (width, height))

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

### Detailed Code ###
----------

{% highlight Python %}

srcPoint=np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint=np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

{% endhighlight %}

원본 이미지에서 4점 변환할 `srcPoint`와 결과 이미지의 위치가 될 `dstPoint`를 선언합니다.

좌표의 순서는 `좌상`, `우상`, `우하`, `좌하` 순서입니다. `numpy` 형태로 선언하며, 좌표의 순서는 **원본 순서와 결과 순서가 동일해야합니다.**

<br>

* Tip : `dtype`을 `float32` 형식으로 선언해야 사용할 수 있습니다.

<br>
<br>

{% highlight Python %}

matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)

{% endhighlight %}

`기하학적 변환`을 위하여 `cv2.getPerspectiveTransform(원본 좌표 순서, 결과 좌표 순서)`를 사용하여 `matrix`를 생성합니다.

다음과 같은 형식으로 매트릭스가 생성됩니다.

<br>

{% highlight Python %}

[[-2.88000000e+01 -9.60000000e+00  1.05600000e+04]
 [-4.44089210e-15 -2.15400000e+01  4.30800000e+03]
 [-1.77809156e-17 -2.00000000e-02  1.00000000e+00]]

{% endhighlight %}

<br>
<br>

{% highlight Python %}

dst = cv2.warpPerspective(src, matrix, (width, height))

{% endhighlight %}

`cv2.warpPerspective(원본 이미지, 매트릭스, (결과 이미지 너비, 결과 이미지 높이))`를 사용하여 이미지를 **변환할 수 있습니다.**

저장된 **매트릭스 값**을 사용하여 이미지를 변환합니다.

이외에도, `보간법`, `픽셀 외삽법`을 추가적인 파라미터로 사용할 수 있습니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch19/2.png)]({{ site.images }}/Python/opencv/ch19/2.png)



