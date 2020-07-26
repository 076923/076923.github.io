---
layout: post
title: "Python OpenCV 강좌 : 제 12강 - 이진화"
tagline: "Python OpenCV Binary"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Binary
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-12/
comments: true
---

## 이진화(Binary) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch12/1.jpg)
영상이나 이미지를 어느 지점을 기준으로 `흑색` 또는 `흰색`의 색상으로 변환하기 위해서 사용합니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/geese.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

{% endhighlight %}

`이진화`를 적용하기 위해서 `그레이스케일`로 변환합니다.

`ret, dst`를 이용하여 `이진화` 결과를 저장합니다. `ret`에는 `임계값`이 저장됩니다.

`cv2.threshold(그레스케일 이미지, 임계값, 최댓값, 임계값 종류)`를 이용하여 **이진화 이미지로 변경합니다.**

`임계값`은 이미지의 `흑백`을 나눌 기준값을 의미합니다. `100`으로 설정할 경우, `100`보다 이하면 `0`으로, `100`보다 이상이면 `최댓값`으로 변경합니다.

`임계값 종류`를 이용하여 **이진화할 방법** 설정합니다.

<br>
<br>

## Additional Information ##
----------

## 임계값 종류 ##

|          속성         |                    의미                   |
|:---------------------:|:-----------------------------------------:|
|   cv2.THRESH_BINARY   |    임계값 이상 = 최댓값 임계값 이하 = 0   |
| cv2.THRESH_BINARY_INV |    임계값 이상 = 0 임계값 이하 = 최댓값   |
|    cv2.THRESH_TRUNC   | 임계값 이상 = 임계값 임계값 이하 = 원본값 |
|   cv2.THRESH_TOZERO   |    임계값 이상 = 원본값 임계값 이하 = 0   |
| cv2.THRESH_TOZERO_INV |    임계값 이상 = 0 임계값 이하 = 원본값   |
|    cv2.THRESH_MASK    |             흑색 이미지로 변경            |
|    cv2.THRESH_OTSU    |             Otsu 알고리즘 사용            |
|  cv2.THRESH_TRIANGLE  |           Triangle 알고리즘 사용          |

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch12/2.png)
