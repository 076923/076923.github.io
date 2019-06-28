---
layout: post
title: "Python OpenCV 강좌 : 제 13강 - 흐림 효과"
tagline: "Python OpenCV Blur"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Blur
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-13/
comments: true
---

## 흐림 효과(Blur) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch13/1.jpg)
영상이나 이미지를 `흐림 효과`를 주어 번지게 하기 위해 사용합니다. 해당 픽셀의 `주변값들과 비교`하고 계산하여 픽셀들의 `색상 값을 재조정`합니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/geese.jpg", cv2.IMREAD_COLOR)

dst = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

dst = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

{% endhighlight %}

`cv2.blur(원본 이미지, (커널 x크기, 커널 y크기), 앵커 포인트, 픽셀 외삽법)`를 이용하여 **흐림 효과를 적용합니다.**

`커널 크기`는 이미지에 흐림 효과를 적용할 크기를 설정합니다. **크기가 클수록 더 많이 흐려집니다.**

`앵커 포인트`는 커널에서의 **중심점**을 의미합니다. `(-1, -1)`로 사용할 경우, 자동적으로 **커널의 중심점으로 할당합니다.**

`픽셀 외삽법`은 이미지를 흐림 효과 처리할 경우, 영역 밖의 픽셀은 `추정`해서 값을 할당해야합니다.

이미지 밖의 픽셀을 외삽하는데 사용되는 **테두리 모드**입니다. `외삽 방식`을 설정합니다.

<br>
<br>

## Additional Information ##
----------

## 픽셀 외삽법 종류 ##

|          속성          |                의미                |
|:----------------------:|:----------------------------------:|
|   cv2.BORDER_CONSTANT  |       iiiiii \| abcdefgh \| iiiiiii      |
|  cv2.BORDER_REPLICATE  |       aaaaaa \| abcdefgh \| hhhhhhh      |
|   cv2.BORDER_REFLECT   |       fedcba \| abcdefgh \| hgfedcb      |
|     cv2.BORDER_WRAP    |       cdefgh \| abcdefgh \| abcdefg      |
| cv2.BORDER_REFLECT_101 |       gfedcb \| abcdefgh \| gfedcba      |
|  cv2.BORDER_REFLECT101 |       gfedcb \| abcdefgh \| gfedcba      |
|   cv2.BORDER_DEFAULT   |       gfedcb \| abcdefgh \| gfedcba      |
| cv2.BORDER_TRANSPARENT |       uvwxyz \| abcdefgh \| ijklmno      |
|   cv2.BORDER_ISOLATED  | 관심 영역 (ROI) 밖은 고려하지 않음 |

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch13/2.png)


