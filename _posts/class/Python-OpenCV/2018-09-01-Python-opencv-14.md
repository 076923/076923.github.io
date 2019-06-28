---
layout: post
title: "Python OpenCV 강좌 : 제 14강 - 가장자리 검출"
tagline: "Python OpenCV Edge Image"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Edge Image
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-14/
comments: true
---

## 가장자리 검출(Edge) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch14/1.jpg)
영상이나 이미지를 `가장자리`를 검출 하기 위해 사용합니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/wheat.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(src, 100, 255)
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

cv2.imshow("canny", canny)
cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Canny Detailed Code ##
----------

{% highlight Python %}

canny = cv2.Canny(src, 100, 255)

{% endhighlight %}

`cv2.Canny(원본 이미지, 임계값1, 임계값2, 커널 크기, L2그라디언트)`를 이용하여 **가장자리 검출을 적용합니다.**

`임계값1`은 임계값1 `이하`에 포함된 가장자리는 가장자리에서 `제외`합니다.

`임계값2`는 임계값2 `이상`에 포함된 가장자리는 가장자리로 `간주`합니다.

`커널 크기`는 `Sobel` 마스크의 `Aperture Size`를 의미합니다. 포함하지 않을 경우, 자동으로 할당됩니다.

`L2그라디언트`는 `L2`방식의 사용 유/무를 설정합니다. 사용하지 않을 경우, 자동적으로 `L1그라디언트` 방식을 사용합니다.

<br>
<br>

`L2그라디언트` : $$ \sqrt{(dI/dx)^2 + (dI/dy)^2} $$

`L1그라디언트` : $$ \|dI/dx\| + \|dI/dy\| $$

<br>
<br>

## Sobel Detailed Code ##
----------

{% highlight Python %}

sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)

{% endhighlight %}

`cv2.Sobel(그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)`를 이용하여 **가장자리 검출을 적용합니다.**

`정밀도`는 결과 이미지의 `이미지 정밀도`를 의미합니다. **정밀도에 따라 결과물이 달라질 수 있습니다.**

`x 방향 미분`은 이미지에서 `x 방향`으로 미분할 값을 설정합니다.

`y 방향 미분`은 이미지에서 `y 방향`으로 미분할 값을 설정합니다.

`커널`은 소벨 커널의 크기를 설정합니다. `1`, `3`, `5`, `7`의 값을 사용합니다.

`배율`은 계산된 미분 값에 대한 `배율값`입니다.

`델타`는 계산전 미분 값에 대한 `추가값`입니다.

`픽셀 외삽법`은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 `추정`해서 값을 할당해야합니다.

이미지 밖의 픽셀을 외삽하는데 사용되는 **테두리 모드**입니다. `외삽 방식`을 설정합니다.

* Tip :  `x방향 미분 값`과 `y방향의 미분 값`의 합이 1 이상이여야 하며 각각의 값은 `0`보다 커야합니다.

<br>
<br>

## Laplacian Detailed Code ##
----------

{% highlight Python %}

laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

{% endhighlight %}

`cv2.Laplacian(그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법)`를 이용하여 **가장자리 검출을 적용합니다.**

`정밀도`는 결과 이미지의 `이미지 정밀도`를 의미합니다. **정밀도에 따라 결과물이 달라질 수 있습니다.**

`커널`은 **2차 미분 필터의 크기**를 설정합니다. `1`, `3`, `5`, `7`의 값을 사용합니다.

`배율`은 계산된 미분 값에 대한 `배율값`입니다.

`델타`는 계산전 미분 값에 대한 `추가값`입니다.

`픽셀 외삽법`은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 `추정`해서 값을 할당해야합니다.

이미지 밖의 픽셀을 외삽하는데 사용되는 **테두리 모드**입니다. `외삽 방식`을 설정합니다.

* Tip :  `커널`의 값이 1일 경우, `3x3 Aperture Size`를 사용합니다. **(중심값 = -4)**

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

## <center>Canny</center> ##

![2]({{ site.images }}/assets/images/Python/opencv/ch14/2.png)

<br>

## <center>Sobel</center> ##

![3]({{ site.images }}/assets/images/Python/opencv/ch14/3.png)

<br>

## <center>Laplacian</center> ##

![4]({{ site.images }}/assets/images/Python/opencv/ch14/4.png)


