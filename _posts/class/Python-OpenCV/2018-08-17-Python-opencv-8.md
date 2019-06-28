---
layout: post
title: "Python OpenCV 강좌 : 제 8강 - 크기 조절"
tagline: "Python OpenCV Resize"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Resize
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-8/
comments: true
---

## 크기 조절(Resize) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch8/1.jpg)
영상이나 이미지의 크기를 `원하는 크기로 조절`할 수 있습니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/champagne.jpg", cv2.IMREAD_COLOR)

dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

{% endhighlight %}

`cv2.resize(원본 이미지, 결과 이미지 크기, 보간법)`로 이미지의 크기를 조절할 수 있습니다.

`결과 이미지 크기`는 `Tuple`형을 사용하며, `(너비, 높이)`를 의미합니다. 설정된 이미지 크기로 변경합니다.

`보간법`은 이미지의 크기를 변경하는 경우, 변형된 이미지의 픽셀은 `추정`해서 값을 할당해야합니다.

`보간법`을 이용하여 픽셀들의 값을 할당합니다.

<br>

{% highlight Python %}

dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

{% endhighlight %}

`cv2.resize(원본 이미지, dsize=(0, 0), 가로비, 세로비, 보간법)`로 이미지의 크기를 조절할 수 있습니다.

`결과 이미지 크기`가 `(0, 0)`으로 **크기를 설정하지 않은 경우**, `fx`와 `fy`를 이용하여 이미지의 비율을 조절할 수 있습니다.

`fx`가 `0.3`인 경우, 원본 이미지 너비의 `0.3배`로 변경됩니다.

`fy`가 `0.7`인 경우, 원본 이미지 높이의 `0.7배`로 변경됩니다.

* Tip : `결과 이미지 크기`와 `가로비`, `세로비`가 모두 설정된 경우, `결과 이미지 크기`의 값으로 이미지의 크기가 조절됩니다.

<br>
<br>

## Additional Information ##
----------

### interpolation 속성 ##

|          속성          |         의미        |
|:----------------------:|:-------------------:|
|    cv2.INTER_NEAREST   |     이웃 보간법     |
|    cv2.INTER_LINEAR    |    쌍 선형 보간법   |
| cv2.INTER_LINEAR_EXACT | 비트 쌍 선형 보간법 |
|     cv2.INTER_CUBIC    |   바이큐빅 보간법   |
|     cv2.INTER_AREA     |     영역 보간법     |
|   cv2.INTER_LANCZOS4   |    Lanczos 보간법   |

<br>

기본적으로 `쌍 선형 보간법`이 가장 많이 사용됩니다.

이미지를 `확대`하는 경우, `바이큐빅 보간법`이나 `쌍 선형 보간법`을 가장 많이 사용합니다.

이미지를 `축소`하는 경우, `영역 보간법`을 가장 많이 사용합니다.

`영역 보간법`에서 이미지를 `확대`하는 경우, `이웃 보간법`과 비슷한 결과를 반환합니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch8/2.png)
