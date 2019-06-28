---
layout: post
title: "Python OpenCV 강좌 : 제 5강 - 대칭"
tagline: "Python OpenCV Flip(Symmetry)"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Flip, OpenCV Symmetry
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-5/
comments: true
---

## 대칭 (Flip, Symmetry) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch5/1.jpg)
영상이나 이미지를 `대칭`시켜 띄울 수 있습니다. **상하** 또는 **좌우**방향으로 대칭할 수 있습니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/glass.jpg", cv2.IMREAD_COLOR)
dst = cv2.flip(src, 0)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

src = cv2.imread("Image/glass.jpg", cv2.IMREAD_COLOR)

{% endhighlight %}

원본 이미지로 사용할 `src`를 선언하고 이미지를 불러옵니다.

<br>

{% highlight Python %}

dst = cv2.flip(src, 0)

{% endhighlight %}

결과 이미지로 사용할 `dst`를 선언하고 `대칭 함수`를 적용합니다.

`cv2.flip(원본 이미지, 대칭 방법)`을 의미합니다.

**대칭 방법**은 **상수**를 입력하여 대칭시킬 수 있습니다.

`0`일 경우, `상하`방향으로 대칭합니다.

`1`일 경우, `좌우`방향으로 대칭합니다.

<br>

{% highlight Python %}

cv2.imshow("src", src)
cv2.imshow("dst", dst)

{% endhighlight %}

`cv2.imshow()`를 사용하여 이미지를 출력할 수 있습니다.

<br>
<br>

## Additional Information ##
----------

대칭 방법 중 **상수**를 `0`보다 **낮은 값**을 입력할 경우, `상하 대칭`으로 간주합니다.

대칭 방법 중 **상수**를 `1`보다 **높은 값**을 입력할 경우, `좌우 대칭`으로 간주합니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch5/2.png)
