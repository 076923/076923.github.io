---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 9강 - 자르기"
crawlertitle: "Python OpenCV 강좌 : 제 9강 - 자르기"
summary: "Python OpenCV Slice"
date: 2018-08-18
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 자르기 (Slice) ###
----------
[![1]({{ site.images }}/Python/opencv/ch9/1.jpg)]({{ site.images }}/Python/opencv/ch9/1.jpg)
영상이나 이미지의 크기를 `원하는 크기로 조절`할 수 있습니다.

<br>
<br>

### Main Code (1) ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/pawns.jpg", cv2.IMREAD_COLOR)

dst = src.copy() 
dst = src[100:600, 200:700]

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

dst = src.copy() 

{% endhighlight %}

이미지는 `numpy` 형식과 동일합니다. 이미지를 복제할 때, `dst=src`로 사용할 경우, 원본에도 영향을 미칩니다.

그러므로, `*.copy()`를 이용하여 `dst`에 이미지를 복제합니다.

<br>
<br>

{% highlight Python %}

dst = src[100:600, 200:700]

{% endhighlight %}

`dst` 이미지에 `src[높이(행), 너비(열)]`에서 **잘라낼 영역을 설정합니다.** `List`형식과 동일합니다.

<br>
<br>

### Main Code (2) ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/pawns.jpg", cv2.IMREAD_COLOR)

dst = src.copy() 
roi = src[100:600, 200:700]
dst[0:500, 0:500] = roi

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

dst = src.copy() 

{% endhighlight %}

이미지는 `numpy` 형식과 동일합니다. 이미지를 복제할 때, `dst=src`로 사용할 경우, 원본에도 영향을 미칩니다.

그러므로, `*.copy()`를 이용하여 `dst`에 이미지를 복제합니다.

<br>
<br>

{% highlight Python %}

roi = src[100:600, 200:700]
dst[0:500, 0:500] = roi

{% endhighlight %}

`roi`를 생성하여 `src[높이(행), 너비(열)]`에서 **잘라낼 영역을 설정합니다.** `List`형식과 동일합니다.

이후, `dst[높이(행), 너비(열)] = roi`를 이용하여 `dst` 이미지에 해당 영역을 붙여넣을 수 있습니다.

<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch9/2.png)]({{ site.images }}/Python/opencv/ch9/2.png)

<br>

[![3]({{ site.images }}/Python/opencv/ch9/3.png)]({{ site.images }}/Python/opencv/ch9/3.png)
