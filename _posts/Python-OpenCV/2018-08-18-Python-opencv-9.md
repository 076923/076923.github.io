---
layout: post
title: "Python OpenCV 강좌 : 제 9강 - 자르기"
tagline: "Python OpenCV Slice"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Slice
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-9/
comments: true
toc: true
---

## 자르기(Slice)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-9/1.jpg)

`자르기(Slice)`는 영상이나 이미지에서 특정 영역을 잘라내는 연산을 의미하니다.

특정 영역을 잘라내는 것을 `관심 영역(Region Of Interest, ROI)`이라 하며, 이미지 상에서 관심 있는 영역을 의미합니다.

이미지를 처리할 때 **객체를 탐지하거나 검출하는 영역**을 명확하게 관심 영역이라 볼 수 있습니다.

관심 영역에만 알고리즘을 적용한다면, 불필요한 연산이 줄어들고 정확도가 늘어나는 효과를 얻을 수 있습니다.

<br>
<br>

## 메인 코드 (1)

{% highlight Python %}

import cv2

src = cv2.imread("apple.jpg", cv2.IMREAD_COLOR)
dst = src[100:600, 200:700].copy()

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

dst = src[100:600, 200:700].copy()

{% endhighlight %}

OpenCV의 이미지는 이미지는 `numpy` 배열 형식과 동일합니다.

`src` 이미지에 `src[높이(행), 너비(열)]`로 **관심 영역을 설정합니다.**

`리스트(List)`나 `배열(Array)`의 특정 영역을 자르는 방식과 동일합니다.

이미지를 자르거나 복사할 때, `dst = src`의 형태로 사용할 경우, **얕은 복사(shallow copy)**가 되어 원본도 영향을 받게 됩니다.

그러므로, `*.copy()`를 이용해 **깊은 복사(deep copy)**를 진행합니다.

<br>
<br>

## 메인 코드 (2)

{% highlight Python %}

import cv2

src = cv2.imread("Image/pawns.jpg", cv2.IMREAD_COLOR)

dst = src.copy() 
roi = src[100:600, 200:700]
dst[0:500, 0:500] = roi

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

dst = src.copy() 

{% endhighlight %}

`dst` 이미지를 생성할 때, `dst = src.copy()`가 아닌 `dst = src`로 적용한다면 깊은 복사가 적용되지 않습니다.

얕은 복사로 이미지를 복사할 경우, `dst` 이미지와 `src` 이미지는 동일한 결과로 반환됩니다.

<br>

{% highlight Python %}

roi = src[100:600, 200:700]
dst[0:500, 0:500] = roi

{% endhighlight %}

`roi` 이미지를 생성하여 `src[높이(행), 너비(열)]`로 **관심 영역을 설정합니다.**

이후, `dst[높이(행), 너비(열)] = roi`를 이용하여 `dst` 이미지에 해당 영역을 붙여넣을 수 있습니다.

잘라낸 이미지와 붙여넣을 이미지의 크기가 다르다면 이미지를 붙여넣을 수 없습니다.

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-9/2.png)

<br>

![3]({{ site.images }}/assets/posts/Python/OpenCV/lecture-9/3.png)
