---
layout: post
title: "Python OpenCV 강좌 : 제 17강 - 채널 분리 & 병합"
tagline: "Python OpenCV Split & Merge"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Split, OpenCV Merge
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-17/
comments: true
toc: true
---

## 채널 분리(Split) 및 병합(Merge)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-17/1.jpg)

`채널 분리(Split)`과 `병합(Merge)`은 영상이나 이미지의 색상 공간의 채널을 분리하거나 합치기 위해 사용합니다.

예를 들어, BGR 색상 공간을 **B(Blue)**, **G(Green)**, **R(Red)**로 분리해 단일 채널을 가진 배열로 반환할 수 있습니다.

분리된 채널의 값을 변경하거나 순서를 변경해, `GB(R/2)` 공간을 만들거나 새로운 색상 공간으로 변경할 수도 있습니다.

- Tip : OpenCV의 가산 혼합의 삼원색 **기본 배열순서**는 `BGR`입니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(src)
inverse = cv2.merge((r, g, b))

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.imshow("inverse", inverse)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

b, g, r = cv2.split(src)

{% endhighlight %}

`채널 분리 함수(cv2.split)`로 이미지에서 채널을 분리할 수 있습니다.

`mv = cv2.split(src)`는 `입력 이미지(src)`에서 채널을 분리해 `단일 채널 이미지 배열(mv)`을 생성합니다.

`mv`는 **목록(list)** 형식으로 반환되며, `b, g, r` 등으로 형태로 각 목록의 원솟값을 변수로 지정할 수 있습니다.

분리된 채널의 순서에 맞게 각 변수에 할당됩니다.

- Tip : 분리된 채널들은 `단일 채널`이므로 흑백 색상으로 표현됩니다.

<br>

{% highlight Python %}

inverse = cv2.merge((r, g, b))

{% endhighlight %}

`채널 병합 함수(cv2.merge)`로 분리된 채널을 병합해 하나의 이미지로 합칠 수 있습니다.

`dst = cv2.merge(mv)`로 `단일 채널 이미지 배열(mv)`를 병합해 `출력 이미지(dst)`를 생성합니다.

채널을 변형한 뒤에 **다시 합치거나 순서를 변경**해 병합할 수 있습니다.

순서가 변경될 경우, 원본 이미지와 다른 색상으로 표현될 수 있습니다.

<br>
<br>

## 추가 정보

### numpy 형식 채널 분리

{% highlight Python %}

b = src[:, :, 0]
g = src[:, :, 1]
r = src[:, :, 2]

{% endhighlight %}

`이미지[높이, 너비, 채널]`을 이용하여 **특정 영역**의 **특정 채널**만 불러올 수 있습니다.

`:, :,  n`을 입력할 경우, 이미지 **높이와 너비**를 그대로 반환하고 `n`번째 채널만 반환하여 적용합니다.

- Tip : `src[..., n]`의 형태로도 사용할 수 있습니다.

<br>

### 빈 이미지

{% highlight Python %}

height, width, channel = src.shape
zero = np.zeros((height, width, 1), dtype=np.uint8)
bgz = cv2.merge((b, g, zero))

{% endhighlight %}

검은색 빈 공간 이미지가 필요할 때는 `np.zeros((높이, 너비, 채널), dtype=정밀도)`을 이용해 빈 이미지를 생성할 수 있습니다.

`Blue, Green, Zero` 이미지를 병합할 경우, `Red` 채널 영역이 모두 `흑백`이미지로 변경됩니다.

- Tip : `import numpy as np`가 포함된 상태여야합니다.

- Tip : 특정 색상의 이미지를 생성하려는 경우에는 `np.full((높이, 너비, 채널), (b, g, r), dtype=정밀도)`을 이용해 특정 색상 이미지를 생성할 수 있습니다.

<br>
<br>

## 출력 결과

### Blue

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-17/2.png)

<br>

### Green

![3]({{ site.images }}/assets/posts/Python/OpenCV/lecture-17/3.png)

<br>

### Red

![4]({{ site.images }}/assets/posts/Python/OpenCV/lecture-17/4.png)

<br>

### Inverse

![5]({{ site.images }}/assets/posts/Python/OpenCV/lecture-17/5.png)

<br>

### Blue, Green, Zero

![6]({{ site.images }}/assets/posts/Python/OpenCV/lecture-17/6.png)
