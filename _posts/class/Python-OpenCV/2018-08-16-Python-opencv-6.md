---
layout: post
title: "Python OpenCV 강좌 : 제 6강 - 회전"
tagline: "Python OpenCV Rotate"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Rotate
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-6/
comments: true
---

## 회전 (Rotate) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch6/1.jpg)

`회전(Rotate)`은 선형 변환 중 하나에 포함되며, `회전 변환 행렬(Rotation matrix)`을 통해 변환이 진행됩니다.

회전 변환 행렬은 임의의 점을 중심으로 물체를 회전시킵니다. 회전 변환 행렬의 일부는 `반사 행렬(Reflection matrix)`과 같은 값을 지닐 수 있습니다.

2차원 유클리드 공간에서의 회전은 크게 두 가지 회전 행렬을 갖습니다. **좌푯값을 회전시키는 회전 행렬**과 *좌표 축을 회전시키는 회전 행렬*이 있습니다.

좌표 회전 행렬은 원점을 중심으로 좌푯값을 회전시켜 매핑하며, 좌표 축 회전 행렬은 원점을 중심으로 행렬 자체를 회전시켜 새로운 행렬의 값을 구성합니다.

OpenCV의 회전 함수는 좌표 축의 회전 이동 행렬과 동일한 형태이며, 비율을 조정하거나 중심점의 기준을 변경하여 회전할 수 있습니다. 

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/ara.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
dst = cv2.warpAffine(src, matrix, (width, height))

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

src = cv2.imread("Image/ara.jpg", cv2.IMREAD_COLOR)

{% endhighlight %}

`이미지 입력 함수(cv2.imread)`를 통해 원본 이미지로 사용할 `src`를 선언하고 **로컬 경로**에서 이미지 파일을 읽어 옵니다.

<br>
<br>

{% highlight Python %}

height, width, channel = src.shape

{% endhighlight %}

`height, width, channel = src.shape`를 이용하여 해당 이미지의 `높이`, `너비`, `채널`의 값을 저장합니다.

`높이`와 `너비`를 이용하여 **회전 중심점**을 설정합니다.

<br>
<br>

{% highlight Python %}

matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

{% endhighlight %}

`2×3 회전 행렬 생성 함수(cv2.getRotationMatrix2D)`로 회전 변환 행렬을 계산합니다.

`matrix = cv2.getRotationMatrix2D(center, angle, scale)`는 `중심점(center)`, `각도(angle)`, `비율(scale)`로 `매핑 변환 행렬(matrix)`을 생성합니다.

`중심점(center)`은 `튜플(Tuple)` 형태로 사용하며 회전의 **기준점**을 설정합니다.

`각도(angle)`는 중심점을 기준으로 **회전할 각도**를 설정합니다.

`비율(scale)`은 이미지의 **확대 및 축소 비율**을 설정합니다.

<br>
<br>

{% highlight Python %}

dst = cv2.warpAffine(src, matrix, (width, height))

{% endhighlight %}

`아핀 변환 함수(cv2.warpAffine)`로 회전 변환을 계산합니다.

`dst = cv2.warpAffine(src, M, dsize)`는 `원본 이미지(src)`에 `M(아핀 맵 행렬)`을 적용하고 `출력 이미지 크기(dsize)`로 변형해서 `출력 이미지(dst)`를 반환합니다.

`아핀 맵 행렬(M)`은 회전 행렬 생성 함수에서 반환된 매핑 변환 행렬을 사용합니다.

`출력 이미지 크기(dsize)`는 `튜플(Tuple)` 형태로 사용하며 출력 이미지의 너비와 높이를 의미합니다.

`아핀 맵 행렬`에 따라 `회전된 이미지`를 반환합니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch6/2.png)
