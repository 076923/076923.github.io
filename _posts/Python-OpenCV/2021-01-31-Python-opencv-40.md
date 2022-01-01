---
layout: post
title: "Python OpenCV 강좌 : 제 40강 - 리매핑"
tagline: "Python OpenCV Remapping"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Remapping, OpenCV remap
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-40/
comments: true
toc: true
---

## 리매핑(Remapping)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-40/1.webp){: width="100%" height="100%"}

`리매핑(Remapping)`은 입력 이미지에 기하학적인 변형을 적용하는 방법입니다.

**기하학적 변환**에서 다루었던 `아핀 변환(Affine Transform)`과 `원근 변환(Perspective Transform)`은 이미지에 변환 행렬을 적용하여, 이미지를 변경합니다.

리매핑은 이미지에 변환 행렬 연산을 적용하는 것이 아닌, 비선형 변환을 적용할 수 있습니다.

즉, **픽셀들의 좌표를 임의의 특정 좌표로 옮겨 이미지를 변경하는 작업을 의미합니다.**

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2
import numpy as np

src = cv2.imread("buildings.jpg")
height, width = src.shape[:2]
map2, map1 = np.indices((height, width), dtype=np.float32)

map1 = map1 + width / 100 * np.sin(map1)
map2 = map2 + height / 100 * np.cos(map2)

dst = cv2.remap(src, map1, map2, cv2.INTER_CUBIC)
cv2.imshow("dst", dst)
cv2.waitKey()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

map2, map1 = np.indices((height, width), dtype=np.float32)

{% endhighlight %}

`색인 배열 생성 함수(np.indices)`을 활용해 원본 이미지 크기와 동일한 색인 배열을 생성합니다.

`np.indices((높이, 너비), 정밀도)`를 의미하며, **Y축 좌표 색인 행렬**, **X축 좌표 색인 행렬**을 반환합니다.

여기서 이동되는 좌표의 값은 정수형이 아닌, 실수형을 갖습니다.

<br>

{% highlight Python %}

map1 = map1 + width / 100 * np.sin(map1)
map2 = map2 + height / 100 * np.cos(map2)

{% endhighlight %}

**map1**과 **map2**에 임의의 삼각 함수를 적용하여 행렬의 형태를 변경합니다.

map1과 map2는 매핑될 좌표의 값을 의미하므로, 해당 행렬의 값을 변경하면 최종 반환 이미지의 형태가 달라집니다.

이 행렬의 값을 대칭하거나 회전한다면, 기존의 기하학적 변환과 동일한 형태의 결과를 얻을 수 있습니다.

<br>

{% highlight Python %}

dst = cv2.remap(src, map1, map2, cv2.INTER_CUBIC)

{% endhighlight %}

`리매핑 함수(cv2.remap)`을 활용하여 원본 이미지에 리매핑을 적용합니다.

`cv2.remap(원본 이미지, X축 좌표 색인 행렬, Y축 좌표 색인 행렬, 보간법, 외삽법, 외삽 색상)`을 의미합니다.

원본 이미지의 픽셀 배열을 **X축 좌표 색인 행렬**과 **Y축 좌표 색인 행렬**의 값을 적용하여 픽셀들을 이동시킵니다.

색인 행렬의 값은 정수 좌표가 아니므로, 보간법과 외삽법을 적용합니다.

보간법과 외삽법은 `제 8강 - 크기 조절`과 `제 13강 - 흐림 효과`에서 내용을 확인하실 수 있습니다.

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-40/2.webp){: width="100%" height="100%"}
