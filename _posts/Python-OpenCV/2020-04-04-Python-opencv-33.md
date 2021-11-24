---
layout: post
title: "Python OpenCV 강좌 : 제 33강 - 히스토그램"
tagline: "Python OpenCV Histogram"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Histogram, OpenCV BINS, OpenCV DIMS, OpenCV RANGE, OpenCV calcHist, OpenCV normalize
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-33/
comments: true
toc: true
---

## 히스토그램(Histogram)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-33/1.jpg)

`히스토그램`이란 **도수 분포표** 중 하나로 데이터의 분포를 몇 개의 구간으로 나누고 **각 구간에 속하는 데이터를 시각적으로 표현한 막대그래프**입니다.

이미지에서 사용하는 히스토그램은 `X 축을 픽셀의 값`으로 사용하고 `Y 축을 해당 픽셀의 개수`로 표현합니다.

이미지의 픽셀값을 히스토그램으로 표현하면 이미지의 특성을 쉽게 확인할 수 있습니다.

히스토그램은 다음과 같은 세 가지의 중요한 요소를 갖고 있습니다.

<br>

1.	`빈도 수(BINS)`: 히스토그램 그래프의 X 축 간격
2.	`차원 수(DIMS)`: 히스토그램을 분석할 이미지의 차원
3.	`범위(RANGE)`: 히스토그램 그래프의 X 축 범위

<br>

`빈도 수`는 **히스토그램의 X 축 간격**입니다. 픽셀값의 범위는 0~255로 총 256개의 범위를 갖고 있으며, 빈도 수의 값이 8이라면 0 ~ 7, 8 ~ 15, …, 248 ~ 255의 범위로 총 32개의 막대가 생성됩니다.

`차원 수`는 이미지에서 분석하고자 하는 색상 차원을 의미합니다. 그레이스케일은 **단일 채널**이므로 `하나의 차원`에 대해 분석할 수 있고 색상 이미지는 **다중 채널**이므로 `세 개 이상`의 차원에 대해 분석할 수 있습니다.

`범위`는 이미지에서 측정하려는 **픽셀값의 범위**로서, 특정 픽셀값 영역에 대해서만 분석하게 하는 데 사용됩니다. 

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2
import numpy as np

src = cv2.imread("road.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
result = np.zeros((src.shape[0], 256), dtype=np.uint8)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
cv2.normalize(hist, hist, 0, result.shape[0], cv2.NORM_MINMAX)

for x, y in enumerate(hist):
    cv2.line(result, (x, result.shape[0]), (x, result.shape[0] - y), 255)

dst = np.hstack([gray, result])

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.imread("road.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
result = np.zeros((src.shape[0], 256), dtype=np.uint8)

{% endhighlight %}

`원본 이미지(src)`와 `그레이스케일(gray)`, `히스토그램 이미지(result)`을 선언합니다.

히스토그램 이미지는 N개의 개수를 갖는 256 분포로 사용할 예정이므로, 이미지 높이는 원본 이미지를 사용하며, 너비는 256을 사용합니다.

<br>

{% highlight Python %}

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

{% endhighlight %}

히스토그램은 `히스토그램 계산 함수(cv2.calcHist)`를 통해 분포를 계산할 수 있습니다.

`cv2.calcHist(연산 이미지, 특정 채널, 마스크, 히스토그램 크기, 히스토그램 범위)`를 이용하여 **히스토그램을 계산합니다.**

`특정 채널`은 **차원 수(DIMS)**를 설정합니다. 그레이스케일 이미지는 단일 채널이므로, 0을 사용합니다.

`마스크`는 특정 영역에 대해서만 연산할 때 사용합니다. 해당 영역은 없으므로, `None`을 할당합니다.

`히스토그램 크기`는 **빈도 수(BINS)**를 설정합니다. 픽셀의 범위는 `0 ~ 255` 이므로, `[256]`을 할당합니다.

`히스토그램 범위`는 **범위(RANGE)**를 설정합니다. 예외 사항이 없으므로, `0 ~ 255`의 범위를 계산하기 위해 `[0, 256]`을 할당합니다.

<br>

{% highlight Python %}

cv2.normalize(hist, hist, 0, result.shape[0], cv2.NORM_MINMAX)

{% endhighlight %}

히스토그램을 통해 연산된 결과는 정규화되지 않은 값입니다.

그러므로, `정규화 함수(cv2.normalize)`를 통해 값을 변경합니다.

`cv2.normalize(입력 배열, 결과 배열, alpha, beta, 정규화 기준)`으로 값을 정규화합니다.

`cv2.NORM_MINMAX`을 통해, 정규화 기준을 최솟값이 `alpha`가 되고, 최댓값이 `beta`가 되게 변경합니다.

이 연산을 통해 최솟값은 `0`이 되며, 최댓값은 `result.shape[0]`이 됩니다.

<br>

{% highlight Python %}

for x, y in enumerate(hist):
    cv2.line(result, (x, result.shape[0]), (x, result.shape[0] - y), 255)

dst = np.hstack([gray, result])

{% endhighlight %}

결과를 시각적으로 확인하기 위해, `hist`값을 `result`에 표시합니다.

이후, `gray`와 `result`는 이미지 높이가 같으므로 `병합 함수(np.hstack)`로 이미지를 연결합니다.

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-33/2.jpg)
