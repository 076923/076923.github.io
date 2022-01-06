---
layout: post
title: "Python OpenCV 강좌 : 제 42강 - K-평균 군집화 알고리즘"
tagline: "Python OpenCV Remapping"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV K-means Clustering Algorithm, OpenCV kmeans
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-42/
comments: true
toc: true
---

## K-평균 군집화 알고리즘(K-means Clustering Algorithm)

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-42/1.webp" class="lazyload" width="100%" height="100%"/>

`K-평균 군집화 알고리즘(K-means Clustering Algorithm)`은 비지도 학습의 대표적인 알고리즘 중 하나로 `라벨(Label)`이 달려 있지 않은 입력 데이터에 라벨을 달아줍니다.

K-평균 군집화 알고리즘의 방식은 임의의 K개의 `중심점(Centroid)`를 기준으로 최소 거리에 기반한 군집화를 진행합니다.

각각의 데이터는 가장 가까운 중심에 `군집(Cluster)`을 이루며, **같은 중심에 할당된 데이터는 하나의 군집군으로 형성됩니다.**

> [군집화(Clustering) 알아보기](https://076923.github.io/posts/AI-3/)

<br>
<br>

## 메인 코드

{% highlight Python %}

import numpy as np
import cv2

src = cv2.imread("flower.jpg")

data = src.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
retval, bestLabels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = centers.astype(np.uint8)
dst = centers[bestLabels].reshape(src.shape)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

data = src.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)

{% endhighlight %}

K-평균 군집화 알고리즘 함수의 입력 데이터 조건을 맞추기 위해, reshape() 메서드와 astype() 메서드를 활용해 차원과 데이터 형식을 변경합니다.

K-평균 군집화 알고리즘은 `[N, 3]`의 차원과 `float32`의 데이터 형식을 입력 조건으로 사용합니다.

또한, 알고리즘의 종료 기준(TermCriteria)을 설정합니다.

종료 기준은 알고리즘의 **반복 횟수**가 10 회가 되거나, **정확도**가 0.001 이하일 때 종료됩니다.

> [종료 기준 함수 자세히 알아보기](https://076923.github.io/docs/TermCriteria)

<br>

{% highlight Python %}

retval, bestLabels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

{% endhighlight %}

`K-평균 군집화 알고리즘 함수(cv2.kmeans)`를 활용해 입력 데이터에 특정 군집 개수만큼 군집화를 진행합니다.

`retval, bestLabels, centers = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags, centers)`은 `입력 데이터(data)`에서 `K(K)`개의 군집을 설정하고 `라벨(bestLabels)`과 `중심점(centers)`을 찾습니다.

`종료 기준(criteria)`은 군집화의 반복 작업의 조건을 설정하며, `시도(attempts)`는 초기에 다른 라벨을 사용해 반복 실행할 횟수를 설정합니다.

`플래그(flags)`는 초기 중심값 위치에 대한 설정을 진행하며, `결괏값(retval)`은 이미지의 압축률을 반환합니다.

<br>

{% highlight Python %}

centers = centers.astype(np.uint8)
dst = centers[bestLabels].reshape(src.shape)

{% endhighlight %}

`중심값(centers)`은 **flaot32** 형식이므로, **uint8** 형식으로 변환해 Python OpenCV에서 주로 사용하는 형식으로 변경합니다. 

`중심값(centers)`은 **(2, 3)**의 차원을 갖으며, `라벨(bestLabels)`은 **(Width * Height, 1)**의 차원을 갖습니다.

`중심값(centers)`에 할당된 값이 `라벨(bestLabels)`에 매핑할 경우, 시각화를 진행할 수 있습니다.

Numpy의 브로드캐스팅을 적용해 `centers[bestLabels]`를 진행합니다. 이때 차원은 **(Width * Height, 3)**으로 변경됩니다.

입력 이미지와 동일한 차원으로 다시 변경하면 출력 이미지를 확인할 수 있습니다.

<br>
<br>

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-42/2.webp" class="lazyload" width="100%" height="100%"/>
