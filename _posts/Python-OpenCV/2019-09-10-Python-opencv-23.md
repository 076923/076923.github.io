---
layout: post
title: "Python OpenCV 강좌 : 제 23강 - 코너 검출"
tagline: "Python OpenCV Good Features To Track"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Corner, OpenCV goodFeaturesToTrack
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-23/
comments: true
toc: true
---

## 코너 검출(Good Features To Track)

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-23/1.webp" class="lazyload" width="100%" height="100%"/>

영상이나 이미지에서 `코너`를 검출하는 알고리즘입니다.

코너 검출 알고리즘은 정확하게는 `트래킹(Tracking)` 하기 `좋은 지점(특징)`을 **코너**라 부릅니다.

꼭짓점은 **트래킹하기 좋은 지점**이 되어 다각형이나 객체의 꼭짓점을 검출하는 데 사용합니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/coffee.jpg")
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)

for i in corners:
    cv2.circle(dst, tuple(i[0]), 3, (0, 0, 255), 2)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)

{% endhighlight %}

`cv2.goodFeaturesToTrack()`를 활용해 윤곽선들의 이미지에서 **코너**를 검출합니다.

`cv2.goodFeaturesToTrack(입력 이미지, 코너 최댓값, 코너 품질, 최소 거리, 마스크, 블록 크기, 해리스 코너 검출기 유/무, 해리스 코너 계수)`을 의미합니다.

`입력 이미지`는 8비트 또는 32비트의 **단일 채널 이미지**를 사용합니다.

`코너 최댓값`은 검출할 최대 코너의 수를 제한합니다. 코너 최댓값보다 **낮은 개수만 반환**합니다.

`코너 품질`은 반환할 코너의 최소 품질을 설정합니다. 코너 품질은 **0.0 ~ 1.0 사이의 값으로 할당**할 수 있으며, 일반적으로 **0.01 ~ 0.10 사이의 값을 사용합니다.**

`최소 거리`는 검출된 코너들의 최소 근접 거리를 나타내며, 설정된 **최소 거리 이상의 값만 검출합니다.**

`마스크`는 **입력 이미지**와 같은 차원을 사용하며, 마스크 요솟값이 **0인 곳은 코너로 계산하지 않습니다.**

`블록 크기`는 코너를 계산할 때, 고려하는 **코너 주변 영역의 크기**를 의미합니다.

`해리스 코너 검출기 유/무`는 해리스 코너 검출 방법 **사용 여부**를 설정합니다.

`해리스 코너 계수`는 해리스 알고리즘을 사용할 때 할당하며 **해리스 대각합의 감도 계수**를 의미합니다.

- Tip : **코너 품질**에서 가장 좋은 코너의 강도가 **1000**이고, 코너 품질이 **0.01**이라면 **10 이하**의 코너 강도를 갖는 코너들은 검출하지 않습니다. 
- Tip : **최소 거리**의 값이 5일 경우, 거리가 5 이하인 코너점은 검출하지 않습니다.

<br>

{% highlight Python %}

for i in corners:
    cv2.circle(dst, tuple(i[0]), 3, (0, 0, 255), 2)

{% endhighlight %}

`코너 검출` 함수를 통해 `corners`가 반환되며, 이 배열안에 코너들의 좌표가 저장돼 있습니다.

반복문을 활용해 `dst`에 빨간색 원으로 지점을 표시합니다.

<br>
<br>

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-23/2.webp" class="lazyload" width="100%" height="100%"/>
