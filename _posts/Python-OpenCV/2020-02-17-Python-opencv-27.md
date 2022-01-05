---
layout: post
title: "Python OpenCV 강좌 : 제 27강 - 모폴로지 연산"
tagline: "Python OpenCV Morphological Calculate"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV morphologyEx, OpenCV MORPH_DILATE, OpenCV MORPH_ERODE, OpenCV MORPH_OPEN, OpenCV MORPH_CLOSE, OpenCV MORPH_GRADIENT, OpenCV MORPH_TOPHAT, OpenCV MORPH_BLACKHAT, OpenCV MORPH_HITMISS
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-27/
comments: true
toc: true
---

## 모폴로지 연산(Morphological Calculate)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-27/1.webp){:class="lazyload" width="100%" height="100%"}

`모폴로지 연산(Perspective Calculate)`은 모폴로지 변환의 `팽창(dilation)`과 `침식(erosion)`을 기본 연산으로 사용해 고급 형태학을 적용하는 변환 연산입니다.

입력 이미지가 이진화된 이미지라면 팽창과 침식 연산으로도 우수한 결과를 얻을 수 있습니다.

하지만, 그레이스케일이나 다중 채널 이미지를 사용하는 경우 **더 복잡한 연산을 필요**로 합니다.

이때 모폴로지 연산을 활용해 우수한 결과를 얻을 수 있습니다.

<br>
<br>

## 열림(Opening)

<h3> $$ open = dilate(erode(src)) $$ </h3>

<br>

`팽창 연산자`와 `침식 연산자`의 조합이며, **침식 연산을 적용한 다음, 팽창 연산을 적용합니다.**

열림 연산을 적용하면 침식 연산으로 인해 밝은 영역이 줄어들고 어두운 영역이 늘어납니다.

줄어든 영역을 다시 복구하기 위해 팽창 연산을 적용하면 반대로 어두운 영역이 줄어들고 밝은 영역이 늘어납니다.

이로 인해 `스펙클(speckle)`이 사라지면서 발생한 **객체의 크기 감소를 원래대로 복구**할 수 있습니다.

<br>
<br>

## 닫힘(Closing)

<h3> $$ close = erode(dilate(src)) $$ </h3>

<br>

`팽창 연산자`와 `침식 연산자`의 조합이며, 열림과 반대로 **팽창 연산을 적용한 다음, 침식 연산을 적용합니다.**

닫힘 연산은 팽창 연산으로 인해 어두운 영역이 줄어들고 밝은 영역이 늘어납니다.

늘어난 영역을 다시 복구하기 위해 침식 연산을 적용하면 밝은 영역이 줄어들고 어두운 영역이 늘어납니다.

그로 인해 객체 내부의 `홀(holes)`이 사라지면서 발생한 **크기 증가를 원래대로 복구**할 수 있다.

<br>
<br>

## 그레이디언트(Gradient)

<h3> $$ gradient = dilate(src) - erode(src) $$ </h3>

<br>

`팽창 연산자`와 `침식 연산자`의 조합이며, 열림 연산이나 닫힘 연산과 달리 **입력 이미지에 각각 팽창 연산과 침식 연산을 적용하고 감산을 진행합니다.**

입력 이미지와 비교했을 때 팽창 연산은 밝은 영역이 더 크며, 반대로 침식 연산은 밝은 영역이 더 작습니다.

각각의 결과를 감산한다면 입력 이미지에 객체의 가장자리가 반환됩니다.

그레이디언트는 밝은 영역의 가장자리를 분리하며 **그레이스케일 이미지가 가장 급격하게 변하는 곳에서 가장 높은 결과를 반환**합니다.

<br>
<br>

## 탑햇(TopHat)

<h3> $$ tophat = src - open(src) $$ </h3>

<br>

`입력 이미지(src)`와 `열림(Opening)`의 조합이며, 그레이디언트 연산과 비슷하게 입력 이미지에 열림 연산을 적용한 이미지를 감산합니다.

열림 연산이 적용된 이미지는 스펙클이 사라지고 객체의 크기가 보존된 결과입니다.

이 결과를 입력 이미지에서 감산한다면 밝은 영역이 분리되어 **사라졌던 스펙클이나 작은 부분들이 표시**됩니다.

즉, 입력 이미지의 **객체들이 제외되고 국소적으로 밝았던 부분들이 분리**됩니다.

- Tip : 탑햇 연산은 **열림 연산에서 사라질 요소들을 표시합니다.**

<br>
<br>

## 블랙햇(BlackHat)

<h3> $$ blackhat = close(src) - src $$ </h3>

<br>

`입력 이미지(src)`와 `닫힘(Closing)`의 조합이며, 탑햇 연산과 비슷하게 닫힘 연산을 적용한 이미지에 입력 이미지를 감산합니다.

닫힘 연산이 적용된 이미지는 객체 내부의 홀이 사라지고 객체의 크기가 보존된 결과입니다.

이 결과에 입력 이미지를 감산한다면 **어두운 영역이 채워져 사라졌던 홀 등이 표시**됩니다.

즉, 입력 이미지의 **객체들이 제외되고 국소적으로 어두웠던 홀들이 분리됩니다.**

- Tip : 블랙햇 연산은 **닫힘 연산에서 사라질 요소들을 표시합니다.**

<br>
<br>

## 히트미스(HitMiss)

`히트미스(HitMiss)` 연산은 앞의 연산자와 다른 형태입니다.

히트미스 연산은 단일 채널 이미지에서 활용하며, 주로 이진화 이미지에 적용합니다.

히트미스 연산은 **이미지의 전경이나 배경 픽셀의 특정 패턴을 찾는 데 사용하는 이진 형태학**으로서 `구조 요소의 형태에 큰 영향`을 받습니다.

히트미스 연산의 커널은 기존 컨벌루션 커널과 다른 역할을 합니다.

내부 요소의 값은 0 또는 1의 값만 의미가 있습니다.

커널 내부의 0은 해당 픽셀을 고려하지 않는다는 의미이며, 1은 해당 요소를 유지하겠다는 의미입니다.

이 특성 덕분에 히트미스 연산을 `모서리(Corner)`를 검출하는 데 활용하기도 합니다.

- Tip : 제한 조건 - `8-bit unsigned integers, 1-Channel`

<br>
<br>

## 메인 코드

{% highlight Python %}

import numpy as np
import cv2

src = cv2.imread('office.jpg')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
dst = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=9)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

dst = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=9)

{% endhighlight %}

생성된 구조 요소(kernel)를 활용해 모폴로지 변환을 적용합니다.

`모폴로지 함수(cv2.morphologyEx)`로 모폴로지 연산을 진행합니다.

`cv2.morphologyEx(원본 배열, 연산 방법, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)`로 모폴로지 연산을 진행합니다.

`연산 방법`에 따라, 모폴로지 연산 결과가 달라집니다. 예제의 연산 방법은 `열림 연산`입니다.

연산 방법에는 기존 `팽창 함수(cv2.dilate)`와 `침식 함수(cv2.erode)`도 포함돼 있습니다.

<br>
<br>

## 추가 정보

### 연산 방법 종류

|     속성    |      의미     |
|:-----------:|:-------------:|
| cv2.MORPH_DILATE | 팽창 연산 |
| cv2.MORPH_ERODE | 침식 연산 |
| cv2.MORPH_OPEN | 열림 연산 |
| cv2.MORPH_CLOSE | 닫힘 연산 |
| cv2.MORPH_GRADIENT | 그레이디언트 연산 |
| cv2.MORPH_TOPHAT | 탑햇 연산 |
| cv2.MORPH_BLACKHAT | 블랙햇 연산 |
| cv2.MORPH_HITMISS | 히트미스 연산 |

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-27/2.webp){:class="lazyload" width="100%" height="100%"}
