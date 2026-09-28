---
layout: post
title: "Computer Vision Theory : 특징 검출"
tagline: "Feature Detection"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Feature Detection, Canny, Sobel, Harris Corner, Hough Transform, Feature Descriptor, SIFT, ORB
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-9/
comments: true
toc: true
---

## 특징 검출(Feature Detection)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-9/1.webp" class="lazyload" width="100%" height="100%"/>

`특징 검출(Feature Detection)`은 이미지 내의 주요한 `특징점`을 검출하는 방법입니다. 해당 특징점이 존재하는 위치를 알려주거나 해당 특징점을 부각시킵니다. 픽셀의 **색상 강도**, **연속성**, **변화량**, **의존성**, **유사성**, **임계점** 등을 사용하여 특징을 파악합니다.

특징 검출을 사용하여 다양한 패턴의 객체를 검출할 수 있습니다. 대표적으로 `가장자리(Edge)`, `윤곽(Contours)`, `모서리(Corner)`, `선(Line)`, `원(Circle)` 등이 있습니다.

<br>
<br>

## 가장자리(Edge)

`가장자리(Edge)` 검출은 이미지 내의 가장자리 검출을 위한 알고리즘입니다. 픽셀의 그라디언트의 **상위 임계값**과 **하위 임계값**을 사용하여 가장자리를 검출합니다. 픽셀의 **연속성**과 **연결성**이 유효해야 합니다. 가장자리의 일부로 간주되지 않는 픽셀은 제거되어 가장자리만 남게됩니다.

<br>
<br>

## 윤곽(Contours)

`윤곽(Contours)` 검출은 이미지 내의 윤곽 검출을 위한 알고리즘입니다. **동일한 색상이나 비슷한 강도**를 가진 **연속한 픽셀**을 묶습니다.

윤곽 검출을 통하여 **중심점**, **면적**, **경계선**, **블록 껍질**, **피팅** 등을 적용할 수 있습니다.

<br>
<br>

## 모서리(Corner)

`모서리(Corner)` 검출은 그라디언트에서 **유사성**을 검출합니다. **픽셀 강도의 차이**를 기준으로 모서리 점을 검출합니다. 이 결과로 **가장자리**, **평면**, **모서리**를 구분합니다.

때에 따라 모서리 강도가 강한 모서리 점을 검출할 수 도 있습니다.

<br>
<br>

## 선(Line)

`선(Line)` 검출은 이미지의 모든 점에 대한 **교차점을 추적**합니다. 교차점의 수가 임계값보다 높을 경우, 매개 변수가 있는 행으로 간주합니다. 즉, 교차점의 교차 수를 찾아 선을 검출합니다. 교차 횟수가 많을 수록 선이 더 많은 픽셀을 가지게 됩니다.

<br>
<br>

## 원(Circle)

`원(Circle)` 검출은 이미지에서 **방사형 대칭성**이 높은 객체를 효과적으로 검출합니다. 특징점을 파라미터 공간으로 매핑하여 검출합니다. **가장자리**에 그라디언트 방법을 이용하여 원의 중심점 (a,b)에 대한 **2D Histogram**으로 선정합니다.

## 특징 기술자(Feature Descriptor)

지금까지의 검출 방법은 특징이 **어디에 있는지**를 찾습니다. 그러나 서로 다른 두 이미지에서 같은 지점을 찾아 연결하려면, 위치뿐 아니라 그 지점이 **어떻게 생겼는지**를 숫자로 표현해야 합니다. 이 표현을 `특징 기술자(Feature Descriptor)`라 합니다.

특징 기술자는 특징점 주변의 밝기 변화나 방향 정보를 벡터 형태로 기록합니다. 두 이미지에서 각각 기술자를 계산한 뒤 벡터 간의 거리를 비교하면, 같은 지점끼리 짝지을 수 있으며 이를 `특징점 매칭(Feature Matching)`이라 합니다.

좋은 기술자는 이미지가 회전하거나 크기가 달라져도 같은 값을 유지해야 하며, 이러한 성질을 각각 `회전 불변성(Rotation Invariance)`과 `크기 불변성(Scale Invariance)`이라 합니다. 이 성질 덕분에 특징점 매칭은 파노라마 이미지 생성이나 객체 추적 등에 활용됩니다.

- Tip : 대표적인 기술자로 `SIFT`, `SURF`, `ORB`, `BRISK`, `AKAZE` 등이 있습니다. 이 중 `ORB`는 연산이 빠르고 라이선스 제약이 없어 실시간 처리에 널리 사용됩니다.

<br>
<br>

## 대표 알고리즘 정리

각 검출 방법에는 널리 사용되는 구현 알고리즘이 있습니다. 이론적인 개념을 실제 코드로 옮길 때 참고할 수 있도록 정리하면 다음과 같습니다.

| 검출 대상 | 대표 알고리즘 | 특징 |
|---|---|---|
| 가장자리 | 소벨(Sobel), 라플라시안(Laplacian) | 미분으로 밝기 변화량을 계산 |
| 가장자리 | 캐니(Canny) | 두 개의 임계값으로 가장자리를 선별 |
| 윤곽 | 윤곽선 검출(findContours) | 같은 값을 갖는 픽셀을 경계로 묶음 |
| 모서리 | 해리스(Harris), 시-토마시(Shi-Tomasi) | 두 방향 모두 변화가 큰 지점을 검출 |
| 선 | 허프 선 변환(Hough Line Transform) | 파라미터 공간의 교차점으로 직선 검출 |
| 원 | 허프 원 변환(Hough Circle Transform) | 중심점 후보의 누적값으로 원 검출 |

가장자리 검출에서 `캐니(Canny)`가 널리 쓰이는 이유는, 단순히 미분값이 큰 지점을 모두 남기지 않고 **두 개의 임계값을 사용해 가장자리를 선별**하기 때문입니다. 상위 임계값을 넘는 픽셀은 확실한 가장자리로 두고, 하위 임계값과 상위 임계값 사이의 픽셀은 확실한 가장자리와 연결된 경우에만 남깁니다.

이러한 방식을 `이력 임계값(Hysteresis Thresholding)`이라 하며, 노이즈로 인한 짧은 가장자리는 제거하면서 실제 경계는 끊기지 않게 유지할 수 있습니다.

<br>
<br>

<br>
<br>

* Writer by : 윤대희
