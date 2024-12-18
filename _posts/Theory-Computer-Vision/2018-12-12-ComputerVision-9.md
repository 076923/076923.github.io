---
layout: post
title: "Computer Vision Theory : 특징 검출"
tagline: "Feature Detection"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Feature Detection
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-9/
comments: true
toc: true
---

## 특징 검출(Feature Detection)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-9/1.webp" class="lazyload" width="100%" height="100%"/>

`특징 검출(Feature Detection)`은 이미지 내의 주요한 `특징점`을 검출하는 방법입니다.

해당 특징점이 존재하는 위치를 알려주거나 해당 특징점을 부각시킵니다.

픽셀의 **색상 강도**, **연속성**, **변화량**, **의존성**, **유사성**, **임계점** 등을 사용하여 특징을 파악합니다.

특징 검출을 사용하여 다양한 패턴의 객체를 검출할 수 있습니다.

대표적으로 `가장자리(Edge)`. `윤곽(Contours)`, `모서리(Corner)`, `선(Line)`, `원(Circle)` 등이 있습니다.

<br>
<br>

## 가장자리(Edge)

`가장자리(Edge)` 검출은 이미지 내의 가장자리 검출을 위한 알고리즘입니다.

픽셀의 그라디언트의 **상위 임계값**과 **하위 임계값**을 사용하여 가장자리를 검출합니다.

픽셀의 **연속성**, **연결성** 등이 유효해야합니다. 가장자리의 일부로 간주되지 않는 픽셀은 제거되어 가장자리만 남게됩니다.

<br>
<br>

## 윤곽(Contours)

`윤곽(Contours)` 검출은 이미지 내의 윤곽 검출을 위한 알고리즘입니다.

**동일한 색상이나 비슷한 강도**를 가진 **연속한 픽셀**을 묶습니다.

윤곽 검출을 통하여 **중심점**, **면적**, **경계선**, **블록 껍질**, **피팅** 등을 적용할 수 있습니다.

<br>
<br>

## 모서리(Corner)

`모서리(Corner)` 검출은 그라디언트에서 **유사성**을 검출합니다.

**픽셀 강도의 차이**를 기준으로 모서리 점을 검출합니다. 이 결과로 **가장자리**, **평면**, **모서리**를 구분합니다.

때에 따라 모서리 강도가 강한 모서리 점을 검출할 수 도 있습니다.

<br>
<br>

## 선(line)

`선 (line)` 검출은 이미지의 모든 점에 대한 **교차점을 추적**합니다.

교차점의 수가 임계값보다 높을 경우, 매개 변수가 있는 행으로 간주합니다. 즉, 교차점의 교차 수를 찾아 선을 검출합니다.

교차 횟수가 많을 수록 선이 더 많은 픽셀을 가지게 됩니다.

<br>
<br>

## 원(Circle)

`원(Circle)` 검출은 이미지에서 **방사형 대칭성**이 높은 객체를 효과적으로 검출합니다.

특징점을 파라미터 공간으로 매핑하여 검출합니다. **가장자리**에 그라디언트 방법을 이용하여 원의 중심점 (a,b)에 대한 **2D Histogram**으로 선정합니다.

<br>
<br>

* Writer by : 윤대희
