---
layout: post
title: "Computer Vision Theory : 움직임 추적"
tagline: "Motion Tracking"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Motion Tracking
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-12/
comments: true
toc: true
---

## 움직임 추적(Motion Tracking)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-12/1.webp" class="lazyload" width="100%" height="100%"/>

`움직임 추적(Motion Tracking)`은 비디오의 연속된 이미지(프레임)에서 특정 객체를 찾는 것을 의미합니다.

**배경(Background)**에서 객체의 **움직임**, **누적 경로**, **예상 경로**, **속도**, **속력** 등을 확인할 수 있습니다.

주요한 추적 방식은 움직이는 객체의 특징점을 찾는 `Point Tracking`, 일정 영역 내부의 있는 움직임을 찾는 `Kernel based Tracking`, 복잡한 형태를 단순화(실루엣) 시켜 움직임을 찾는 `Silhouette based Tracking`이 있습니다.

대표적으로 `광학 흐름(Optical Flow)`, `칼만 필터링(Kalman Filtering)`, `에고 모션(Ego Motion)` 등이 있습니다.

<br>
<br>

## 광학 흐름(Optical Flow)

`광학 흐름(Optical Flow)`은 **카메라와 피사체의 상대 운동에 의하여 피사체의 운동에 대한 패턴**을 의미합니다.

밝기 변화가 거의 없고 일정 블록 내의 모든 픽셀이 모두 같은 운동을 한다 가정하여 움직임을 추정합니다.

또 다른 방식으로는 특정점만을 사용하여 광학흐름을 찾거나 일정 블록 내의 움직임을 판단하여 감지합니다.

주로 **동작 감지**, **물체 추적**, **구조 분석** 등에 이용합니다.

<br>
<br>

## 칼만 필터링(Kalman Filtering)

`칼만 필터링(Kalman Filtering)`은 노이즈가 포함되어 있는 선형 역학계의 상태를 추적하는 **재귀 필터**입니다.

**이전 프레임의 움직임 정보**를 기반으로 움직이는 물체의 위치를 ​​예측하는데 사용되는 신호 처리 알고리즘입니다.

추적을 위한 효과적인 계산 알고리즘이며 **노이즈 측정에 관한 피드백**을 제공합니다. 

<br>
<br>

## 에고 모션(Ego Motion)

`에고 모션(Ego Motion)`은 카메라가 이미지를 사용하여 **카메라의 모션**을 결정합니다.

**깊이 지도(Depth Map)**와 **시차 지도(Parallax Map)**을 생성하여 움직임을 추정합니다.

대표적으로 **시각적 주행 측정법(Visual Odometry)**으로 사용할 수 있습니다.

카메라의 이미지를 분석하여 위치와 방향을 확인하거나 결정할 수 있습니다. 액추에이터(Actuator)의 움직임 데이터를 활용할 수 있습니다.

<br>
<br>

* Writer by : 윤대희
