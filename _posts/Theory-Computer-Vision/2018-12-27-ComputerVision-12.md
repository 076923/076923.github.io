---
layout: post
title: "Computer Vision Theory : 움직임 추적"
tagline: "Motion Tracking"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Motion Tracking, Optical Flow, Lucas-Kanade, Farneback, MeanShift, CamShift, Kalman Filtering, Multi-Object Tracking
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-12/
comments: true
toc: true
---

## 움직임 추적(Motion Tracking)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-12/1.webp" class="lazyload" width="100%" height="100%"/>

`움직임 추적(Motion Tracking)`은 비디오의 연속된 이미지(프레임)에서 특정 객체를 찾는 것을 의미합니다. **배경(Background)**에서 객체의 **움직임**, **누적 경로**, **예상 경로**, **속도**, **속력** 등을 확인할 수 있습니다.

주요한 추적 방식은 움직이는 객체의 특징점을 찾는 `Point Tracking`, 일정 영역 내부의 있는 움직임을 찾는 `Kernel based Tracking`, 복잡한 형태를 단순화(실루엣) 시켜 움직임을 찾는 `Silhouette based Tracking`이 있습니다. 대표적으로 `광학 흐름(Optical Flow)`, `칼만 필터링(Kalman Filtering)`, `에고 모션(Ego Motion)` 등이 있습니다.

<br>
<br>

## 광학 흐름(Optical Flow)

`광학 흐름(Optical Flow)`은 **카메라와 피사체의 상대 운동에 의하여 피사체의 운동에 대한 패턴**을 의미합니다. 밝기 변화가 거의 없고 일정 블록 내의 모든 픽셀이 모두 같은 운동을 한다 가정하여 움직임을 추정합니다. 또 다른 방식으로는 특징점만을 사용하여 광학 흐름을 찾거나 일정 블록 내의 움직임을 판단하여 감지합니다.

주로 **동작 감지**, **물체 추적**, **구조 분석** 등에 이용합니다.

<br>
<br>

## 칼만 필터링(Kalman Filtering)

`칼만 필터링(Kalman Filtering)`은 노이즈가 포함되어 있는 선형 역학계의 상태를 추적하는 **재귀 필터**입니다. **이전 프레임의 움직임 정보**를 기반으로 움직이는 물체의 위치를 예측하는 데 사용되는 신호 처리 알고리즘입니다.

추적을 위한 효과적인 계산 알고리즘이며 **노이즈 측정에 관한 피드백**을 제공합니다.

<br>
<br>

## 에고 모션(Ego Motion)

`에고 모션(Ego Motion)`은 카메라가 이미지를 사용하여 **카메라의 모션**을 결정합니다. **깊이 지도(Depth Map)**와 **시차 지도(Parallax Map)**를 생성하여 움직임을 추정합니다.

대표적으로 **시각적 주행 측정법(Visual Odometry)**으로 사용할 수 있습니다. 카메라의 이미지를 분석하여 위치와 방향을 확인하거나 결정할 수 있습니다. 액추에이터(Actuator)의 움직임 데이터를 활용할 수 있습니다.

## 희소 광학 흐름과 밀집 광학 흐름

`광학 흐름(Optical Flow)`은 모든 픽셀을 계산하는지에 따라 두 가지로 나뉩니다. `희소 광학 흐름(Sparse Optical Flow)`은 모서리와 같은 **특징점만 골라** 이동을 추정하며, `루카스-카나데(Lucas-Kanade)` 방법이 대표적입니다.

`밀집 광학 흐름(Dense Optical Flow)`은 **모든 픽셀**에 대해 이동 벡터를 계산하며, `파네백(Farnebäck)` 방법이 대표적입니다. 화면 전체의 움직임을 파악할 수 있어 동작 분석이나 배경 분리에 유리하지만, 연산량이 훨씬 많아 실시간 처리에는 부담이 됩니다.

추적할 대상이 명확한 몇 개의 지점이라면 희소 방식이, 화면 전체의 흐름을 봐야 한다면 밀집 방식이 적합합니다. 두 방식 모두 **밝기가 급격히 변하지 않는다**는 가정 위에서 동작하므로, 조명이 갑자기 바뀌는 환경에서는 추정이 흐트러질 수 있습니다.

<br>
<br>

## 객체 추적기(Object Tracker)

광학 흐름이 픽셀이나 특징점의 이동을 추정한다면, `객체 추적기(Object Tracker)`는 첫 프레임에서 지정한 **영역 자체를 따라가는** 방법입니다. 색상 분포의 중심을 반복적으로 이동시키는 `민 시프트(MeanShift)`와, 여기에 크기와 회전 대응을 더한 `캠 시프트(CamShift)`가 기본적인 형태입니다.

OpenCV는 이 외에도 `KCF`, `CSRT`, `MOSSE` 등의 추적기를 제공합니다. `MOSSE`는 가장 빠르지만 정확도가 낮고, `CSRT`는 가장 정확하지만 느리며, `KCF`는 그 중간에 해당하므로 요구 성능에 따라 선택합니다.

한편 추적 대상이 여러 개이거나 화면에서 사라졌다 다시 나타나는 경우에는, 매 프레임 객체를 검출한 뒤 **프레임 사이에서 같은 객체끼리 연결**하는 방식을 사용합니다. 이를 `다중 객체 추적(Multi-Object Tracking)`이라 하며, `SORT`나 `DeepSORT` 등이 대표적입니다.

- Tip : 추적은 검출보다 연산이 가볍습니다. 그러므로 실시간 처리에서는 **몇 프레임마다 한 번만 검출을 수행하고 그 사이는 추적으로 채우는** 방식을 흔히 사용합니다.

<br>
<br>

<br>
<br>

* Writer by : 윤대희
