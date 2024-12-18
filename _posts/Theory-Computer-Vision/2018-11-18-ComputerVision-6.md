---
layout: post
title: "Computer Vision Theory : 전처리 알고리즘"
tagline: "Preprocessing Algorithm"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Preprocessing Algorithm, Computer Vision Grayscale, Computer Vision Threshold, Computer Vision Binary, Computer Vision Zoom In, Computer Vision Zoom Out, Computer Vision Rotation, Computer Vision Transform
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-6/
comments: true
toc: true
---

## 전처리 알고리즘(Preprocessing Algorithm)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-6/1.webp" class="lazyload" width="100%" height="100%"/>

이미지에는 매우 많은 데이터가 존재합니다.

예를 들어, `1,000 × 1,000` 크기의 이미지에는 `1,000,000` 개의 픽셀이 존재합니다.

그리고 각 픽셀마다 `RGB` 값이 할당되어 있으므로, `256 × 256 × 256` 개의 분포가 있어 `16,777,216` 종류의 픽셀 종류가 존재할 수 있습니다.

즉, `(0, 0, 0)`은 검은색 픽셀이 되며, `(255, 0, 0)`은 빨간색 픽셀이 됩니다.

이미지마다 다양한 픽셀의 조합으로 구성되어 있습니다. 이미지 자체를 별도의 처리 없이 분석한다는 것은 매우 어렵고 오랜 시간이 소요됩니다.

그러므로, 이미지 내에서 **불필요한 데이터를 줄이고 유의미한 데이터를 정제**하는 과정이 필요합니다.

이때 전처리 알고리즘을 사용합니다. 전처리 과정이란 이미지를 처리하는 알고리즘에서 효율적으로 활용할 수 있도록 **유의미한 정보로 가공하는 과정**입니다.

객체의 위치를 탐지하는 알고리즘을 구성한다 가정했을 때, 이미지에서 객체를 구성하고 있는 데이터(픽셀)보다 더 많은 데이터(픽셀)들이 존재한다면 객체를 탐지하는 데 방해되는 요소가 됩니다.

이미지를 구성하고 있는 정보들을 가공하여 본격적인 알고리즘이 적용되기 전에 데이터를 간략화하여 알고리즘에 필요하는 데이터만 남겨야합니다.

전처리 알고리즘은 탐지에 **악영향을 주는 부분들을 최소화하는 역할**을 합니다.

<br>
<br>

## 전처리 알고리즘의 종류 

전처리 알고리즘에는 다양한 알고리즘이 존재합니다.

영상 처리에서 전처리 알고리즘은 필수 불가결한 알고리즘이며, 이 알고리즘들을 어떻게 사용하냐에 따라 **정확도**, **정밀도**, **연산 시간** 감소 등의 이점을 얻을 수 었습니다.

전처리 알고리즘에는 크게, `그레이스케일(회색조)`, `이진화`, `확대/축소`, `회전/변환` 등이 있습니다. **이러한 전처리 알고리즘 과정을 처리한 후에, 주요한 알고리즘을 적용합니다.**

<br>
<br>

## 그레이스케일

그레이스케일은 `다중 채널` 이미지를 `단일 채널` 이미지로 변환하여 **데이터의 폭을 줄이는 역할**을 합니다.

색상 이미지는 필연적으로 다중 채널 이미지입니다. 색상으로 구분하지 않고 형상이나 형태를 찾는 것은 흑백 이미지로도 검출이 가능합니다.

그러므로, 다중 채널 이미지를 단일 채널 이미지인 **회색조 이미지**로 변경합니다.

대부분의 알고리즘이 이미지의 소스(source)를 그레이스케일 이미지를 사용하여 처리합니다.

다중 채널은 채널이 3개 또는 4개를 가지고 있습니다. 그레이스케일은 1개의 채널을 가지고 있으므로 데이터의 양이 `1/3` 또는 `1/4`로 줄어들게 되지만, **이미지의 형상에는 크게 훼손을 주지 않습니다.**

이미지가 갖고 있는 형태나 픽셀의 분포에는 크게 영향을 주지는 않지만, 데이터의 양이 크게 줄어들게 됩니다. 그레이스케일 이미지로 주요 알고리즘을 연산한다면, 정확도와 연산량 등에 큰 이점을 얻을 수 있습니다.

<br>
<br>

## 이진화

이진화는 어느 지점을 기준으로 `검은색` 또는 `흰색`으로 변형하기 위해서 사용합니다.

전처리 알고리즘으로 그레이스케일을 적용했다하더라도, 픽셀의 분포 폭이 `256 × 256 × 256`에서 `256`으로 줄어든 효과에 그치게 됩니다.

하지만, 어떤 형상을 검출한다 가정했을 때, 이 분포폭은 너무 많은 범위를 갖고 있다 볼 수 있습니다.

여기서, 더 극단적으로 값의 범위를 `0(검은색)` 또는 `255(흰색)` 등으로 변경한다면 물체의 검출의 정확도를 높이고, 연산량을 감소시킬 수 있습니다.

이진화 처리를 진행할 경우, 이미지를 두 가지의 색상으로 변형하여 일정 임계값 이하는 모두 검은색 또는 흰색으로 변형됩니다. 

이진화를 처리하면, 데이터의 개수가 극단적으로 줄어들게 됩니다. 이 처리를 통하여 검출하고자 하는 **객체를 검출하기 쉬운 상태**로 변형합니다.

이진화 알고리즘은 그레이스케일과 함께 가장 많이 사용되는 전처리 알고리즘 중 하나입니다.

<br>
<br>

## 확대/축소

앞선, `그레이스케일`이나 `이진화`는 픽셀 데이터의 폭을 줄이는 역할을 했습니다.

효율적으로 데이터의 폭을 줄였다 하더라도, 검출이 어렵거나 불가능한 경우가 존재할 수 있습니다.

검출하려는 이미지에서 데이터의 양이 너무 많거나 너무 적을 경우, 이미지를 확대 또는 축소를 통하여 활용하려는 데이터의 범위를 키우거나 줄일 수 있습니다.

**검출하려는 객체가 너무 작을 경우, 이미지를 확대하며 검출하려는 이미지가 너무 클 경우, 축소 과정을 진행합니다.**

이미지 확대나 축소를 통해, 객체의 크기가 너무 크거나 작은 경우 **이미지의 크기를 변경하여 검출하기 쉬운 상태**로 만듭니다.

<br>
<br>

## 회전/변환

앞서서 설명한 전처리 알고리즘을 적용하더라도, 검출하려는 객체가 이미지에서 지정된 형태가 아니라면 검출에 어려움이 있을 수 있습니다.

그러므로, 이미지를 알고리즘이 **검출하기 쉬운 상태로 변형하는 과정**이 필요합니다.

이미지가 틀어져 있는 경우에는 회전 연산을 적용하거나, 이미지 자체에 왜곡이나 객체에 비틀림이 발생한다면, 캘리브레이션이나 이미지의 왜곡을 변형하는 과정을 진행할 수 있습니다.

이를 통해, 주요 알고리즘이 동일하거나 비슷한 패턴의 이미지를 처리할 수 있게되어 **알고리즘의 복잡도가 줄어들게 됩니다.**

<br>
<br>

* Writer by : 윤대희
