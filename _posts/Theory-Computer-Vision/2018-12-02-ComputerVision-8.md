---
layout: post
title: "Computer Vision Theory : 이미지 연산"
tagline: "Image Calculation"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Image Calculation
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-8/
comments: true
toc: true
---

## 이미지 연산(Image Calculation)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-8/1.webp" class="lazyload" width="100%" height="100%"/>

`이미지 연산(Image Calculation)`란 각 픽셀마다 특정 함수나 수식을 적용하여 변형하는 것을 의미합니다.

이미지에 일괄적으로 값을 **더하거나 빼는 연산**을 예로 들 수 있습니다. 이외에도 색상을 흑백으로 변환하는 것도 연산으로 볼 수 있습니다.

예를 들어, RGB의 색상 이미지를 흑백 이미지 변경할 때 다음의 수식을 적용합니다.

`Y = 0.299 * R + 0.587 * G + 0.114 * B`을 사용해, RGB 색상을 하나의 채널만 갖는 그레이스케일로 변환할 수 있습니다.

이렇듯 이미지에 대해여 일괄적으로 적용하는 경우엔 이미지 연산이 적용되었다 볼 수 있습니다. 

이미지 연산을 통하여 **특정 범위의 픽셀을 제거하거나 변환하는 등의 작업을 진행하거나 모든 픽셀에 대하여 동일한 처리**를 진행할 수 있습니다.

해당 결과를 통하여 **불필요한 영역을 제거하거나 필요한 영역을 두드러지게 할 수 있습니다.**

대표적으로 `비트 연산(Bitwise)`, `연산(Calculation)`, `색상 공간 변경(Convert Color Space)` 등이 있습니다.

<br>
<br>

## 비트 연산(Bitwise)

`비트 연산(Bitwise)`는 이미지에 대하여 `AND`, `OR`, `XOR`, `NOT`의 비트 연산을 적용할 수 있습니다.

두 이미지의 픽셀에 비트 연산을 적용하여 논리곱, 논리합, 배타적 논리합, 부정을 연산합니다.

픽셀값은 10진수의 형태로 0에서 255 사이의 값을 가지고 있습니다.

이 10진수의 픽셀값을 2진수로 변환한 다음, 각 자릿수에 대해 비트 연산을 진행합니다.

만약, 198과 255의 픽셀을  배타적 논리합 비트 연산을 진행한다면 다음과 같습니다.

198은 `1100 0110(2)`이 되며, 255는 `1111 1111(2)`이 됩니다.

XOR 연산은 비트 값이 같으면 0, 다르다면 1이 됩니다. 각 자리수 마다 값을 비교한다면 0011 1001이 됩니다.

이 값을 10진수로 변경한다면, 57이 됩니다. 그러므로, 최종 반환값은 57의 값으로 할당됩니다.

<br>
<br>

## 연산(Calculation)

`연산(Calculation)`은 이미지에 대하여 `더하기(ADD)`, `빼기(SUB)`, `곱하기(MUL)`, `나누기(DIV)`, `최댓값(MAX)`, `최솟값(MIN)`, `절댓값 감산(ABSDIFF)` 등의 연산을 적용할 수 있습니다.

두 개 이상의 이미지를 연산을 하거나, 하나의 이미지에 특정값을 더하거나 곱하는 등의 연산을 할 수 있습니다.

위와 같은 연산을 통해 이미지 위에 다른 이미지를 겹치거나, 특정 픽셀 값을 변경하는 등의 기능을 수행할 수 있습니다. 이러한 연산으로 전체적인 알고리즘의 연산량을 줄이고 정확도를 높일 수 있습니다.

<br>
<br>

## 색상 공간 변경(Convert Color Space)

`색상 변경(Convert Color Space)`은 전체 이미지에 대하여 동일한 함수를 적용하여 해당 이미지가 지니는 의미를 변환하는 역할을 합니다.

`RGB` 속성을 지닌 이미지를 `GRAY` 이미지로 변경하거나 `HSV`, `YCrCb` 등의 **다양한 색상 공간으로 변환**할 수 있습니다.

`RGB` 속성 이외의 다른 속성 색상 공간으로 변환하여 처리하므로 복잡한 알고리즘을 사용하지 않아도 원하는 결과를 얻어낼 수 있도록 하는 전처리 과정으로도 볼 수 있습니다.

빨간색이나 파란색 등의 특정 색상을 추출하려고 한다 가정한다면, `RGB` 색상 공간에서 추출하는 것 보다 `HSV` 등의 색상 공간에서 색상을 추출하면, 더 효율적이고 빠르게 색상을 검출할 수 있습니다.

<br>
<br>

* Writer by : 윤대희
