---
layout: post
title: "Computer Vision Theory : 이미지 연산"
tagline: "Image Calculation"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Image Calculation
ref: ComputerVision
category: posts
permalink: /posts/ComputerVision-8/
comments: true
---

## 이미지 연산(Image Calculation) ##
----------

![1]({{ site.images }}/assets/images/ComputerVision/ch8/1.jpg)
`이미지 연산(Image Calculation)`란 각 픽셀마다 특정 함수나 수식을 적용하여 변형하는 것을 의미합니다. 이미지에 일괄적으로 값을 **더하거나 빼는 연산**을 예로 들 수 있습니다.  이외에도 색상을 흑백으로 변환하는 것도 연산으로 볼 수 있습니다. 색상 이미지를 흑백 이미지 변경할 때 다음의 수식이 적용됩니다. `Y = 0.299 * R + 0.587 * G + 0.114 * B`을 통하여 그레이스케일로 변환할 수 있습니다. 이렇듯 이미지에 대해여 일괄적으로 적용하는 경우 이미지 연산이 적용되었다 볼 수 있습니다. 

이미지 연산을 통하여 **특정 범위의 픽셀을 제거하거나 변환하는 등의 작업을 진행하거나 모든 픽셀에 대하여 동일한 처리**를 진행할 수 있습니다. 해당 결과를 통하여 **불필요한 영역을 제거하거나 필요한 영역을 두드러지게 할 수 있습니다.**

대표적으로 `비트 연산(Bitwise)`, `연산(Calculation)`, `색상 변경(CvtColor)` 등이 있습니다.

<br>
<br>

## 비트 연산(Bitwise) ##

`비트 연산(Bitwise)`는 이미지에 대하여 `AND`, `OR`, `XOR`, `NOT`의 비트 연산을 적용할 수 있습니다. 두 이미지에서 동일한 픽셀을 지니는 값만 표시하거나 표시하지 않거나 동일하지 않는 픽셀만을 표시하거나 색상 반전 등을 적용할 수 있습니다.

<br>
<br>

## 연산(Calculation) ##

`연산(Calculation)`은 이미지에 대하여 `ADD`, `SUB`, `MUL`, `DIV`, `MAX`, `MIN`, `ABSDIFF` 등의 연산을 적용할 수 있습니다. 두 이미지의 픽셀의 값을 더하기, 빼기, 곱하기, 나누기, 최대값, 최소값, 절대값 감산 등을 적용하게 됩니다. 이를 통하여 이미지위에 다른 이미지를 겹치거나 특정 픽셀을 제거하는 등의 기능을 수행할 수 있습니다.

<br>
<br>

## 색상 변경(CvtColor) ##

`색상 변경(CvtColor)`은 전체 이미지에 대하여 동일한 함수를 적용하여 해당 이미지가 지니는 의미를 변환하는 역할을 합니다. `RGB` 속성을 지닌 이미지를 `GRAY` 이미지로 변경하거나 `HSV`, `YCrCb` 등의 **다양한 색상 공간으로 변환**할 수 있습니다. `RGB` 속성 이외의 다른 속성 색상 공간으로 변환하여 처리하므로 복잡한 알고리즘을 사용하지 않아도 원하는 결과를 얻어낼 수 있도록 하는 전처리 과정으로도 볼 수 있습니다.

<br>
<br>

* Writer by : 윤대희