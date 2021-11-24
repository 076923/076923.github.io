---
layout: post
title: "Computer Vision Theory : Digital Image Processing이란?"
tagline: "Digital Image Processing이란?"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Digital Image Processing, Jet Propulsion Laboratory, Computer Vision RGB, Computer Vision xy
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-2/
comments: true
toc: true
---

## Digital Image Processing 이란?

![1]({{ site.images }}/assets/posts/Theory/ComputerVision/lecture-2/1.jpg)

`Digital Image Processing`이란 디지털 이미지를 신호 처리하는 분야로 이미지로부터 **유의미한 정보를 얻기 위하여 사용되는 일련의 알고리즘**을 의미합니다.

**변환, 분류, 탐지, 인식, 검출, 분석, 왜곡, 수정, 향상, 복원, 압축, 필터링** 등의 다양한 처리를 할 수 있습니다.

위에서 언급된 알고리즘을 처리하기 위해서는 이미지나 영상을 컴퓨터가 처리할 수 있는 데이터로 변경해야 합니다.

디지털 이미지는 `2차원 배열` 형태로 표시할 수 있습니다. 2차원 배열의 행과 열은 픽셀의 **좌표(x, y)**가 되며, 특정 행과 열에 포함된 값은 **픽셀의 값(r, g, b)**가 됩니다.

그러므로, 2차원 평면 공간에서 이미지의 `픽셀의 좌표`와 `해당 값`을 알 수 있습니다.

이 정보를 토대로 일련의 알고리즘을 통해 특정 형태를 가지고 있는 픽셀들의 위치를 찾거나, 조건에 만족하는 픽셀들의 값을 알아낼 수 있습니다.

이를 신호 처리 기법이라 하며, 알고리즘을 통해 이미지에서 원하는 정보를 얻을 수 있습니다.

<br>
<br>

## Digital Image Processing의 시작

![2]({{ site.images }}/assets/posts/Theory/ComputerVision/lecture-2/2.jpg)

이미지 프로세싱(영상 처리)은 1964년 미국의 `제트 추진 연구소(Jet Propulsion Laboratory)`에서 시작되었습니다.

`미국 항공우주국(NASA)`의 우주 개발 계획, 무인 탐사 우주선 등의 연구 개발 및 운용을 담당하는 연구소입니다.

디지털 이미지 프로세싱의 탄생은 달 표면을 촬용한 **위성 사진의 화질과 왜곡을 개선**하기 위하여 디지털 컴퓨터를 사용하면서 시작되었습니다.

달 표면의 근접 이미지를 지구로 전송한 미국 최초의 우주 탐사선 **Ranger 7**에서 전송된 달 영상은 카메라의 성능과 왜곡으로 인해 정확한 달 표면 이미지를 제대로 확인할 수 없었습니다.

우주선에서 전달된 달 표면 영상의 화질과 왜곡 등을 **복원**하기 위해 디지털 이미지 프로세싱을 적용하는 계기가 되었습니다.

달 표면 영상의 `화질 개선`과 `기하학적 왜곡 보정` 등을 통해 디지털 이미지 프로세싱의 유의미한 결과를 얻어내었고, 우주 탐사 이외에도 산업 현장과 의학 분야에도 적용되기 시작했습니다. 

이 후, 디지털 이미지 프로세싱 분야는 `물체 인식`, `제조 공정 자동화`, `의료 영상 처리`, `문자 인식`, `얼굴 인식` 등의 영역까지 발전하였습니다.

<br>
<br>

## Digital Image Processing의 분류

![3]({{ site.images }}/assets/posts/Theory/ComputerVision/lecture-2/3.jpg)

이미지 프로세싱은 영상의 모든 형태의 정보처리를 의미하며, 주로 영상의 **인식**과 **이해** 등을 중점적으로 **연구하고 해석**하는 분야입니다.

즉, 영상이나 이미지를 **재가공하여 정보를 추출하는 역할**을 합니다. 

다양한 이미지 프로세싱 기법이 존재하며, 일상 생활에서도 쉽게 접할 수 있는 이미지의 **확대**, **축소**, **회전**, **색상 보정** 등도 이미지 프로세싱 기법들 중 하나입니다.

비교적 간단하다고 생각했었던, 이미지 확대나 회전도 `아핀 변환(Affine Transformation)` 연산을 적용해 이미지를 변경합니다.

비교적 심화된 연산으로는 `인식`, `분석`, `조작` 등이 있습니다.

`인식`은 육안으로 식별이 불가능한 영역에서 차이점을 찾아 다른 이미지 또는 영상과 **비교 분석하여 특징을 찾는 것**을 의미합니다. 지문 인식, 병변 검출 등이 있습니다.

`분석`은 이미지 프로세싱에 의하여 **보정 및 변형된 이미지에서 특징을 찾아내는 것**을 의미합니다. 물체의 치수를 측정하거나 위성 사진 분석 등을 의미합니다.

`조작`은 이미지가 너무 흐리거나 노이즈가 많은 경우, 이를 **보정하거나 원하는 정보를 얻기 위하여 변형**하는 것을 의미합니다.

주로, 전처리 단계에서 가장 많이 쓰이며 가장 중요한 부분입니다.

<br>
<br>

* Writer by : 윤대희
