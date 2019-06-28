---
layout: post
title: "Computer Vision Theory : 유사성 검출"
tagline: "Similarity Detection"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Similarity Detection
ref: ComputerVision
category: posts
permalink: /posts/ComputerVision-10/
comments: true
---

## 유사성 검출(Similarity Detection) ##
----------

![1]({{ site.images }}/assets/images/ComputerVision/ch10/1.jpg)
`유사성 검출(Similarity Detection)`은 이미지 내의 주요한 `유사 영역`을 검출하는 방법입니다. 해당 비슷한 영역이 존재하는 위치를 알려주거나 해당 유사 영역을 부각시킵니다. 이미지에서 검출하려는 오브젝트의 **유사성**을 기준으로 검출을 진행합니다. 감지할 이미지와 유사한 이미지인 **Positive Image**와 전혀 다른 이미지인 **Negative Image**로 훈련된 모델(Model)을 사용하거나 검출하려는 이미지를 **특정 크기로 변경**시켜 이미지에서 **일치하는 영역이 높은 영역**을 찾습니다. 또한 **특정 범위 내**에 있는 객체를 검출합니다.

대표적으로 `얼굴 검출(Face Detection)`, `피부색 검출(Skin Color Detection)`, `템플릿 매칭(Template Matching)` 등이 있습니다.

<br>
<br>

## 얼굴 검출(Face Detection) ##

`얼굴 검출(Face Detection)`은 이미지 상에서 얼굴을 검출하기 위한 알고리즘입니다. 수만 수천개의 **Postive Image**와 **Negative Image**로 교육된 모델을 사용하여 검출을 진행합니다. 얼굴에서의 특징점인 **눈**이나 **코**, **입술** 등에서 밝고 어두운 정도의 특징점을 사용하여 검출을 진행합니다. 해당 특징점들이 모두 존재하는 위치를 찾습니다.

<br>
<br>

## 피부색 검출(Skin Color Detection) ##

`피부색 검출(Skin Color Detection)`은 이미지 내의 피부색을 검출을 위한 알고리즘입니다. 피부색으로 간주할 수 있는 색상에 대해 **하위 임계값**과 **상위 임계값**을 두어 피부색과 유사한 색상을 피부색으로 간주합니다.

<br>
<br>

## 템플릿 매칭(Template Matching) ##

`템플릿 매칭(Template Matching)`은 이미지에서 **유사성이** 가장 높은 이미지를 검출합니다. 원본 이미지와 검출하려는 템플릿 이미지를 변형시킨 뒤 서로가 얼마나 비슷한지 **대조**하여 검출을 진행합니다. 서로를 대조하였을 때** 가장 높은 일치점**을 찾아 결과로 반환합니다.

<br>
<br>

* Writer by : 윤대희