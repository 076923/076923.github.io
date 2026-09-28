---
layout: post
title: "Computer Vision Theory : 유사성 검출"
tagline: "Similarity Detection"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Similarity Detection, Haar Cascade, Template Matching, Skin Color Detection
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-10/
comments: true
toc: true
---

## 유사성 검출(Similarity Detection)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-10/1.webp" class="lazyload" width="100%" height="100%"/>

`유사성 검출(Similarity Detection)`은 이미지 내의 주요한 `유사 영역`을 검출하는 방법입니다. 해당 비슷한 영역이 존재하는 위치를 알려주거나 해당 유사 영역을 부각시킵니다. 이미지에서 검출하려는 오브젝트의 **유사성**을 기준으로 검출을 진행합니다. 감지할 이미지와 유사한 이미지인 **Positive Image**와 전혀 다른 이미지인 **Negative Image**로 훈련된 모델(Model)을 사용하거나 검출하려는 이미지를 **특정 크기로 변경**시켜 이미지에서 **일치하는 영역이 높은 영역**을 찾습니다.

또한 **특정 범위 내**에 있는 객체를 검출합니다. 대표적으로 `얼굴 검출(Face Detection)`, `피부색 검출(Skin Color Detection)`, `템플릿 매칭(Template Matching)` 등이 있습니다.

<br>
<br>

## 얼굴 검출(Face Detection)

`얼굴 검출(Face Detection)`은 이미지 상에서 얼굴을 검출하기 위한 알고리즘입니다. 수천에서 수만 장에 이르는 **Positive Image**와 **Negative Image**로 학습된 모델을 사용하여 검출을 진행합니다. 얼굴에서의 특징점인 **눈**이나 **코**, **입술** 등에서 밝고 어두운 정도의 특징점을 사용하여 검출을 진행합니다.

해당 특징점들이 모두 존재하는 위치를 찾습니다.

<br>
<br>

## 피부색 검출(Skin Color Detection)

`피부색 검출(Skin Color Detection)`은 이미지 내의 피부색을 검출하기 위한 알고리즘입니다. 피부색으로 간주할 수 있는 색상에 대해 **하위 임계값**과 **상위 임계값**을 두어 피부색과 유사한 색상을 피부색으로 간주합니다.

<br>
<br>

## 템플릿 매칭(Template Matching)

`템플릿 매칭(Template Matching)`은 이미지에서 **유사성이 가장 높은 영역**을 검출합니다. 원본 이미지와 검출하려는 템플릿 이미지를 변형시킨 뒤 서로가 얼마나 비슷한지 **대조**하여 검출을 진행합니다. 서로를 대조하였을 때 **가장 높은 일치점**을 찾아 결과로 반환합니다.

## 검출 방식의 한계

앞서 설명한 세 가지 방법은 모두 **미리 정해 둔 기준과 얼마나 닮았는지**를 비교합니다. 구현이 간단하고 연산이 가벼우며 학습 데이터가 거의 필요 없다는 장점이 있지만, 그만큼 조건이 달라지면 검출에 실패하기 쉽습니다.

`템플릿 매칭(Template Matching)`은 템플릿 이미지를 원본 위에서 한 칸씩 옮겨 가며 비교하는 방식이므로, 대상이 **회전하거나 크기가 달라지면 찾지 못합니다.** 크기 변화에 대응하려면 템플릿의 크기를 여러 단계로 바꿔 가며 반복해야 하고, 회전까지 고려하면 연산량이 급격히 늘어납니다.

`피부색 검출(Skin Color Detection)`은 색상 범위에 의존하므로 조명이 바뀌거나 피부색이 다양한 환경에서는 임계값을 다시 설정해야 합니다. 이때는 밝기와 색상이 분리돼 있는 `HSV`나 `YCrCb` 색상 공간을 사용하면 조명 변화의 영향을 비교적 덜 받습니다.

`얼굴 검출(Face Detection)`에 주로 사용되는 `하르 캐스케이드(Haar Cascade)`는 밝고 어두운 영역의 대비 패턴을 단계적으로 검사하는 방식이라 정면 얼굴에는 잘 동작하지만, 옆얼굴이나 기울어진 얼굴에는 취약합니다.

이러한 한계 때문에 회전이나 크기 변화가 큰 환경에서는 [특징 검출][9강]에서 다룬 `특징점 매칭(Feature Matching)`을 사용하거나, 학습 기반의 검출 모델을 사용합니다. 반대로 대상의 형태와 촬영 조건이 일정한 환경이라면 유사성 검출만으로도 충분히 빠르고 안정적인 결과를 얻을 수 있습니다.

- Tip : 어떤 방법이 항상 우월한 것은 아니므로, **검출 대상이 놓이는 조건이 얼마나 일정한가**를 기준으로 선택해야 합니다.

<br>
<br>

<br>
<br>

* Writer by : 윤대희

[9강]: https://076923.github.io/posts/ComputerVision-9/
