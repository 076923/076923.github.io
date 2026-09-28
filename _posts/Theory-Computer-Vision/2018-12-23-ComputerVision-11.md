---
layout: post
title: "Computer Vision Theory : 이미지 인식"
tagline: "Image Recognition"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Image Recognition, CNN, YOLO, SSD, Faster R-CNN, Segmentation, OCR, Face Embedding, Vision Transformer
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-11/
comments: true
toc: true
---

## 이미지 인식(Image Recognition)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-11/1.webp" class="lazyload" width="100%" height="100%"/>

`이미지 인식(Image Recognition)`은 **객체(Object)**, **장소(Place)**, **얼굴(Face)**, **문자(Character)** 등을 인식하기 위한 방법입니다. `인공 신경망(Artificial Neural Network)`을 사용하여 통계적으로 **높은 확률**을 지니는 값을 찾는 학습 알고리즘입니다.

특정 커널 안의 데이터가 **일정 패턴**을 보이거나 **회귀 분석**을 통한 데이터, 일련의 **함수 형태와 일치**하는 형태 등의 형식을 지니는 이미지를 찾습니다. 대표적으로 `객체 인식(Object Recognition)`, `문자 인식(Character Recognition)`, `얼굴 인식(Face Recognition)` 등이 있습니다.

<br>
<br>

## 객체 인식(Object Recognition)

`객체 인식(Object Recognition)`은 **머신 러닝(Machine Learning)**과 **딥 러닝(Deep Learning)** 기반의 특징 추출을 기반으로 합니다. **Haar**, **HOG**, **SIFT**, **SURF** 등을 사용하여 전처리 과정을 거치며, **가장자리 검출(Edge Detection)** 등 을 통하여 **분류(Classification)**를 진행하게 됩니다.

해당 객체와 배경 등의 수 천개 이상의 데이터를 통하여 학습을 진행합니다.

<br>
<br>

## 문자 인식(Character Recognition)

`문자 인식(Character Recognition)`은 **CNN(Convolutional Neural Network)**, **RNN(Recurrent Neural Network)**이나 **SVM(Support Vector Machines)** 등의 알고리즘을 통하여 학습을 진행합니다. 문자를 더 단순화하여 2 차원 벡터 좌표의 값을 계산합니다. **윤곽(Contours)** 등을 검출하여 진행하기도 합니다. **문자 영역(Character Localization)**을 검출한 뒤, 여러 문자들을 정합하여 **문자 인식(Character Recognition)**을 진행합니다.

<br>
<br>

## 얼굴 인식(Face Recognition)

`얼굴 인식(Face Recognition)`은 사람의 얼굴마다 고유한 특징을 비교하여 특정 인물에 대한 값을 반환합니다. **눈의 크기**, **코의 길이**, **얼굴형** 등을 구분하여 특정 인물의 특징을 찾아냅니다.

기본적으로 **얼굴 검출(Face Detection)**을 통하여 얼굴이 존재하는 위치를 검출한 후, 해당 얼굴의 **특징점(Landmark)**들을 찾아 해당 위치들이 어떤 패턴을 가지는지로 여러 얼굴을 구분합니다. 특정 위치들이 어떤 형태를 지니는지로 **표정**이나 **상태** 등도 구분할 수 있습니다.

## 딥러닝 기반의 이미지 인식

앞서 언급한 `Haar`, `HOG`, `SIFT`, `SURF`는 사람이 직접 설계한 특징을 사용합니다. 반면 현재의 이미지 인식은 대부분 `합성곱 신경망(Convolutional Neural Network, CNN)`을 기반으로, **어떤 특징을 사용할지까지 모델이 스스로 학습**하는 방식으로 이뤄집니다.

객체 검출은 크게 두 가지 방식으로 나뉩니다. `2단계 검출기(Two-Stage Detector)`는 객체가 있을 만한 후보 영역을 먼저 찾고 그 영역을 분류하며, `R-CNN` 계열이 여기에 속합니다. `1단계 검출기(One-Stage Detector)`는 두 과정을 한 번에 처리하며 `YOLO`나 `SSD`가 대표적입니다.

일반적으로 2단계 방식은 정확도가 높고 1단계 방식은 속도가 빠릅니다. 그러므로 실시간 처리가 필요한 환경에서는 `YOLO` 계열이, 정밀한 검출이 필요한 환경에서는 `Faster R-CNN` 계열이 주로 사용됩니다.

객체의 위치를 사각형이 아니라 **픽셀 단위로 구분**해야 한다면 `분할(Segmentation)`을 사용합니다. 같은 종류의 객체를 하나로 묶는 `시맨틱 분할(Semantic Segmentation)`과 개체마다 구분하는 `인스턴스 분할(Instance Segmentation)`로 나뉘며, 각각 `U-Net`과 `Mask R-CNN` 등이 사용됩니다.

문자 인식은 문자의 위치를 찾는 `문자 검출(Text Detection)`과 찾아낸 영역을 읽어내는 `문자 인식(Text Recognition)`의 두 단계로 구성됩니다. 검출에는 `EAST`나 `CRAFT`가, 인식에는 `CRNN` 계열이 주로 사용되며, 이 둘을 묶은 도구로 `Tesseract`나 `PaddleOCR` 등이 있습니다.

얼굴 인식은 얼굴을 **고정된 길이의 벡터로 변환**한 뒤 벡터 사이의 거리로 동일 인물 여부를 판단하는 방식으로 바뀌었습니다. 이 벡터를 `임베딩(Embedding)`이라 하며, `FaceNet`이나 `ArcFace` 등이 대표적입니다. 이 방식은 새로운 인물이 추가돼도 모델을 다시 학습할 필요가 없다는 장점이 있습니다.

- Tip : 최근에는 `트랜스포머(Transformer)`를 영상에 적용한 `ViT(Vision Transformer)`나 `DETR` 계열도 널리 사용되며, 대량의 데이터로 사전 학습된 모델을 가져와 목적에 맞게 조정하는 `전이 학습(Transfer Learning)`이 일반적인 접근 방법으로 자리 잡았습니다.

<br>
<br>

<br>
<br>

* Writer by : 윤대희
