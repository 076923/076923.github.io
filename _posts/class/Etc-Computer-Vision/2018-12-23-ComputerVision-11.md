---
layout: post
title: "Computer Vision Theory : 이미지 인식"
tagline: "Image Recognition"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Image Recognition
ref: ComputerVision
category: posts
permalink: /posts/ComputerVision-11/
comments: true
---

## 이미지 인식(Image Recognition) ##
----------

![1]({{ site.images }}/assets/images/ComputerVision/ch11/1.jpg)
`이미지 인식(Image Recognition)`은 **객체(Object)**, **장소(Place)**, **얼굴(Face)**, **문자(Character)** 등을 인식하기 위한 방법입니다. `인공신경망(Artificial Neural Network)`을 사용하여 통계적으로 **높은 확률**을 지니는 값을 찾는 학습 알고리즘입니다. 특정 커널 안의 데이터가 **일정 패턴**을 보이거나 **회귀 분석**을 통한 데이터, 일련의 **함수 형태와 일치**하는 형태 등의 형식을 지니는 이미지를 찾습니다.

대표적으로 `객체 인식(Object Recognition)`, `문자 인식(Character Recognition)`, `얼굴 인식(Face Recognition)` 등이 있습니다.

<br>
<br>

## 객체 인식(Object Recognition) ##

`객체 인식(Object Recognition)`은 **머신 러닝(Machine Learning)**과 **딥 러닝(Deep Learning)** 기반의 특징 추출을 기반으로 합니다. **Haar**, **HOG**, **SIFT**, **SURF** 등을 사용하여 전처리 과정을 거치며, **가장자리 검출(Edge Detection)** 등 을 통하여 **분류(Classification)**를 진행하게 됩니다. 해당 객체와 배경 등의 수 천개 이상의 데이터를 통하여 학습을 진행합니다.

<br>
<br>

## 문자 인식 (Character Recognition) ##

`문자 인식 (Character Recognition)`은 **CNN(Convolutional Neural Network)**, **RNN(Recurrent Neural Network)**이나 **SVM(Support Vector Machines)** 등의 알고리즘을 통하여 학습을 진행합니다. 문자를 더 단순화하여 2 차원 벡터 좌표의 값을 계산합니다. **윤곽(Contours)** 등을 검출하여 진행하기도 합니다. **문자 영역(Character Localization)**을 검출한 뒤, 여러 문자들을 정합하여 **문자 인식(Character Recongnition)**을 진행합니다.

<br>
<br>

## 얼굴 인식(Face Recognition) ##

`얼굴 인식(Facial Recognition)`은 사람의 얼굴마다 고유한 특징을 비교하여 특정 인물에 대한 값을 반환합니다. **눈의 크기**, **코의 길이**, **얼굴형** 등을 구분하여 특정 인물의 특징을 찾아냅니다. 기본적으로 **얼굴 검출(Face Detection)**을 통하여 얼굴이 존재하는 위치를 검출한 후, 해당 얼굴의 **특정 위치(Land Mark)**들을 찾아 해당 위치들이 어떤 패턴을 가지는지로 여러 얼굴을 구분합니다. 특정 위치들이 어떤 형태를 지니는지로 **표정**이나 **상태** 등도 구분할 수 있습니다. 

<br>
<br>

* Writer by : 윤대희