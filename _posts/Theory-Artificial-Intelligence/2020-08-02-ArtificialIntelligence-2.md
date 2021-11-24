---
layout: post
title: "Artificial Intelligence Theory : 지도 학습"
tagline: "Supervised Learning"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Machine Learning, Regression, Univariate, Multivariate, Predictor Variable, Explanatory Variable, Outcome Variable, Response Variable, Classification, Logistic Regression, Softmax Regression
ref: Theory-AI
category: Theory
permalink: /posts/AI-2/
comments: true
toc: true
---

## 지도 학습(Supervised Learning)

![1]({{ site.images }}/assets/posts/Theory/ArtificialIntelligence/lecture-2/1.jpg)

지도 학습이란 컴퓨터(알고리즘)에 `훈련 데이터(train data)`와 `레이블(Label)`을 포함시켜 학습을 하는 방법입니다.

여기서 훈련 데이터는 **입력 데이터(input data)**가 되며, 레이블은 **정답(label data)**이 됩니다.

즉, 훈련 데이터로부터 하나의 함수를 유추해내기 위한 기계 학습의 한 방법입니다.

지도 학습의 훈련 데이터(train data)는 일반적으로 입력 데이터(input data)에 대한 속성을 `벡터` 형태로 포함하고 있으며, 각각의 벡터에 대해 원하는 결과(label data)가 무엇인지 표시되어 있습니다.

지도 학습에는 크게 `회귀 분석(Regression)`과 `분류(Classification)` 등이 있습니다.

<br>
<br>

## 회귀 분석(Regression)

회귀 분석은 연속형 변수들에 대해 두 변수 사이의 모형을 구한뒤 적합도를 측정해 내는 분석 방법입니다.

`로버스트(Robust)`, `라쏘(Lasso)`, `로지스틱(Logistic)` 등을 비롯해, `합성곱 신경망(CNN)`, `순환 신경망(RNN)`까지 회귀분석에 포함할 수 있습니다.

회귀 분석은 크게, **선형(Linear)**과 **비선형(Non-Linear)**으로 나눌 수 있습니다. 

여기서 선형은 직선의 특징을 갖고 있음을 뜻하며, `중첩의 원리(superposition principle)`가 적용되는 것을 의미합니다.

즉, 비선형은 예측하기 힘든 값이 나와 함수의 수식을 예측하기가 어렵습니다.

그러므로, 비선형 회귀 분석을 진행할 때에는 **CNN, RNN, DNN** 등을 이용합니다.

선형 회귀는 다시 크게 `단변량(Univariate)`과 `다변량(Multivariate)`으로 나눠집니다.

단변량은 종속 변수(Y)가 하나일 때를 의미하며, 다변량은 종속 변수(Y)가 두 개 이상일 때를 의미합니다.

단변량에서는 종속 변수(Y)가 하나만 존재하므로, 독립 변수(X)가 하나 이상의 값을 사용할 수 있습니다.

여기서 독립 변수(X)가 하나라면 `단순(Simple)`이 되며, 두 개 이상이 되면 `다중(Multiple)`이 됩니다.

이를 다시 정리하자면, 다음과 같이 설명할 수 있습니다.

하나의 종속 변수와 하나의 독립 변수 사이의 관계를 분석하는 경우에는 `단순 선형 회귀 분석(Simple Linear Regression Analysis)`이라 합니다.

만약, 하나의 종속 변수와 여러 개의 독립 변수 사이의 관계를 분석한다면 `다중 선형 회귀 분석(Multiple Linear Regression Analysis)`이라 합니다.

단순 선형 회귀 분석을 수식으로 나타낸다면 `Y = Wx + b`의 형태가 되며, 다중 선형 회귀 분석을 수식으로 나타낸다면 `Y = W1x1 + W2x2 + ... + b`의 형태가 됩니다.

Y는 결과로 종속 변수를 의미합니다.

X는 결과에 영향을 미치는 요소로 독립 변수를 의미합니다.

W는 독립 변수에 영향을 미치는 `가중치(Weight)`를 의미합니다.

b는 외부에서 영향을 미치는 값으로 `편향성(Bias)`을 의미합니다. 

- Tip : 독립 변수(X)는 **예측 변수(Predictor Variable)** 또는 **원인 변수, 설명 변수(Explanatory Variable)**라고도 부릅니다.

- Tip : 종속 변수(Y)는 **결과 변수(Outcome Variable)** 또는 **반응 변수, 목적 변수(Response Variable)**라고도 부릅니다.

<br>

예를 들어, 나의 출근 시간을 예측하는 모델을 만든다 가정한다면 Y는 출근 시간, X는 출근 시간에 영향을 주는 요소입니다.

W는 출근 시간에 영향을 주는 요소의 가중치를 의미합니다.

출근 시간에 영향을 주는 요소를 일어난 시간, 아침 식사 시간, 아침 출근 준비로 가정한다면 각각, X1, X2, X3로 볼 수 있습니다.

만약, 단순 선형 회귀 분석으로 출근 시간을 예측할 때 X은 일어난 시간이며, W은 해당 독립 변수의 영향력입니다.

다시 수식으로 표현한다면, `출근 시간 = 영향력 * 일어난 시간 + 편향성`으로 정의할 수 있습니다.

이번에는 다중 선형 회귀 분석으로 출근 시간을 예측한다면 `출근 시간 = 영향력1 * 일어난 시간 + 영향력2 * 아침 식사 시간 + 영향력3 * 아침 출근 시간 + 편향성`이 됩니다.

- Tip : 여기서 `일어난 시간`, `아침 식사 시간`, `아침 출근 준비`는 **특성(feature)**이 되며, `출근 시간`은 **목표(target)**가 됩니다.

<br>
<br>

## 분류(Classification)

분류는 `훈련 데이터(train data)`에서 지정된` 레이블(label)`과의 관계를 분석해, 새로운 데이터의 레이블을 **스스로 판별하는 방법**입니다.

즉, 새로운 데이터를 대상으로 **카테고리(category)**를 스스로 판단합니다.

새로운 데이터를 대상으로 참인지 거짓인지 분류할 수 있다면, `이진 분류(Binary Classification)`라 합니다.

만약, 새로운 데이터를 대상으로 두 개 이상의 카테고리를 나눠 분류 할 수 있다면, `다중 분류(Multi-label Classification)`라 합니다.

예를 들어, 시험 성적으로 합격 여부를 판단한다면 `합격(참)`과 `불합격(거짓)`으로 구분할 수 있으므로 이진 분류가 됩니다.

다중 분류는 동물 이미지를 입력했을 때 강아지, 고양이, 새 등으로 분류하는 것을 의미합니다.

이진 분류의 대표적인 알고리즘은 `로지스틱 회귀(Logistic Regression)`가 있습니다.

분류에도 회귀가 사용될 수 있는데, 로지스틱 회귀는 종속변수에 로짓 변환을 실시해 편향성(bias)이 없는 타당한 계수를 추정할 수 있습니다.

간단하게 설명하자면 시험 성적을 그래프화 하여 합격 여부에 참과 거짓을 나타낸다면 극단적인 그래프가 그려질 수 있습니다.

이 그래프에서 특정 점수을 대입해 합격 여부를 확인할 수 있습니다.

다중 분류의 대표적인 알고리즘은 `소프트맥스 회귀(Softmax Regression)`가 있습니다.

이진 분류에서 참/거짓으로 분류할 때 참일 경우 1, 거짓일 경우 0으로 분류했습니다.

즉, 나온 결괏값을 모두 합한다면 1이 됩니다. 이와 같이 다중 분류도 합이 1이 되는 확률 분포를 구성할 수 있습니다.

소프트맥스 회귀는 가중치를 정규화해 나온 결괏값을 모두 더할 때 1이 되게끔 구성하는 것을 의미합니다.

A, B, C를 분류하는 알고리즘을 만든다면 A는 0.1, B는 0.1, C는 0.8로 나오게 될 수 있습니다.

이는 A일 확률은 10%, B일 확률은 10%, C일 확률은 80%를 의미합니다. 이 중 **가장 높은 확률인 C를 선택**하게 됩니다.

이 확률 분포를 통해서 다중 분류를 진행할 수 있습니다. 

- Tip : 레이블의 범주를 클래스(class)로 부르며, Python의 Class와는 다른 의미입니다.

<br>
<br>

* Writer by : 윤대희