---
layout: post
title: "Artificial Intelligence Theory : 인공 지능이란?"
tagline: "Artificial Intelligence이란?"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Machine Learning, Deep Learning
ref: Theory-AI
category: Theory
permalink: /posts/AI-1/
comments: true
toc: true
---

## 인공 지능(Artificial Intelligence)

![1]({{ site.images }}/assets/posts/Theory/ArtificialIntelligence/lecture-1/1.png)

인공지능이란 `학습`, `추론`, `인식` 등 인간과 관계된 인지 문제를 컴퓨터 프로그램으로 구현한 기술입니다.

인공지능 용어의 첫 등장은 1956년 미국 다트머스 대학교에서 열린 학회에서 처음 사용됐습니다.

스탠퍼드 대학교의 존 매카시(John McCarthy) 교수가 인공지능이란 용어를 처음 사용하였습니다.

기존에 인간만이 실현할 수 있다고 생각한 역할을 컴퓨터가 수행할 수 있도록 구현하여 **인위적으로 만들어진 지능**을 뜻합니다.

즉, **인간의 지능을 기계나 프로그램 등에 인공적으로 구현한 것**을 의미합니다.

인공지능을 크게 두 가지로 나누자면 `강인공지능(Strong Artificial Intelligence)`과 `약인공지능(Weak Artificial Intelligence)`이 있습니다.

`강인공지능`은 스스로 학습과 인식 등이 가능하며, 지능 또는 지성의 수준이 **인간과 근사한 수준**까지 이른 경우를 뜻합니다.

주로 SF 영화 등에 나타나는 휴머노이드나 안드로이드를 생각할 수 있습니다.

`약인공지능`은 인간이 해결할 수 있으나, 기존의 컴퓨터로 처리하기 **힘든 작업들을 처리하기 위한 일련의 알고리즘**을 의미합니다.

현재 많은 곳에서 활용되고 있는 A.I. 서비스라고 볼 수 있습니다. 

다시 말해, 주어진 시스템에서 입력을 조절해 출력을 원하는 대로 조절하는 `제어기`로부터 측정가능한 경험적(heuristic) 속성을 학습해 스스로 판단하는 `심층 학습`까지의 전반을 의미합니다.

인공지능은 인간과 비슷하거나 **합리적 행동을 통해 특정한 문제를 해결하는데 중점**을 두고 있습니다.

인공 지능 분야에서 파생된 컴퓨터 과학 분야로는 `머신 러닝(기계 학습, Machine Learning)`과 `딥 러닝(심층 학습, Deep Learning)` 등이 있습니다.

<br>
<br>

## 머신 러닝(Machine Learning)

![2]({{ site.images }}/assets/posts/Theory/ArtificialIntelligence/lecture-1/2.png)

`머신 러닝(Machine Learning)`이란 데이터를 기반으로 컴퓨터를 프로그래밍하는 연구 분야입니다.

기존 프로그래밍은 명시적인 프로그래밍을 통해서 시스템을 구축하였지만, 머신 러닝은 데이터를 기반으로 학습하거나 시스템의 성능을 개선하는 데 중점을 두고 있습니다.

즉, 기존의 프로그래밍은 `규칙`과 `데이터`를 기반으로 **결괏값**을 예측햇지만 머신 러닝은 `데이터`와 `결괏값`으로 **규칙**을 찾아내는 분야입니다.

머신 러닝은 학습 데이터를 분석하여 일정한 규칙이나 패턴을 찾아 **예측 알고리즘**을 생성합니다.

이 예측 알고리즘을 `모델(Model)`이라 하며, 모델에 새로운 데이터가 입력되었을 때 모델의 예측값으로 결과를 추론할 수 있습니다. 

데이터를 기반으로 알고리즘을 구성하므로, 통계적인 접근 방법을 사용한다 볼 수 있습니다. 

머신 러닝 알고리즘에는 `지도 학습 (Supervised Learning)`, `비지도 학습(Unsupervised Learning)`, `강화 학습(Reinforcement learning)`, `심층 학습(Deep Learning)` 등이 있습니다.

<br>
<br>

## 딥 러닝(Deep Learning)

![3]({{ site.images }}/assets/posts/Theory/ArtificialIntelligence/lecture-1/3.png)

`딥 러닝(Deep Learning)`은 머신 러닝의 기법 중 하나로, 모델이 스스로 데이터의 관계를 파악해 학습합니다.

**여러 층(Layer)**을 가진 `인공 신경망(Artificial Neural Network, ANN)`을 사용하여 머신 러닝 학습을 수행합니다.

여기서 인공 신경망은 인간의 뇌에서 있는 뉴런의 네트워크에서 영감을 얻은 **통계학적 학습 알고리즘**입니다.

생물학적 뉴런은 단순하게 다른 뉴런에게 신호를 받아 또 다른 뉴런에게 신호를 전달합니다.

뉴런은 수십업 개 이상 구성된 네트워크로 이뤄져 있어 신호의 흐름으로 복잡하고 다양한 활동을 할 수 있게합니다.

결국, 신경망은 서로 연결된 `노드(Node)`의 집합으로 구성되어 있으며 여러 층으로 이뤄집니다. 

층(Layer)은 크게 `입력층(Input Layer)`, `은닉층(Hidden Layer)`, `출력층(Output Layer)`이 존재합니다.

입력층으로 학습하고자 하는 데이터를 받고, 여러 개의 은닉층을 지나 단계를 거쳐 출력층에서 결과를 반환합니다.

인공 신경망에 학습 알고리즘과 데이터를 지속적으로 제공함으로써, 학습 능력과 사고 능력을 지속적으로 개선됩니다.

딥러닝 알고리즘에는 `합성곱 신경망(Convolutional Neural Network, CNN)`, `순환 신경망(Recurrent Neural Network, RNN)`, `심층 신경망(Deep Neural Network, DNN)` 등이 있습니다.

- Tip : 딥(Deep)이란, 지속적인 개선으로 얻어지는 신경망의 은닉 층(Hidden Layer)을 의미합니다.

<br>
<br>

* Writer by : 윤대희