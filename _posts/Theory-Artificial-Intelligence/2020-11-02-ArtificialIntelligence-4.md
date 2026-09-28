---
layout: post
title: "Artificial Intelligence Theory : 강화 학습(Reinforcement Learning)"
tagline: "Reinforcement Learning"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Machine Learning, Reinforcement Learning, Exploration, Exploitation, Q-Learning, DQN, Policy Gradient, PPO
ref: Theory-AI
category: Theory
permalink: /posts/AI-4/
comments: true
toc: true
---

## 강화 학습(Reinforcement Learning)

<img data-src="{{ site.images }}/assets/posts/Theory/ArtificialIntelligence/lecture-4/1.webp" class="lazyload" width="100%" height="100%"/>

`강화 학습(Reinforcement Learning)`은 **머신 러닝(Machine Learning)**의 분야 중 하나로, 행동주의(Behaviorism) 심리학 이론을 토대로 구현된 알고리즘입니다. 행동주의는 모든 동물은 학습 능력을 가지고 있으므로, 어떤 행동을 수행했을 때 **보상(reward)**이 주어진다면, 보상을 받았던 행동의 발생 빈도가 높아진다는 이론입니다.

이러한 이론을 토대로 보상을 최적화하기 위한 `강화 학습`이 생겨났습니다. 강화 학습은 크게, `환경(Environment)`, `에이전트(Agent)`, `상태(State)`, `행동(Action)`, `보상(Reward)`, `정책(Policy)` 등으로 구성되어 있습니다. `환경(Environment)`이란 학습을 진행하는 **공간** 또는 **배경**을 의미합니다. 예를 들어, 바둑에서의 환경은 바둑판이며, 게임에서의 환경은 게임 속 세상을 의미합니다.

`에이전트(Agent)`는 환경과 상호작용하는 프로그램을 의미합니다. 즉, **플레이어(Player)**나 **관측자(Observer)**를 지칭합니다. `상태(State)`란 환경에서 에이전트의 상황을 의미합니다. 바둑에서 매 턴 돌의 상태나 게임의 각각의 프레임일수도 있습니다. `행동(Action)`이란 주어진 환경의 상태에서 에이전트가 취하는 행동을 의미합니다. 돌을 놓거나 움직이는 등의 모든 행위를 지칭합니다.

`보상(Reward)`은 현재 환경의 상태에서 에이전트가 어떠한 행동을 취했을 때, 제공되는 보상을 의미합니다. **양(Positive)의 보상**, **음(Negative)의 보상**, **0의 보상** 등을 돌려받습니다. 양의 보상이 주어진다면, 그 환경의 상태에서 에이전트는 해당 행동을 **더 많이 취할 가능성이 높아집니다.** 반대로 음의 보상에서는 **해당 행동의 발생 빈도가 낮아집니다.**

마지막으로 `정책(Policy)`은 에이전트가 보상을 최대화하기 위해 행동하는 알고리즘을 의미합니다. 즉, 에이전트는 반복되는 학습을 통해 보상을 최대화하는 행동을 취하게 됩니다.

- Tip : 심리학에서 reinforcement는 생물이 어떤 자극에 반응해 미래의 행동을 바꾸는 것을 의미합니다.

<br>
<br>

## 마르코프 결정 과정(Markov Decision Process, MDP)

<img data-src="{{ site.images }}/assets/posts/Theory/ArtificialIntelligence/lecture-4/2.webp" class="lazyload" width="100%" height="100%"/>

강화 학습을 간단한 순서도로 변경한다면 위와 같은 이미지로 표현할 수 있습니다. 위의 순서도는 마르코프 결정 과정에서 행동이라는 요소가 추가된 형태입니다. 에이전트는 $$S_t$$ 상태에서 $$A_t$$의 행동을 수행합니다. 그러면 환경은 다음 번 상태의 $$ S_{t+1} $$과 다음 번 보상인 $$ R_{t+1} $$을 에이전트에게 전달합니다.

강화 학습에서 중요한 가정 중 하나는 환경이 마르코프 결정 과정의 **마르코프 속성(Markov property)**을 가진다는 것 입니다. 마르코프 속성은 과거 상태($$ S_1, S_2, ..., S_{t-1} $$)들과 현재 상태($$ S_t $$)가 주어졌을 때, 미래 상태($$ S_{t+1} $$)가 **현재 상태와 현재 행동에 의해서만 결정된다**는 것을 의미합니다. 즉, 현재 상태가 과거의 모든 정보를 요약하고 있으므로 과거 상태를 따로 기억할 필요가 없습니다.

지속적으로 상태가 변화하게 되는데, 어떤 상태에서 다음 상태로 변화하는 것을 `전이(transition)`라 합니다. 결국, $$ t $$ 시점의 상태 $$ S_t $$에서 행동($$ A_t $$)를 할 때 수행하는 다음 상태의 $$ S_{t+1} $$을 결정하게 됩니다. 이를 `상태 전이(state transition)`라 하며, 상태 전이에서 받는 보상은 $$ R(S_{t+1}) $$로 표현할 수 있습니다.

<br>
<br>

## 가치 함수(Value Function)

`가치 함수(Value Function)`란 어떤 상태 $$ S_t $$에서 정책에 따라 행동을 할 때 얻게되는 `기대 보상(Expected Reward)`을 의미합니다. 즉, 상태와 행동에 따라 최종적으로 어떤 보상을 제공해줄지에 대한 예측 함수입니다. 또한, `상태-행동 가치 함수(State-Action Value Function)`도 존재하는데 이는 어떤 상태 $$ S_t $$에서 행동을 한 다음, 정책에 따라 행동을 할 때 얻게되는 `기대 보상`을 의미합니다.

이 가치 함수에 따라 학습이 진행되게 됩니다. 최적의 가치 함수를 구현한다면, 효율적인 정책을 구성할 수 있습니다. 가치 함수는 `벨먼 최적 방정식(Bellman Optimality Equation)`을 적용하며, `동적계획법(Dynamic Programming)`, `몬테 카를로 방법(Monte Carlo Method)`, `모수적 함수(parameterized function)` 등을 사용할 수 있습니다.

결국, 강화 학습은 에이전트의 **시행착오(trial and error)**를 통해 보상을 최대로 할 수 있는 정책을 찾는 방법으로 학습이 진행되게 됩니다. 강화 학습은 크게 모델 기반의 `강화 학습(Model-based Reinforcement Learning)`과 `모델이 없는 강화 학습(Model-free Reinforcement Learning)`이 있습니다.

<br>
<br>

## 모델 기반(Model-based) & 모델 프리(Model-free)

`모델 기반의 강화 학습`과 `모델이 없는 강화 학습`의 이름에서 알 수 있듯이, 모델을 사용하는 여부에 따라 나뉩니다. 강화 학습에서의 모델(Model)은 데이터와 결괏값에 대한 규칙(f(x))이 아닌, **환경에 대한 가정을 모델로 간주합니다.** 즉, 에이전트는 환경의 모델을 통해 `상태 전이`와 `보상`을 예측하게 됩니다.

모델 기반의 강화 학습에서는 에이전트가 어떠한 행동을 할 때, 환경이 어떻게 바뀔지 미리 예측할 수 있습니다. 그러므로, 에이전트가 행동하기 전에 환경의 변화를 예상하여 최적의 행동을 실행할 수 있습니다. 모델 기반의 강화 학습은 **적은 양의 데이터로도 효율적인 학습**을 할 수 있지만, 모델이 정확한 환경을 구현하지 않는다면 **올바른 학습을 진행할 수 없습니다.**

모델을 사용하지 않는 강화 학습은 모델 기반의 강화학습과 정반대의 장/단점을 갖습니다. 즉, **모델을 구현하기 어려운 상황에도 사용할 수 있다는 장점이 있습니다.**

## 탐험과 활용(Exploration & Exploitation)

강화 학습의 에이전트는 학습 과정에서 **이미 알고 있는 좋은 행동을 반복할지**, **아직 해보지 않은 행동을 시도할지**를 매 순간 선택해야 합니다. 전자를 `활용(Exploitation)`, 후자를 `탐험(Exploration)`이라 하며, 두 가지를 어떻게 배분할지가 강화 학습의 핵심 과제 중 하나입니다.

활용에만 치우치면 처음 찾아낸 그럭저럭 괜찮은 행동에 머물러 더 나은 선택지를 영영 발견하지 못합니다. 반대로 탐험에만 치우치면 학습이 수렴하지 않고 보상을 제대로 쌓지 못합니다. 이를 `탐험과 활용의 딜레마(Exploration-Exploitation Dilemma)`라 합니다.

가장 널리 쓰이는 해법은 `엡실론 그리디(ε-greedy)` 방식입니다. $$ \epsilon $$의 확률로 무작위 행동을 선택하고 나머지 확률로는 현재 가장 좋다고 판단한 행동을 선택하며, 학습이 진행될수록 $$ \epsilon $$ 값을 점차 줄여 탐험에서 활용으로 무게 중심을 옮깁니다.

<br>
<br>

## 주요 알고리즘

강화 학습 알고리즘은 무엇을 학습하는지에 따라 크게 세 갈래로 나뉩니다. `가치 기반(Value-based)`은 가치 함수를 학습해 가치가 가장 높은 행동을 선택하고, `정책 기반(Policy-based)`은 정책을 직접 학습하며, `행위자-비평가(Actor-Critic)`는 두 방식을 결합합니다.

가치 기반의 대표적인 알고리즘은 `Q-러닝(Q-Learning)`으로, 상태와 행동의 쌍마다 기대 보상을 표 형태로 기록하며 갱신합니다. 다만 상태의 수가 많아지면 표로 관리할 수 없으므로, 이를 신경망으로 근사한 `DQN(Deep Q-Network)`이 등장했습니다. 가치 기반은 행동의 종류가 유한한 문제에 적합합니다.

정책 기반은 가치 함수를 거치지 않고 정책 자체를 매개변수로 두어 학습하므로, 로봇 관절의 각도처럼 **행동이 연속적인 값**을 갖는 문제에 사용할 수 있습니다. 이 둘을 결합한 행위자-비평가 방식에서는 행위자가 행동을 선택하고 비평가가 그 행동을 평가하며, `A2C`, `PPO`, `SAC` 등이 여기에 속합니다.

- Tip : 강화 학습은 게임이나 로보틱스뿐 아니라 `대규모 언어 모델(Large Language Model, LLM)`을 사람의 선호에 맞추는 `RLHF(Reinforcement Learning from Human Feedback)`에도 활용되며, 이때는 주로 `PPO(Proximal Policy Optimization)`가 사용됩니다.

<br>
<br>

<br>
<br>

* Writer by : 윤대희