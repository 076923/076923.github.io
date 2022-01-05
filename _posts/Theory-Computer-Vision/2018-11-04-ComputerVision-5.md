---
layout: post
title: "Computer Vision Theory : 관심 영역 & 관심 채널"
tagline: "Region Of Interest & Channel Of Interest"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Region Of Interest, Channel Of Interest
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-5/
comments: true
toc: true
---

## 관심 영역(ROI)

![1]({{ site.images }}/assets/posts/Theory/ComputerVision/lecture-5/1.webp){:class="lazyload" width="100%" height="100%"}

`관심 영역`이란 이미지 상에서 **관심 있는 영역을 의미**합니다.

`ROI`라 부르며 **Region Of Interest**의 약자입니다. 이미지를 처리함에 있어서 객체를 탐지하거나 검출하는 경우, 이부분을 명확하게 관심 영역이라 지정할 수 있습니다.

알고리즘을 구성시, 초기 검출 단계에서는 이미지 전체에 대해 오브젝트를 검출합니다.

이 후, 추가적인 알고리즘이 적용된다면 해당 오브젝트의 이미지 영역에서 알고리즘을 진행하는 방법이 가장 좋습니다.

객체를 탐지한 이후에 **두 번째 알고리즘을 적용할 때 객체 주변 영역에 대해서 불필요한 연산이 들어가게 됩니다.**

불필요한 이미지 영역에 대해서도 연산이 시작되므로 그 만큼 **연산량이 많아지고 많은 리소스를 소모하게됩니다.**

알고리즘의 정확도와 연산속도의 향상을 위해서는 `관심 영역`을 설정해야합니다.

만약, 이미지에서 시계에서 시간을 받아오는 알고리즘을 구성한다면, 첫 번째로 시계라는 객체를 검출하고, **붉은색 영역**을 관심 영역으로 설정한 다음 해당 이미지 안에서 시간을 검출하는 알고리즘을 구성해야합니다.

관심 영역 지정을 효율적으로 사용한다면 알고리즘의 `정확도`와 `연산 속도`를 높일 수 있습니다.

<br>
<br>

## 관심 채널(COI)

![2]({{ site.images }}/assets/posts/Theory/ComputerVision/lecture-5/2.webp){:class="lazyload" width="100%" height="100%"}

`관심 채널`이란 이미지 상에서 **관심 있는 채널을 의미**합니다.

`COI`라 부르며 **Channel Of Interest**의 약자입니다.

역시, 이미지를 처리함에 있어서 특정 채널을 사용하여 연산을 진행하는 경우, 이 부분을 관심 채널이라 부를 수 있습니다.

**색상 이미지 (BGR)**에는 매우 많은 데이터가 담겨있습니다.

이때 채널을 분리하여 특정 채널에 대해 연산을 시작한다면 단순하게 `1/3배`로 데이터의 양이 줄어듭니다.

또한, 채널을 모두 분리한 뒤에 동일한 알고리즘을 적용하여 더 많은 결과를 얻을 수 있습니다. 단순하게 계산하는 데이터의 양은 `1/3배`로 줄지만, 얻어내는 데이터의 양은 `3배`로 늘어나게 됩니다.

채널을 분리하였을 때, `GrayScale`과 비슷한 형태를 보입니다.

하지만, `GrayScale`의 경우, `Y = 0.299 x R + 0.587 x G + 0.114 x B`의 공식을 통해 얻어진 값입니다.

각 채널의 값에 대한 가중치의 곱으로 `GrayScale`을 생성합니다. 확실하게 데이터의 변형이 생깁니다. `Green 채널`에 대한 값에 가장 영향을 많이 받으며, `Blue 채널`에 대해 가장 영향을 적게 받습니다.

특정 알고리즘을 구성할 때에는 각각의 채널에 대한 연산이 더 높은 정확도를 얻어낼 수 있습니다. 

<br>

`관심 영역`과 `관심 채널` 둘 다 데이터의 형태를 변형하는 것이 아닌, **특정 영역을 불러와 연산량과 정확도를 높이는데 사용할 수 있습니다.**

알고리즘을 구성하는데 있어 **관심 영역**, **관심 채널**, **비관심 영역**, **비관심 채널**을 구분한다면 효율 높은 프로그램을 구성할 수 있습니다.

<br>
<br>

* Writer by : 윤대희
