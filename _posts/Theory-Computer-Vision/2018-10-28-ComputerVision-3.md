---
layout: post
title: "Computer Vision Theory : 이미지의 세 가지 구성요소"
tagline: "Three components of the image"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Three components of the image
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-3/
comments: true
toc: true
---

## 이미지의 세 가지 구성요소

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/1.webp" class="lazyload" width="100%" height="100%"/>

OpenCV를 통한 이미지를 처리시 세 가지의 구성요소가 존재합니다. 이 세 가지의 구성요소는 영상 처리시 가장 중요하게 고려되어야할 속성입니다.

`이미지의 크기`, `정밀도`, `채널`입니다. 앞의 구성요소를 통하여 불러온 이미지가 어떤 속성을 가지는지 설정합니다.

이 구성요소들이 올바르게 설정되지 않는다면 **영상 처리시 불필요하게 너무 많은 데이터를 처리하거나 부족한 데이터를 받아오게 됩니다.**

이제, 각각의 속성들이 어떤 의미를 지니는지 알아보도록 하겠습니다.

<br>
<br>

## 이미지의 크기(Image Size)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/2.webp" class="lazyload" width="100%" height="100%"/>

먼저 `이미지의 크기`입니다. 이미지의 크기는 해당 필드나 변수에 할당될 이미지의 크기를 설정합니다.

이미지의 크기를 원본 이미지의 크기를 **2배**, **4배**, **1/2배**, **1/4배** 등 설정할 수 있으며 또는 **임의의 크기**로 설정할 수 있습니다.

이미지의 크기는 `데이터의 크기`라 볼 수 있습니다.

고화질의 이미지의 경우, 이미지의 크기가 매우 큽니다. 그에 따라 영상 처리를 진행할 경우, **이미지의 크기 만큼 데이터가 생성되어 너무 많은 연산을 진행하게 됩니다.**

많은 알고리즘에서 **이미지의 크기를 변경하는 메소드 (Image Pyramid, Resize) 등을 전처리 후 적용합니다.**

OpenCV에서는 변수나 필드에 설정된 이미지 크기로 원본 이미지를 불러올 경우, **오류**가 발생합니다.

그 이유는 변수나 필드에 설정된 속성 값은 `액자`의 역할을 한다 볼 수 있습니다.

필드나 변수에 설정된 이미지의 크기에 따라 원본 이미지를 삽입하게 되는데, 이 경우 크기가 다르다면 `액자`의 공간에 담을 수 없습니다.

그러므로, 원본 이미지의 크기를 변경한 후 설정된 변수나 필드에 포함되어야합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/7.webp" class="lazyload" width="100%" height="100%"/>

앞선 이미지를 회전할 경우, 액자의 크기를 변경해주지 않는다면 아래와 같은 현상이 발생합니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/8.webp" class="lazyload" width="100%" height="100%"/>

원본 이미지에서 반 시계방향으로 `45°` 회전하였을 경우, 각각의 모서리 부분이 잘려나가는 것을 확인 할 수 있습니다.

액자의 크기는 동일한 상태로 이미지를 회전하였을 경우, 이미지가 잘려나가거나 오류가 발생합니다.

또한, 의도하지 않은 이미지의 누락 현상이 발생합니다. 이를 해결하기 위해서 **액자의 크기도 재 설정해주어야합니다.**

<br>

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/9.webp" class="lazyload" width="100%" height="100%"/>

액자의 크기까지 재 설정한 경우, 정상적으로 이미지가 잘려나가지 않게 표시되는것을 확인할 수 있습니다. 

<br>
<br>

## 정밀도(Bit Depth)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/3.webp" class="lazyload" width="100%" height="100%"/>

다음으로는 `정밀도`입니다. `비트 깊이`, `색상 깊이`, `색 심도` 등과 동일한 의미를 갖습니다.

정밀도란 이미지가 얼마나 많은 색상을 표현할 수 있는지를 의미합니다. 정밀도가 높을 수록 더 많은 색상을 표현할 수 있어서 데이터의 폭이 넓어지고 더 자연스러운 이미지로 표시됩니다.

반대로 정밀도가 낮을수록 육안으로 확인할 수 없을 정도의 변형됩니다.

일반적으로 유효 비트가 많을수록 데이터의 처리 결과는 더 정밀해집니다.

여기서 `비트 (Bit)`의 값이 색상의 표현 개수를 설정합니다.

`1-Bit`의 경우, `0`과 `1`의 두 가지의 값만 가지게 되어 모든 색상을 `0`의 값을 지니는 색상과 `1`의 값을 지니는 색상으로만 표현됩니다.

중요한 점은 **두 가지의 색상이 아닌 두 가지의 값으로 표현할 수 있다는 의미입니다.**

`8-Bit`의 경우, **256** 가지의 값을 가질 수 있습니다.

`2x2x2x2x2x2x2x2=256`을 의미합니다.

256 가지의 방법으로 값을 표현할 수 있습니다.

최소 `8-Bit`를 가질 때 유의미한 데이터를 얻게되어 색상을 표현할 수 있습니다.

`8-Bit`의 정밀도를 사용할 경우, 흑백의 색상을 원할하게 표현할 수 있습니다. 주로, **GrayScale** 메소드에서 많이 활용합니다.

OpenCV에서는 `U8`의 값을 가장 많이 사용합니다.

`U8`은 `unsigned 8-bit intergers`를 의미합니다.

**unsigned**는 `부호 비트`를 제거해 저장 가능한 양수의 범위를 두 배로 늘리는 역할을 합니다.

`signed`의 경우에는 `-127~127`의 값을 표현할 수 있으며, `unsigned`의 경우, `0~255`의 값을 표현할 수 있습니다.

대부분의 색상 채널은 `0~255`의 값으로 색상을 표현합니다. 

<br>

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/10.webp" class="lazyload" width="100%" height="100%"/>

위의 이미지는 `1-bit`, `4-bit`, `8-bit`를 표현한 이미지입니다.

`1-bit`의 이미지는 `binary`한 이미지가 되며, `4-bit`의 이미지는 `저화질`의 이미지가 됩니다.

`8-bit`의 이미지는 `GrayScale`의 이미지가 됩니다. 이미지의 `정밀도`가 높을수록 더 고화질의 이미지가 되며, 데이터의 개수가 많아집니다.

처리하는 단계의 역할에 따라 적절한 정밀도를 선정해야합니다.

<br>
<br>

## 채널(Channel)

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/4.webp" class="lazyload" width="100%" height="100%"/>

마지막으로 `채널`은 그래픽스 이미지의 색상 정보를 포함하고 있습니다.

채널은 일반적으로 `Red`, `Blue`, `Green`과 추가적으로 `Alpha`가 존재합니다.

이외에도 `Hue`, `Saturation`, `Value` 등의 채널도 존재합니다.

색상을 표시할 때는 주로, `3 ~ 4` 채널의 값을 사용하고, 흑백의 이미지를 표현할 때는 `1` 채널을 사용합니다.

`3 ~ 4` 개의 채널을 가질 때는 `다 채널` 또는 `다중 채널`을 뜻합니다. `1` 개의 채널을 가질 때는 `단일 채널`을 뜻합니다.

색상 이미지 (RGB)에서 `Red`의 값만 추출한다해서 빨간색으로 표현되지는 않습니다.

그 이유는 **한 가지의 채널로만 색상을 표현**해야하기 때문입니다. 다음의 이미지에서 이미지를 채널별로 분리했을 때의 결과를 확인할 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/5.webp" class="lazyload" width="100%" height="100%"/>

`R의 성분`, `G의 성분`, `B의 성분`만 따로 뽑아내서 출력했지만 `흑백`으로 출력됩니다.

즉, **해당 성분에 가까울수록 하얀색으로 출력되고 아닌 값은 검은색으로 출력됩니다.**

<br>

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-3/6.webp" class="lazyload" width="100%" height="100%"/>

이미지에서 파란 부분의 색상 정보를 확인해보겠습니다. 보시는 바와 같이 아무리 파란색이라도 약간의 적색이나 녹색이 포함되어있습니다.

만약, 파란색의 색상만 출력하고 싶다면 채널을 `다 채널`로 사용하고 파란색의 성분을 가지는 블루 채널을 `마스크`로 씌우거나 `Hue`의 색상을 가져와야 파란색의 색상으로 출력할 수 있습니다.

간단하게 흑백이나 특정 색상 데이터를 가져온다면 채널은 `단일 채널`로 사용해야합니다.

많은 함수나 메소드에서 계산 시, 데이터의 양을 줄이고 정확도를 높이기 위하여 `단일 채널`로 계산을 진행합니다.

우리가 `OpenCV`에서 알고리즘을 적용할 때, 계산 이미지를 `Binary`나 `GrayScale`을 적용하는 이유입니다.

<br>

이미지의 속성 정보에 대한 이해는 매우 중요합니다.

세 가지 구성요소를 이해하지 못한다면 여러 함수나 알고리즘을 적용하는데 큰 어려움을 겪습니다.

위의 세 가지 구성요소는 `데이터`를 의미합니다. **데이터를 어떻게 정제하고 확장, 축소 시킬지의 여부가 알고리즘의 정확도에 큰 영향을 미칩니다.**

`전처리`나 `후처리` 작업을 진행할 때 큰 변수로 작용합니다.

<br>
<br>

* Writer by : 윤대희
