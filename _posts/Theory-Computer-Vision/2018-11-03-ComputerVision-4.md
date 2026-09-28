---
layout: post
title: "Computer Vision Theory : 이미지와 동영상의 이해"
tagline: "Understanding images and videos"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Understanding images and videos, Codec, Container, FourCC, FPS
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-4/
comments: true
toc: true
---

## 이미지와 동영상의 이해

<img data-src="{{ site.images }}/assets/posts/Theory/ComputerVision/lecture-4/1.webp" class="lazyload" width="100%" height="100%"/>

**이미지 파일 형식**은 수백 가지의 종류가 존재합니다. **OpenCV**에서는 `래스터 그래픽스` 이미지 파일 포맷을 쉽게 불러올 수 있습니다.

가장 많이 사용되는 파일 포맷으로는 `BMP (Bitmap)`, `JPEG (Joint Photographic Experts Group)`, `GIF (Graphics Interchange Format)`, `PNG (Portable Network Graphics)` 등이 존재합니다. 동영상의 경우에는 `AVI (Audio Video Interleave)`, `MP4 (MPEG-4 Part 14)`, `WMV (Windows Media Video)` 등이 존재합니다.

* `BMP` : **1~24 Bit**, 무압축 또는 낮은 압축률
* `JPEG` : **8/24 Bit**, 손실 압축, 압축률 높음
* `GIF` : **1~8 Bit**(최대 256색), 무손실 압축
* `PNG` : **1~48 Bit**, 무손실 압축
* `AVI` : **다양한 코덱으로 인코딩 가능**
* `MP4` : 고압축, **비디오 품질 높음**
* `WMV` : 고압축, **비디오 품질 낮음**
 
<br>
<br>

## GIF

래스터 그래픽스의 경우, **OpenCV**를 통하여 쉽게 불러올 수 있는데 `GIF` 이미지의 경우에는 **프레임**이 존재합니다. GIF의 경우, 움직이는 이미지이므로 동영상으로 간주하여 작업해야 합니다. 또한, 프레임이 없는 GIF의 이미지도 프레임이 하나인 동영상으로 간주해야 합니다. 그러므로 GIF 확장자를 **OpenCV**에서 처리할 경우, `VideoCapture()` 함수를 사용하거나, `Pillow` 라이브러리 등을 이용해야합니다.

<br>
<br>

## 동영상의 프레임

동영상은 **이미지의 연속**입니다. 동영상은 멈추어 있는 사진들이 연속되어 움직이는 동영상이 됩니다. 이 각각의 이미지를 `프레임`이라 부르며, **프레임들을 초당 몇 장의 이미지를 보여주냐에 따라 동영상의 자연스러움이 결정됩니다.** 동영상에 이미지 프로세싱을 적용할 경우, 모든 프레임에 동일한 알고리즘을 적용하게 됩니다.

그러므로 동영상을 처리할 경우, 프레임 속도에 따라 적절한 알고리즘을 설계해야합니다. 결국 원할한 알고리즘의 구현을 위해선 `FPS (Frames Per Second)`의 간단한 이해가 필요합니다.

<br>
<br>

## FPS

`FPS`는 영상이 바뀌는 속도를 의미합니다. 즉, **화면의 부드러움을 의미합니다.** 화면이 부드럽게 처리되면서 함수를 적용해야합니다. 알고리즘의 처리 속도가 오래걸리는 경우, `FPS`의 값을 적절히 조정하거나 알고리즘을 수정해야합니다. 좋은 알고리즘을 설계하지 못할 경우, `FPS`의 속도를 따라가지 못하여 **오류**나 **지연**이 발생하게 됩니다.

또한, 동영상을 처리함에 있어서 `이미지의 크기`, `정밀도`, `채널`의 값을 적절하게 사용한다면 높은 `FPS`의 값을 가지는 동영상에서도 수준 높은 알고리즘을 구현할 수 있습니다.

## 코덱과 컨테이너

앞서 동영상 형식으로 언급한 `AVI`, `MP4`, `WMV`는 정확히 말하면 영상을 압축하는 방식이 아니라, 압축된 영상과 음성을 하나로 묶어 담는 `컨테이너(Container)`입니다. 실제 압축과 복원을 담당하는 것은 `코덱(Codec)`이며, 코덱은 `COmpressor`와 `DECompressor`를 합친 말입니다.

그러므로 같은 `MP4` 확장자라도 내부에 `H.264`로 압축된 영상이 들어 있을 수도 있고 `H.265`나 `VP9`로 압축된 영상이 들어 있을 수도 있습니다. 확장자가 같다고 해서 동일한 방식으로 처리되는 것이 아니므로, 영상을 다룰 때는 컨테이너와 코덱을 구분해서 이해해야 합니다.

OpenCV에서 동영상을 저장할 때 `FourCC(Four Character Code)`라는 네 글자 코드로 코덱을 지정하는 이유도 여기에 있습니다. `VideoWriter`에 `DIVX`, `XVID`, `MJPG`, `mp4v` 등의 코드를 전달해 어떤 방식으로 압축할지 명시합니다.

- Tip : 동영상을 읽을 때 `VideoCapture`가 파일을 열지 못하거나 프레임이 비어 있는 경우, 경로 문제가 아니라 **해당 코덱이 시스템에 설치돼 있지 않은 경우**가 많습니다.

<br>
<br>

<br>
<br>

* Writer by : 윤대희
