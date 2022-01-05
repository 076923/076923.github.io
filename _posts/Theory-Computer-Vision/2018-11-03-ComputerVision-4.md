---
layout: post
title: "Computer Vision Theory : 이미지와 동영상의 이해"
tagline: "Understanding images and videos"
image: /assets/images/theory.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['ComputerVision']
keywords: Computer Vision, OpenCV, Understanding images and videos
ref: Theory-ComputerVision
category: Theory
permalink: /posts/ComputerVision-4/
comments: true
toc: true
---

## 이미지와 동영상의 이해

![1]({{ site.images }}/assets/posts/Theory/ComputerVision/lecture-4/1.webp){:class="lazyload" width="100%" height="100%"}

**이미지 파일 형식**은 수백 가지의 종류가 존재합니다. **OpenCV**에서는 `래스터 그래픽스` 이미지 파일 포맷을 쉽게 불러올 수 있습니다.

가장 많이 사용되는 파일 포맷으로는 `BMP (Bitmap)`, `JPEG (Joint Photographic Experts Group)`, `GIF (Graphics Interchange Format)`, `PNG (Portable Network Graphics)` 등이 존재합니다. 동영상의 경우에는 `AVI (Audio Video Interleave)`, `MP4 (MPEG-4 Part 14)`, `WMV (Windows Media Video)` 등이 존재합니다.

* `BMP` : **1~24 Bit**, 압축율 낮음
* `JPEG` : **1~24 Bit**, 압축율 높음
* `GIF` : **1~8 Bit**, 무손실 압축
* `PNG` : **1~48 Bit**, 무손실 압축
* `AVI` : **다양한 코덱으로 인코딩 가능**
* `MP4` : 고압축, **비디오 품질 높음**
* `WMV` : 고압축, **비디오 품질 낮음**
 
<br>
<br>

## GIF

래스터 그래픽스의 경우, **OpenCV**를 통하여 쉽게 불러올 수 있는데 `GIF` 이미지의 경우에는 **프레임**이 존재합니다.

GIF의 경우, 움직이는 이미지이므로 동영상으로 간주하여 작업해야 합니다.

또한, 프레임이 없는 GIF의 이미지도 프레임이 하나인 동영상으로 간주해야 합니다.

그러므로 GIF 확장자를 **OpenCV**에서 처리할 경우, `VideoCapture()` 함수를 사용하거나, `pilow` 라이브러리 등을 이용해야합니다.

<br>
<br>

## 동영상의 프레임

동영상은 **이미지의 연속**입니다.

동영상은 멈추어 있는 사진들이 연속되어 움직이는 동영상이 됩니다.

이 각각의 이미지를 `프레임`이라 부르며, **프레임들을 초당 몇 장의 이미지를 보여주냐에 따라 동영상의 자연스러움이 결정됩니다.**

동영상에 이미지 프로세싱을 적용할 경우, 모든 프레임에 동일한 알고리즘을 적용하게 됩니다.

그러므로 동영상을 처리할 경우, 프레임 속도에 따라 적절한 알고리즘을 설계해야합니다. 결국 원할한 알고리즘의 구현을 위해선 `FPS (Frame Per Second)`의 간단한 이해가 필요합니다. 

<br>
<br>

## FPS

`FPS`는 영상이 바뀌는 속도를 의미합니다.

즉, **화면의 부드러움을 의미합니다.** 화면이 부드럽게 처리되면서 함수를 적용해야합니다.

알고리즘의 처리 속도가 오래걸리는 경우, `FPS`의 값을 적절히 조정하거나 알고리즘을 수정해야합니다.

좋은 알고리즘을 설계하지 못할 경우, `FPS`의 속도를 따라가지 못하여 **오류**나 **지연**이 발생하게 됩니다.

또한, 동영상을 처리함에 있어서 `이미지의 크기`, `정밀도`, `채널`의 값을 적절하게 사용한다면 높은 `FPS`의 값을 가지는 동영상에서도 수준 높은 알고리즘을 구현할 수 있습니다. 

<br>
<br>

* Writer by : 윤대희
