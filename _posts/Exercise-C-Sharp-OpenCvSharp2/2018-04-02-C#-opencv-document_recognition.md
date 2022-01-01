---
layout: post
title: "OpenCvSharp2 예제 : 명함(문서) 인식"
tagline: "C# OpenCvSharp2 Document Recognition"
image: /assets/images/csharp.svg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ["OpenCvSharp2 : Exercise"]
keywords: Visual Studio, OpenCV, OpenCvSharp2, Document Recognition, Corner Detector, Dot Product
ref: Exercise-C#-OpenCvSharp2
category: Exercise
permalink: /exercise/C-opencv-document_recognition/
comments: true
toc: true
---

## Document Recognition

![1]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/document_recognition/1.webp){: width="100%" height="100%"}

**명함이나 문서의 크기를 변환 후 해당 문자를 인식합니다.**

동영상이나 이미지에서 `코너 검출(Corner Detector)`하여 `벡터의 내적(Dot Product)`을 사용해 사각형을 검출합니다.

이후, `기하학적 변환(Warp Perspective)`을 이용하여 검출하기 쉬운 이미지로 변경합니다.

검출용 이미지를 `Tesseract 라이브러리`를 이용하여 문자를 검출하며 `foreach`문과 `유니코드` 등을 이용하여 문자들에서 유의미한 데이터(이름, 전화번호, 상호명 등)를 얻어냅니다.

<br>
<br>

## Step 1 ##

![2]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/document_recognition/2.webp){: width="100%" height="100%"}

1. 관심 채널
2. 이진화 적용 & 캐니 엣지
3. 코너 검출
4. 벡터 내적을 통하여 사각형 파악

- [이진화 바로가기][12강]

- [케니 엣지 바로가기][14강]

- [코너 검출 바로가기][21강]

<br>
<br>

## Step 2

![3]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/document_recognition/3.webp){: width="100%" height="100%"}

1. 검출 좌표 저장
2. 기하학적 변환

- [기하학적 변환 바로가기][18강]

<br>
<br>

## Step 3

![4]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/document_recognition/4.webp){: width="100%" height="100%"}

1. tesseract 라이브러리
2. 영문자 판독

- [tesseract 라이브러리 바로가기][2강]

<br>
<br>

## Step 4

![5]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/document_recognition/5.webp){: width="100%" height="100%"}

1. foreach문
2. 아스키 코드 & 유니 코드

- [foreach문 바로가기][11강]

[12강]: https://076923.github.io/posts/C-opencv-12/
[14강]: https://076923.github.io/posts/C-opencv-14/
[21강]: https://076923.github.io/posts/C-opencv-21/

[18강]: https://076923.github.io/posts/C-opencv-18/
[2강]: https://076923.github.io/posts/C-tesseract-2/

[11강]: https://076923.github.io/posts/C-11/
