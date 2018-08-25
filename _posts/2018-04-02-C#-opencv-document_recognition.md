---
bg: "opencv.png"
layout: post
comments: true
title: "C#-OpenCV 예제 : 명함(문서) 인식"
crawlertitle: "C#-OpenCV 예제 : 명함(문서) 인식"
summary: "C#-OpenCV Document Recognition"
date: 2018-04-02
categories: exercise
tags: ['C#-OpenCV 예제']
author: 윤대희
star: true
---

### Document Recognition ###
----------
[![1]({{ site.images }}/C/opencv/ex/ch1/1.png)]({{ site.images }}/C/opencv/ex/ch1/1.png)
**명함이나 문서의 크기를 변환 후 해당 문자를 인식합니다.**

동영상이나 이미지에서 `코너 검출(Corner Detector)`하여 `벡터의 내적(Dot Product)`을 사용해 사각형을 검출합니다.

이 후, `기하학적 변환(Warp Perspective)`을 이용하여 검출하기 쉬운 이미지로 변경합니다.

검출용 이미지를 `Tesseract 라이브러리`를 이용하여 문자를 검출하며 `foreach`문과 `유니코드` 등을 이용하여 문자들에서 유의미한 데이터(이름, 전화번호, 상호명 등)를 얻어냅니다.

<br>
<br>
## Step 1 ##
----------
[![2]({{ site.images }}/C/opencv/ex/ch1/2.png)]({{ site.images }}/C/opencv/ex/ch1/2.png)

<br>

1. 관심 채널
2. 이진화 적용 & 캐니 엣지
3. 코너 검출
4. 벡터 내적을 통하여 사각형 파악

<br>

[이진화 바로가기][12강] <br>
[케니 엣지 바로가기][14강] <br>
[코너 검출 바로가기][21강]

<br>
<br>
## Step 2 ##
----------
[![3]({{ site.images }}/C/opencv/ex/ch1/3.png)]({{ site.images }}/C/opencv/ex/ch1/3.png)

<br>

1. 검출 좌표 저장
2. 기하학적 변환

<br>

[기하학적 변환 바로가기][18강]

<br>
<br>
## Step 3 ##
----------
[![4]({{ site.images }}/C/opencv/ex/ch1/4.png)]({{ site.images }}/C/opencv/ex/ch1/4.png)

<br>

1. tesseract 라이브러리
2. 영문자 판독

<br>

[tesseract 라이브러리 바로가기][2강]

<br>
<br>
## Step 4 ##
----------
[![5]({{ site.images }}/C/opencv/ex/ch1/5.png)]({{ site.images }}/C/opencv/ex/ch1/5.png)

<br>

1. foreach문
2. 아스키 코드 & 유니 코드

<br>

[foreach문 바로가기][11강]

<br>
<br>

[12강]: https://076923.github.io/posts/C-opencv-12/
[14강]: https://076923.github.io/posts/C-opencv-14/
[21강]: https://076923.github.io/posts/C-opencv-21/

[18강]: https://076923.github.io/posts/C-opencv-18/
[2강]: https://076923.github.io/posts/C-tesseract-2/

[11강]: https://076923.github.io/posts/C-11/
