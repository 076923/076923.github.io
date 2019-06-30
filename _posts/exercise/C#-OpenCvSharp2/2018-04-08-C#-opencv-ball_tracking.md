---
layout: post
title: "C#-OpenCV 예제 : 공 추적"
tagline: "C#-OpenCV Ball Tracking"
image: /assets/images/Clogo.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ["C#-OpenCvSharp-Exercise"]
keywords: Visual Studio, OpenCV, OpenCvSharp2, Ball Tracking
ref: C#
category: exercise
permalink: /exercise/C-opencv-ball_tracking/
comments: true
---

### Ball Tracking ###
----------

![1]({{ site.images }}/assets/images/C/opencv/ex/ch3/1.png)

**영상이나 이미지에서 공을 추적합니다.** 움직이는 물체를 검출 할 수 있습니다.

카메라나 동영상에서 움직이는 소형 물체를 추적할 수 있습니다.

`이진화(Binary)`를 적용하여 검출하기 쉬운 상태로 변환합니다.

`모폴로지(Morphology)`와 `비트연산(Bitwise)` 활용하여 배경을 삭제합니다.

배경이 삭제된 이미지에서 `라벨링(Labeling)`을 통하여 물체를 검출합니다.

<br>
<br>

## Step 1 ##
----------

![2]({{ site.images }}/assets/images/C/opencv/ex/ch3/2.png)

1. 그레이스케일 적용
2. 블러 & 이진화 적용

<br>

[그레이스케일 바로가기][10강] <br>
[블러 바로가기][13강] <br>
[이진화 바로가기][12강] <br>

<br>
<br>

## Step 2 ##
----------

![3]({{ site.images }}/assets/images/C/opencv/ex/ch3/3.png)

배경 적용

1. 모폴로지 적용
2. 모폴로지 연산 적용

<br>

[모폴로지 바로가기][27강] <br>
[모포롤지 연산 바로가기][28강] <br>

<br>
<br>

## Step 3 ##
----------

![4]({{ site.images }}/assets/images/C/opencv/ex/ch3/4.png)

배경 병합

1. 비트 연산 (Or)

<br>

[비트 연산 바로가기][42강]

<br>
<br>

## Step 4 ##
----------

![5]({{ site.images }}/assets/images/C/opencv/ex/ch3/5.png)

<br>

물체 연산

1. 블러 & 이진화 적용
2. 모폴로지 연산

<br>

[블러 바로가기][13강] <br>
[이진화 바로가기][12강] <br>
[이진화 바로가기][12강] <br>

<br>
<br>

## Step 5 : Way - 1 ##
----------
![6]({{ site.images }}/assets/images/C/opencv/ex/ch3/6.png)

Way 1 : 물체 남기기

1. 비트 연산 (And)

<br>

[비트 연산 바로가기][42강]

<br>
<br>

## Step 5 : Way - 2 ##
----------
![7]({{ site.images }}/assets/images/C/opencv/ex/ch3/7.png)

Way 2 : 물체 추적

1. 라벨링
2. 결과 표시

<br>

[라벨링 바로가기][32강] <br>
[결과 표시 바로가기][17강]

<br>
<br>

## Video File ##
----------

## <center>Way - 1</center> ##

<video src="{{ site.images }}/assets/images/C/opencv/ex/ch3/way1.mp4" autoplay loop controls height="700"></video>

<br>

## <center>Way - 2</center> ##

<video src="{{ site.images }}/assets/images/C/opencv/ex/ch3/way2.mp4" autoplay loop controls height="700"></video>


[10강]: https://076923.github.io/posts/C-opencv-10/
[12강]: https://076923.github.io/posts/C-opencv-12/
[13강]: https://076923.github.io/posts/C-opencv-13/

[27강]: https://076923.github.io/posts/C-opencv-27/
[28강]: https://076923.github.io/posts/C-opencv-28/

[42강]: https://076923.github.io/posts/C-opencv-42/

[32강]: https://076923.github.io/posts/C-opencv-32/

[17강]: https://076923.github.io/posts/C-opencv-17/
