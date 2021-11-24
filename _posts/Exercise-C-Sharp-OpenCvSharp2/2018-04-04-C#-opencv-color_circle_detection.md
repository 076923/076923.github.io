---
layout: post
title: "OpenCvSharp2 예제 : 원의 색상 검출"
tagline: "C# OpenCvSharp2 Color Circle Detection"
image: /assets/images/csharp.svg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ["OpenCvSharp2 : Exercise"]
keywords: Visual Studio, OpenCV, OpenCvSharp2, Color Circle Detection
ref: Exercise-C#-OpenCvSharp2
category: Exercise
permalink: /exercise/C-opencv-color_circle_detection/
comments: true
toc: true
---

## Color Circle Detection

![1]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/1.jpg)

**원을 검출하여 색상을 파악합니다.** 다양한 이미지나 동영상에서 색상이 포함된 원을 검출 할 수 있습니다.

동영상이나 이미지에서 `이진화(Binary)`와 `모폴로지(Morphology)`를 통하여 이미지를 검출용 이미지로 변환합니다.

`원 검출(Hough Transform Circles)`를 사용하여 원을 검출합니다.

검출이 완료되었다면 색상 검출을 위하여 `관심 영역(Region of Interest)`을 생성하여 개별 적용합니다.

관심 영역 위에 `HSV`를 적용한 후, `난수(Random)`을 생성하여 검출 이미지의 정확도를 파악합니다.

<br>
<br>

## Step 1

![2]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/2.png)

1. 이진화 적용
2. 모폴로지 & 블러

- [이진화 바로가기][12강]

- [모폴로지 바로가기][27강]

- [블러 바로가기][13강]

<br>
<br>

## Step 2

![3]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/3.png)

1. 원 검출
2. 좌표 저장

- [원 검출 바로가기][26강]

<br>
<br>

## Step 3

![4]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/4.png)

1. 좌표 불러오기
2. 관심 영역 설정

- [관심 영역 바로가기][9강]

<br>
<br>

## Step 4

![5]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/5.png)

1. 구역 설정
2. 난수 생성
3. 색상 검출
4. 정확도 파악

- [난수 바로가기][27강-2]

- [색상 검출 바로가기][15강]

<br>
<br>

## Step 5

![6]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/6.png)

- 결과 표시

- [결과 표시 바로가기][17강]

<br>
<br>

## Step 6

![7]({{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/7.png)

- HDC 적용

- [HDC 바로가기][36강]

<br>
<br>

## Video File

<video src="{{ site.images }}/assets/posts/Exercise/C-Sharp/OpenCvSharp2/color_circle_detection/git.mp4" autoplay loop controls height="700"></video>

[12강]: https://076923.github.io/posts/C-opencv-12/
[27강]: https://076923.github.io/posts/C-opencv-27/
[13강]: https://076923.github.io/posts/C-opencv-13/

[26강]: https://076923.github.io/posts/C-opencv-26/

[9강]: https://076923.github.io/posts/C-opencv-9/

[27강-2]: https://076923.github.io/posts/C-27/
[15강]: https://076923.github.io/posts/C-opencv-15/

[17강]: https://076923.github.io/posts/C-opencv-17/

[36강]: https://076923.github.io/posts/C-opencv-36/
