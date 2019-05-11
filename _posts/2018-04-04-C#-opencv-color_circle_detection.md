---
bg: "opencv.png"
layout: post
comments: true
title: "C#-OpenCV 예제 : 색상 원 검출"
crawlertitle: "C#-OpenCV 예제 : 색상 원 검출"
summary: "C#-OpenCV Color Circle Detection"
date: 2018-04-04
categories: exercise
tags: ['C#-OpenCV 예제']
author: 윤대희
star: true
---

### Color Circle Detection  ###
----------
[![1]({{ site.images }}/C/opencv/ex/ch2/1.jpg)]({{ site.images }}/C/opencv/ex/ch2/1.jpg)
**원을 검출하여 색상을 파악합니다.** 다양한 이미지나 동영상에서 색상이 포함된 원을 검출 할 수 있습니다.

동영상이나 이미지에서 `이진화(Binary)`와 `모폴로지(Morphology)`를 통하여 이미지를 검출용 이미지로 변환합니다.

`원 검출(Hough Transform Circles)`를 사용하여 원을 검출합니다.

검출이 완료되었다면 색상 검출을 위하여 `관심 영역(Region of Interest)`을 생성하여 개별 적용합니다.

관심 영역 위에 `HSV`를 적용한 후, `난수(Random)`을 생성하여 검출 이미지의 정확도를 파악합니다.


<br>
<br>
## Step 1 ##
----------
[![2]({{ site.images }}/C/opencv/ex/ch2/2.png)]({{ site.images }}/C/opencv/ex/ch2/2.png)

<br>

1. 이진화 적용
2. 모폴로지 & 블러

<br>

[이진화 바로가기][12강] <br>
[모폴로지 바로가기][27강] <br>
[블러 바로가기][13강]

<br>
<br>
## Step 2 ##
----------
[![3]({{ site.images }}/C/opencv/ex/ch2/3.png)]({{ site.images }}/C/opencv/ex/ch2/3.png)

<br>

1. 원 검출
2. 좌표 저장

<br>

[원 검출 바로가기][26강]

<br>
<br>
## Step 3 ##
----------
[![4]({{ site.images }}/C/opencv/ex/ch2/4.png)]({{ site.images }}/C/opencv/ex/ch2/4.png)

<br>

1. 좌표 불러오기
2. 관심 영역 설정

<br>

[관심 영역 바로가기][9강]

<br>
<br>
## Step 4 ##
----------
[![5]({{ site.images }}/C/opencv/ex/ch2/5.png)]({{ site.images }}/C/opencv/ex/ch2/5.png)

<br>

1. 구역 설정
2. 난수 생성
3. 색상 검출
4. 정확도 파악

<br>

[난수 바로가기][27강-2] <br>
[색상 검출 바로가기][15강]

<br>
<br>
## Step 5 ##
----------
[![6]({{ site.images }}/C/opencv/ex/ch2/6.png)]({{ site.images }}/C/opencv/ex/ch2/6.png)

<br>

결과 표시

<br>

[결과 표시 바로가기][17강]

<br>
<br>
## Step 6 ##
----------
[![7]({{ site.images }}/C/opencv/ex/ch2/7.png)]({{ site.images }}/C/opencv/ex/ch2/7.png)

<br>

HDC 적용

<br>

[HDC 바로가기][36강]

<br>
<br>
## Video File ##
----------
<video src="{{ site.images }}/C/opencv/ex/ch2/git.mp4" autoplay loop controls height="700"></video>




<br>
<br>

[12강]: https://076923.github.io/posts/C-opencv-12/
[27강]: https://076923.github.io/posts/C-opencv-27/
[13강]: https://076923.github.io/posts/C-opencv-13/

[26강]: https://076923.github.io/posts/C-opencv-26/

[9강]: https://076923.github.io/posts/C-opencv-9/

[27강-2]: https://076923.github.io/posts/C-27/
[15강]: https://076923.github.io/posts/C-opencv-15/

[17강]: https://076923.github.io/posts/C-opencv-17/

[36강]: https://076923.github.io/posts/C-opencv-36/
