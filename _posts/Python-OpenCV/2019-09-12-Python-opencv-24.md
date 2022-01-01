---
layout: post
title: "Python OpenCV 강좌 : 제 24강 - 블록 껍질"
tagline: "Python OpenCV Convex Hull"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV ConvexHull, OpenCV Contours ConvexHull
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-24/
comments: true
toc: true
---

## 블록 껍질(Convex Hull)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-24/1.webp){: width="100%" height="100%"}

`윤곽선(points, contours)`의 **경계면을 둘러싸는 다각형**을 구하는 알고리즘입니다.

반환되는 결과는 윤곽선 검출 결과와 동일한 형식을 띄며, `스크랜스키(Sklansky) 알고리즘`을 이용해 입력된 좌표들의 볼록한 외곽을 찾습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/convex.png")
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in contours:
    hull = cv2.convexHull(i, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

for i in contours:
    hull = cv2.convexHull(i, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)

{% endhighlight %}

`cv2.convexHull()`를 활용해 윤곽선에서 **블록 껍질**을 검출합니다.

`cv2.convexHull(윤곽선, 방향)`을 의미합니다.

`윤곽선`은 윤곽선 검출 함수에서 반환되는 구조를 사용합니다.

`방향`은 검출된 볼록 껍질의 볼록점들의 인덱스 순서를 의미합니다.

**블록 껍질 함수**는 단일 형태에서만 검출이 가능합니다.

그러므로, 반복문을 활용해 단일 형태의 윤곽선 구조에서 블록 껍질을 검출합니다.

- Tip : `윤곽선` 구조는 윤곽선 검출 함수의 반환값과 형태가 동일하다면, 임의의 배열에서도 검출이 가능합니다.

- Tip : `방향`이 `True`라면 **시계 방향**, `False`라면 **반시계 방향**으로 정렬됩니다.

<br>
<br>

## 추가 정보

블록 껍질 검출은 `스크랜스키(Sklansky) 알고리즘`을 사용합니다.

경계 사각형의 **정점(Vertex)**을 검출합니다.

경계면을 둘러싸는 다각형은 경계 사각형 내부에 포함되며, 해당 정점을 볼록점으로 사용합니다.

영역 내부에도 다양한 윤곽점들이 존재하므로 여기서 볼록 껍질을 이루는 **볼록점들을 선별해서 선택합니다.**

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-24/2.webp){: width="100%" height="100%"}
