---
layout: post
title: "Python OpenCV 강좌 : 제 41강 - 색상 맵"
tagline: "Python OpenCV Remapping"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV applyColorMap, OpenCV Lookup table
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-41/
comments: true
toc: true
---

## 색상 맵(Color Map)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-41/1.webp){: width="100%" height="100%"}

`색상 맵(Color Map)`은 입력 이미지에 **순람표(Lookup table)** 구조로 이루어진 데이터를 적용합니다.

주로 데이터를 시각화하기 위해 사용되며, 색상의 분포표로 데이터를 쉽게 확인할 수 있습니다.

픽셀값이 1:1로 매칭되기 때문에 **선형 구조나 비선형 구조로도 데이터를 매핑해 표현할 수 있습니다.**

<br>
<br>

## 메인 코드 (1)

{% highlight Python %}

import cv2

src = cv2.imread("beach.jpg")
dst = cv2.applyColorMap(src, cv2.COLORMAP_OCEAN)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

dst = cv2.applyColorMap(src, cv2.COLORMAP_OCEAN)

{% endhighlight %}

`색상 맵 적용 함수(cv2.applyColorMap)`를 활용해 원본 이미지에 특정 색상 맵 배열이 적용된 이미지를 생성합니다.

`dst = cv2.applyColorMap(src, colormap)`은 `입력 이미지(src)`에 `색상 맵(colormap)`을 적용한 `결과 이미지(dst)`를 반환합니다.

색상 맵 적용 함수는 색상 맵 플래그가 아닌, `사용자 정의 색상 맵(userColor)`을 활용해 이미지를 적용할 수 있습니다.

순람표는 `Numpy`를 활용해 다음과 같이 생성할 수 있습니다.

<br>

{% highlight Python %}

userColor_8UC1 = np.linspace(0, 255, num=256, endpoint=True, retstep=False, dtype=np.uint8).reshape(256, 1)

userColor_8UC3 = np.linspace(0, 255, num=256 * 3, endpoint=True, retstep=False, dtype=np.uint8).reshape(256, 1, 3)

{% endhighlight %}

사용자 정의 색상 맵은 `순람표(Lookup table)`의 구조만 적용이 가능합니다.

단일 채널 순람표는 `CV_8UC1` 형식의 **[256, 1]** 형태를 갖습니다.

다중 채널 순람표는 `CV_8UC3` 형식의 **[256, 1, 3]** 형태를 갖습니다.

`색상 맵(colormap)` 매개변수 대신에 `사용자 정의 색상 맵(userColor)`을 적용합니다.

> [색상 맵 적용 함수 자세히 알아보기](https://076923.github.io/docs/applyColorMap)

> [색상 맵 플래그 자세히 알아보기](https://076923.github.io/docs/ColormapTypes)

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-41/2.webp){: width="100%" height="100%"}
