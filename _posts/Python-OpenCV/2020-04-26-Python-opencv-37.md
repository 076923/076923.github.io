---
layout: post
title: "Python OpenCV 강좌 : 제 37강 - 템플릿 매칭"
tagline: "Python OpenCV Template Matching"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Template Matching, OpenCV matchTemplate, OpenCV cv2.TM_SQDIFF, OpenCV cv2.TM_SQDIFF_NORMED, OpenCV cv2.TM_CCORR, OpenCV cv2.TM_CCORR_NORMED, OpenCV cv2.TM_CCOEFF, OpenCV cv2.TM_CCOEFF_NORMED
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-37/
comments: true
toc: true
---

## 템플릿 매칭(Template Matching)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/1.webp){: width="100%" height="100%"}

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/2.webp){: width="100%" height="100%"}

`템플릿 매칭`은 **원본 이미지**에서 **템플릿 이미지**와 일치하는 영역을 찾는 알고리즘입니다.

**원본 이미지** 위에 **템플릿 이미지**를 놓고 조금씩 이동해가며 이미지 끝에 도달할 때 까지 비교해 찾아갑니다.

이 방식을 통해, **템플릿 이미지**와 동일하거나, 가장 유사한 영역을 **원본 이미지**에서 검출합니다. 

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("hats.png", cv2.IMREAD_GRAYSCALE)
templit = cv2.imread("hat.png", cv2.IMREAD_GRAYSCALE)
dst = cv2.imread("hats.png")

result = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
x, y = minLoc
h, w = templit.shape

dst = cv2.rectangle(dst, (x, y), (x +  w, y + h) , (0, 0, 255), 1)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.imread("hats.png", cv2.IMREAD_GRAYSCALE)
templit = cv2.imread("hat.png", cv2.IMREAD_GRAYSCALE)
dst = cv2.imread("hats.png")

{% endhighlight %}

`원본 이미지(src)`와 `템플릿 이미지(templit)`을 선언합니다.

탬플릿 매칭은 `그레이스케일` 이미지를 사용하므로, `cv2.IMREAD_GRAYSCALE`를 적용합니다.

결과를 표시할 `결과 이미지(dst)`를 선언합니다.

<br>

{% highlight Python %}

result = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED)

{% endhighlight %}

`템플릿 매칭 함수(cv2.matchTemplate)`로 템플릿 매칭을 적용합니다.

`cv2.matchTemplate(원본 이미지, 템플릿 이미지, 템플릿 매칭 플래그)`을 의미합니다.

`원본 이미지`와 `템플릿 이미지`는 8비트의 **단일 채널 이미지**를 사용합니다.

`템플릿 매칭 플래그`는 템플릿 매칭에 사용할 연산 방법을 설정합니다.

<br>

### 템플릿 매칭 플래그

|   플래그   |               수식               |
|:----------:|:--------------------------------:|
| cv2.TM_SQDIFF | ![3]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/3.webp){: width="100%" height="100%"} |
| cv2.TM_SQDIFF_NORMED | ![4]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/4.webp){: width="100%" height="100%"} |
| cv2.TM_CCORR | ![5]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/5.webp){: width="100%" height="100%"} |
| cv2.TM_CCORR_NORMED | ![6]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/6.webp){: width="100%" height="100%"} |
| cv2.TM_CCOEFF | ![7]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/7.webp){: width="100%" height="100%"}![9]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/9.webp){: width="100%" height="100%"} |
| cv2.TM_CCOEFF_NORMED | ![8]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/8.webp){: width="100%" height="100%"} |

<br>

반환되는 `결괏값(dst)`은 32비트의 **단일 채널 이미지**로 반환됩니다.

또한, 배열의 크기는 `W - w + 1`, `H - h + 1`의 크기를 갖습니다.

`(W, H)`는 원본 이미지의 크기이며, `(w, h)`는 템플릿 이미지의 크기입니다.

결괏값이 위와 같은 크기를 갖는 이유는 원본 이미지에서 템플릿 이미지를 일일히 비교하기 때문입니다.

예를 들어, `4×4` 크기의 원본 이미지와 `3×3` 크기의 템플릿 이미지가 있다면 아래의 그림과 같이 표현할 수 있습니다.

<br>

![10]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/10.png)

<br>

총 **4번의 비교**를 진행할 수 있으며, 이를 배열로 옮긴다면 `2×2` 크기를 갖게 됩니다.

수식으로 다시 표현한다면, $$ (W - w + 1, H - h + 1) = (4 - 3 + 1, 4 - 3 + 1) = (2, 2) $$가 됩니다. 

<br>

{% highlight Python %}

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
x, y = minLoc
h, w = templit.shape

{% endhighlight %}

`결괏값(dst)`에서 가장 유사한 부분을 찾기 위해 `최소/최대 위치 함수(cv2.minMaxLoc)`로 검출값을 찾습니다.

`최소/최대 위치 함수`는 **최소 포인터**, **최대 포인터**, **최소 지점**, **최대 지점**을 반환합니다.

검출 위치의 좌측 상단 모서리 좌표는 `최소 지점(minLoc)`이나 `최대 지점(maxLoc)`에 위치합니다.

`템플릿 이미지`를 일일히 비교하므로, 이미지 크기는 `템플릿 이미지`와 동일합니다.

- Tip : `cv2.TM_SQDIFF`, `cv2.TM_SQDIFF_NORMED`는 `최소 지점(minLoc)`이 검출된 위치입니다.

- Tip : `cv2.TM_CCORR`, `cv2.TM_CCORR_NORMED`, `cv2.TM_CCOEFF`, `cv2.TM_CCOEFF_NORMED`는 `최대 지점(maxLoc)`이 검출된 위치입니다.

<br>

{% highlight Python %}

dst = cv2.rectangle(dst, (x, y), (x +  w, y + h) , (0, 0, 255), 1)

{% endhighlight %}

검출된 결과를 `결과 이미지(dst)`위에 표시합니다.

<br>
<br>

## 출력 결과

![11]({{ site.images }}/assets/posts/Python/OpenCV/lecture-37/11.png)
