---
layout: post
title: "Python OpenCV 강좌 : 제 35강 - 트랙 바"
tagline: "Python OpenCV Track Bar"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Track Bar, OpenCV namedWindow, OpenCV createTrackbar, OpenCV setTrackbarPos, OpenCV getTrackbarPos
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-35/
comments: true
---

## 트랙 바(Track Bar) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch35/1.jpg)
`트랙 바`란 스크롤 바의 하나로, 슬라이더 바의 형태를 갖고 있습니다.

트랙 바는 일정 범위 내의 값을 변경할 때 사용하며, 적절한 임곗값을 찾거나 변경하기 위해 사용합니다.

OpenCV의 트랙 바는 **생성된 윈도우 창에 트랙바를 부착**해 사용할 수 있습니다.


<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

def onChange(pos):
    pass

src = cv2.imread("cherryblossom.jpg", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Trackbar Windows")

cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)

while cv2.waitKey(1) != ord('q'):

    thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
    maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")

    _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

    cv2.imshow("Trackbar Windows", binary)

cv2.destroyAllWindows()
    
{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

src = cv2.imread("cherryblossom.jpg", cv2.IMREAD_GRAYSCALE)

{% endhighlight %}

`원본 이미지(src)`를 선언하고, 그레이스케일을 적용합니다.

<br>
<br>

{% highlight Python %}

cv2.namedWindow("Trackbar Windows")

{% endhighlight %}

트랙 바를 윈도우 창에 부착하기 위해선, 미리 윈도우 창에 생성된 상태여야 합니다.

`윈도우 창 생성 함수(cv2.namedWindow)`로 윈도우 창을 생성합니다.

`cv2.namedWindow("윈도우 창 제목")`을 사용해 윈도우 창을 생성합니다.

여기서, `윈도우 창 제목`은 변수와 같은 기능을 합니다.

<br>
<br>

{% highlight Python %}

cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

{% endhighlight %}

<br>

윈도우 창이 생성되었다면, 트랙 바를 생성합니다.

`트랙 바 생성 함수(cv2.createTrackbar)`로 트랙 바를 생성합니다.

`cv2.createTrackbar("트랙 바 이름", "윈도우 창 제목", 최솟값, 최댓값, 콜백 함수)`을 사용해 트랙 바를 생성합니다.

`트랙 바 이름`은 트랙 바의 명칭이며, `윈도우 창 제목`과 같이 변수와 비슷한 역할을 합니다.

`윈도우 창 제목`은 트랙 바를 부착할 윈도우 창을 의미합니다.

`최솟값`과 `최댓값`은 트랙 바를 조절할 때 사용할 최소/최대 값을 의미합니다.

`콜백 함수`는 트랙 바의 바를 조절할 때 위치한 값을 전달합니다.

`onChange` 함수의 `pos`는 현재 발생한 트랙 바 값을 반환합니다.

특별한 이벤트를 처리하지 않는다면, 함수의 반환값에 `pass`나 `return`를 사용하거나 `lambda` 함수로 아무 작업을 하지 않습니다.

<br>
<br>

{% highlight Python %}

cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)

{% endhighlight %}

`트랙 바 값 설정 함수(cv2.setTrackbarPos)`로 트랙 바의 값을 설정합니다.

`cv2.setTrackbarPos("트랙 바 이름", "윈도우 창 제목", 설정값)`을 사용해 트랙 바의 값을 설정합니다.

`트랙 바 이름`, `윈도우 창 제목`은 앞서 설명한 역할과 동일합니다.

`설정값`은 초기에 할당된 값이나, 특정 조건 등을 만족했을 때 강제로 할당할 값을 설정합니다.

일반적으로, 초깃값을 할당할 때 사용합니다.

<br>
<br>

{% highlight Python %}

while cv2.waitKey(1) != ord('q'):
  ...

cv2.destroyAllWindows()

{% endhighlight %}

`반복문(while)`을 활용해, 지속적으로 화면을 갱신합니다.

`cv2.waitKey(1)`은 `1ms`마다, 키보드 이벤트를 감지합니다.

`ord('q')`는 문자 `q`를 아스키 코드 값으로 변경합니다.

즉, 1ms 마다 값이 q가 눌러졌는지 확인하며, q가 눌러지면 반복문이 종료됩니다.

반복문이 종료되면, 모든 윈도우 창을 제거합니다.

<br>
<br>

{% highlight Python %}

thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")

_, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

cv2.imshow("Trackbar Windows", binary)

{% endhighlight %}

`트랙 바 값 받기 함수(cv2.setTrackbarPos)`로 트랙 바의 값을 받아옵니다.

`cv2.getTrackbarPos("트랙 바 이름", "윈도우 창 제목")`을 사용해 트랙 바의 값을 받아옵니다.

`트랙 바 이름`, `윈도우 창 제목`은 앞서 설명한 역할과 동일합니다.

이때, 다른 트랙 바 함수와 다르게 **값을 반환합니다.**

이 값을 직접 다른 함수에 넣어, 변경된 화면을 `1ms`마다 확인할 수 있습니다.

<br>

* Tip : 트랙 바의 값을 변경하지 않더라도, 반복문으로 인하여 연산은 계속해서 실행됩니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch35/2.jpg)