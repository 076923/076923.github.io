---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 4강 - VIDEO 출력"
crawlertitle: "Python OpenCV 강좌 : 제 4강 - VIDEO 출력"
summary: "Python OpenCV Using Video"
date: 2018-08-15
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### Video 출력 ###
----------
[![1]({{ site.images }}/Python/opencv/ch4/1.jpg)]({{ site.images }}/Python/opencv/ch4/1.jpg)
**동영상 파일**에서 이미지를 얻어와 프레임을 재생할 수 있습니다.

<br>
<br>

### Main Code ###
----------

{% highlight Python %}

import cv2

capture = cv2.VideoCapture("Image/Star.mp4")

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("Image/Star.mp4")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(33) > 0: break

capture.release()
cv2.destroyAllWindows()
{% endhighlight %}

<br>
<br>

### Detailed Code ###
----------

{% highlight Python %}

capture = cv2.VideoCapture("Image/Star.mp4")

{% endhighlight %}

`cv2.VideoCapture("경로")`을 이용하여 **동영상 파일**에서 프레임을 받아옵니다.


<br>
<br>

{% highlight Python %}

if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
    capture.open("Image/Star.mp4")

{% endhighlight %}

`if`문을 이용하여 가장 처음 `현재 프레임 개수`와 `총 프레임 개수`를 비교합니다.

`capture.get(속성)`을 이용하여 `capture`의 `속성`을 반환합니다.

`cv2.CAP_PROP_POS_FRAMES`는 **현재 프레임 개수를 의미합니다.**

`cv2.CAP_PROP_FRAME_COUNT`는 **총 프레임 개수를 의미합니다.**

같을 경우, 마지막 프레임이므로, `capture.open(경로)`를 이용하여 **다시 동영상 파일을 불러옵니다.**

<br>
<br>

{% highlight Python %}

if cv2.waitKey(33) > 0: break

{% endhighlight %}

`cv2.waitkey(time)`을 이용하여 `33ms`마다 프레임을 재생합니다.

어떤 키라도 누를 경우, `break`하여 `while`문을 종료합니다.

<br>
<br>

### Additional Information ###
----------

#### VideoCapture 함수 ###

`capture.get(속성)` : `VideoCapture`의 **속성**을 반환합니다.

`capture.grab()` : `Frame`의 **호출 성공 유/무**를 반환합니다.

`capture.isOpened()` : `VideoCapture`의 **성공 유/무**를 반환합니다.

`capture.open(카메라 장치 번호 또는 경로)` : `카메라`나 `동영상 파일`을 엽니다.

`capture.release()` : `VideoCapture`의 **장치를 닫고 메모리를 해제합니다.**

`capture.retrieve()` : `VideoCapture`의 **프레임**과 **플래그**를 반환합니다.

`capture.set(속성, 값)` :  `VideoCapture`의 **속성**의 **값**을 설정합니다.

<br>
<br>

#### VideoCapture 속성 ###

|            속성           |          의미         |          비고         |
|:-------------------------:|:---------------------:|:---------------------:|
|  cv2.CAP_PROP_FRAME_WIDTH |     프레임의 너비     |           -           |
| cv2.CAP_PROP_FRAME_HEIGHT |     프레임의 높이     |           -           |
|  cv2.CAP_PROP_FRAME_COUNT |    프레임의 총 개수   |           -           |
|      cv2.CAP_PROP_FPS     |      프레임 속도      |           -           |
|    cv2.CAP_PROP_FOURCC    |       코덱 코드       |           -           |
|  cv2.CAP_PROP_BRIGHTNESS  |      이미지 밝기      |     카메라만 해당     |
|   cv2.CAP_PROP_CONTRAST   |      이미지 대비      |     카메라만 해당     |
|  cv2.CAP_PROP_SATURATION  |      이미지 채도      |     카메라만 해당     |
|      cv2.CAP_PROP_HUE     |      이미지 색상      |     카메라만 해당     |
|     cv2.CAP_PROP_GAIN     |      이미지 게인      |     카메라만 해당     |
|   cv2.CAP_PROP_EXPOSURE   |      이미지 노출      |     카메라만 해당     |
|   cv2.CAP_PROP_POS_MSEC   |    프레임 재생 시간   |        ms 반환        |
|  cv2.CAP_PROP_POS_FRAMES  |      현재 프레임      | 프레임의 총 개수 미만 |
|   CAP_PROP_POS_AVI_RATIO  | 비디오 파일 상대 위치 |    0 = 시작, 1 = 끝   |


<br>
<br>

### Result ###
----------

[![2]({{ site.images }}/Python/opencv/ch4/2.png)]({{ site.images }}/Python/opencv/ch4/2.png)
