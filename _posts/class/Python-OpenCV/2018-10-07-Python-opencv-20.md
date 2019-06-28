---
layout: post
title: "Python OpenCV 강좌 : 제 20강 - 캡쳐 및 녹화"
tagline: "Python OpenCV Capture & Record"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Capture, OpenCV Record
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-20/
comments: true
---

## 캡쳐 및 녹화(Capture & Record) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch20/1.png)
영상이나 이미지를 `캡쳐하거나 녹화`하기 위해 사용합니다. 영상이나 이미지를 `연속적 또는 순간적으로 캡쳐하거나 녹화`할 수 있습니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import datetime
import cv2

capture = cv2.VideoCapture("/Image/Star.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("/Image/Star.mp4")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == 26:
        print("캡쳐")
        cv2.imwrite("D:/" + str(now) + ".png", frame)
    elif key == 24
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("D:/" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    elif key == 3
        print("녹화 중지")
        record = False
        video.release()
        
    if record == True:
        print("녹화 중..")
        video.write(frame)

capture.release()
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

{% endhighlight %}

`fourcc`를 생성하여 **디지털 미디어 포맷 코드**를 생성합니다. `cv2.VideoWriter_fourcc(*'코덱')`을 사용하여 인코딩 방식을 설정합니다.

`record` 변수를 생성하여 `녹화 유/무`를 설정합니다.

* Tip : `FourCC(Four Character Code)` : 디지털 미디어 포맷 코드입니다. 즉, **코덱의 인코딩 방식**을 의미합니다.

<br>

{% highlight Python %}

import datetime

now = datetime.datetime.now().strftime("%d_%H-%M-%S")

{% endhighlight %}

`datetime` 모듈을 포함하여 **현재 시간**을 받아와 제목으로 사용합니다.

`now`에 파일의 제목을 설정합니다. `날짜_시간-분-초`의 형식으로 제목이 생성됩니다.

<br>

{% highlight Python %}

key = cv2.waitKey(33)

{% endhighlight %}

`key` 값에 현재 눌러진 `키보드 키`의 값이 저장됩니다. `33ms`마다 갱신됩니다.

<br>

{% highlight Python %}

if key == 27:
    break
elif key == 26:
    print("캡쳐")
    cv2.imwrite("D:/" + str(now) + ".png", frame)
elif key == 24
    print("녹화 시작")
    record = True
    video = cv2.VideoWriter("D:/" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
elif key == 3
    print("녹화 중지")
    record = False

{% endhighlight %}

`if-elif``문을 이용하여 눌러진 키의 값을 판단합니다.

`27 = ESC`, `26 = Ctrl + Z`, `24 = Ctrl + X`, `3 = Ctrl + C`를 의미합니다.

`ESC`키를 눌렀을 경우, 프로그램을 종료합니다.

`Ctrl + Z`를 눌렀을 경우, 현재 화면을 **캡쳐**합니다. `cv2.imwrite("경로 및 제목", 이미지)`를 이용하여 해당 `이미지`를 저장합니다.

`Ctrl + X`를 눌렀을 경우, **녹화를 시작합니다.** `video`에 녹화할 파일 형식을 설정합니다.

`cv2.VideoWriter("경로 및 제목", 비디오 포맷 코드, FPS, (녹화 파일 너비, 녹화 파일 높이))`를 의미합니다.

`Ctrl + C`를 눌렀을 경우, **녹화를 중지합니다.** `video.release()`를 사용하여 `메모리`를 해제합니다.

녹화 시작할 때, `record`를 `True`로, 녹화를 중지할 때 `record`를 `False`로 변경합니다.

* Tip : `key` 값은 `아스키 값`을 사용합니다.
* Tip : `FPS(Frame Per Second)` : 영상이 바뀌는 속도를 의미합니다. 즉, 화면의 부드러움을 의미합니다.
* Tip : `frame.shape`는 (높이, 너비, 채널)의 값이 저장되어있습니다.

<br>

{% highlight Python %}

if record == True:
    print("녹화 중..")
    video.write(frame)

{% endhighlight %}

`if`문을 이용하여 `record`가 `True`일때 `video`에 `프레임을 저장합니다.

`video.write(저장할 프레임)`을 사용하여 프레임을 저장할 수 있습니다.

<br>
<br>

## Additional Information ##
----------

## FourCC 종류 ##

`CVID`, `Default`, `DIB`, `DIVX`, `H261`, `H263`, `H264`, `IV32`, `IV41`, `IV50`, `IYUB`, `MJPG`, `MP42`, `MP43`, `MPG4`, `MSVC`, `PIM1`, `Prompt`, `XVID`

* Tip : `단일 채널` 이미지의 경우, 사용할 수 없는 `디지털 미디어 포맷 코드`가 존재합니다.





