---
layout: post
title: "Python tkinter 강좌 : 제 32강 - OpenCV 적용하기"
tagline: "Python tkinter applying OpenCV"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, Python OpenCV, OpenCV With tkinter
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-32/
comments: true
toc: true
---

## OpenCV 적용하기 

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-32/1.webp" class="lazyload" width="100%" height="100%"/>

`OpenCV`와 `tkinter`를 결합해 GUI로 표시할 수 있습니다.

이때 `PIL` 라이브러리를 활용합니다.

`PIL` 모듈은 **Python Imaging Library**로, 다양한 이미지 파일 형식을 지원하는 범용 라이브러리입니다.

<br>
<br>

## OpenCV & PIL

{% highlight Python %}

import cv2
import tkinter
from PIL import Image
from PIL import ImageTk

def convert_to_tkimage():
    global src

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    img = Image.fromarray(binary)
    imgtk = ImageTk.PhotoImage(image=img)

    label.config(image=imgtk)
    label.image = imgtk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")

src = cv2.imread("giraffe.jpg")
src = cv2.resize(src, (640, 400))

img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

img = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=img)

label = tkinter.Label(window, image=imgtk)
label.pack(side="top")

button = tkinter.Button(window, text="이진화 처리", command=convert_to_tkimage)
button.pack(side="bottom", expand=True, fill='both')

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

import cv2
import tkinter
from PIL import Image
from PIL import ImageTk

{% endhighlight %}

<br>

상단에 `from PIL import Image`, `from PIL import ImageTk`를 사용하여 `PIL 모듈`을 포함시킵니다.

`OpenCV`의 `numpy` 형식 이미지를 표시하려면 `PIL` 모듈을 사용합니다.

<br>

{% highlight Python %}

img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

{% endhighlight %}

`tkinter`의 색상 체계는 `OpenCV`와 다르게 `RGB` 패턴을 사용합니다.

그러므로, **BGR**에서 **RGB**로 변환합니다.

<br>

{% highlight Python %}

img = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=img)

{% endhighlight %}

`PIL.Image` 모듈의 `fromarray` 함수를 활용해 `Numpy` 배열을 `Image` 객체로 변환합니다.

`PIL.ImageTk` 모듈의 `PhotoImage` 함수를 활용해 `tkinter`와 호환되는 객체로 변환합니다.

<br>

{% highlight Python %}

label=tkinter.Label(window, image=imgtk)
label.pack(side="top")

{% endhighlight %}

`라벨(Label)`의 `image` 매개변수에 `imgtk`를 적용해 **이미지를 표시할 수 있습니다.**

<br>

{% highlight Python %}

button=tkinter.Button(window, text="이진화 처리", command=convert_to_tkimage)
button.pack(side="bottom", expand=True, fill='both')

{% endhighlight %}

`버튼(Button)`의 `command` 매개변수에 `convert_to_tkimage` 함수를 실행시킵니다.

<br>

{% highlight Python %}

def convert_to_tkimage():
    global src

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    img = Image.fromarray(binary)
    imgtk = ImageTk.PhotoImage(image=img)

    label.config(image=imgtk)
    label.image = imgtk

{% endhighlight %}

`global src`를 적용해 `src` 변수를 가져옵니다.

`이진화`를 처리한 다음, 앞의 형식과 동일하게 `tkinter`와 호환되는 이미지로 변환합니다.

이미지를 갱신하기 위해 `config`를 통해 `image` 매개변수를 설정합니다.

또한, `가비지 컬렉터(garbage collector)`가 이미지를 삭제하지 않도록 라벨의 `image` 매개변수에 `imgtk`를 **한 번 더 등록합니다.**

<br>
<br>

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-32/2.webp" class="lazyload" width="100%" height="100%"/>
