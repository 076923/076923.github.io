---
layout: post
title: "Python tkinter 강좌 : 제 22강 - PhotoImage"
tagline: "Python tkinter PhotoImage"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter PhotoImage, tkinter 이미지 표시, tkinter png gif, tkinter 이미지 크기 조절, tkinter subsample zoom, tkinter Pillow ImageTk
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-22/
comments: true
toc: true
---

## PhotoImage(이미지)

`PhotoImage`를 이용하여 위젯들의 공간에 `이미지`를 설정할 수 있습니다.

<br>
<br>

## PhotoImage 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

image=tkinter.PhotoImage(file="a.png")

label=tkinter.Label(window, image=image)
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

image=tkinter.PhotoImage(file="a.png")

label=tkinter.Label(window, image=image)
label.pack()

{% endhighlight %}

`tkinter.PhotoImage(file=경로)`를 사용하여 `위젯`에 표시할 `이미지`의 경로를 설정할 수 있습니다.

이미지를 적용할 `위젯의 image 매개변수`에 사용합니다.

상대 경로의 기준은 스크립트 파일의 위치가 아니라, 스크립트를 **실행한 현재 작업 디렉터리**입니다.

- Tip : 스크립트가 저장된 위치와 이미지의 위치가 `동일`하다면 **이미지 파일의 이름만 입력**

- Tip : 실행 위치와 관계없이 스크립트 파일 기준으로 이미지를 찾으려면 `os.path.join(os.path.dirname(__file__), "a.png")`과 같이 경로를 만들어 사용합니다.

<br>
<br>

## 지원 이미지 형식

`PhotoImage`는 `PNG`, `GIF`, `PPM`, `PGM` 형식의 이미지를 지원합니다.

`JPEG`, `BMP`, `WEBP` 등의 형식은 지원하지 않으므로, 이러한 이미지를 사용하려면 `Pillow` 라이브러리의 `PIL.ImageTk.PhotoImage`를 사용합니다.

{% highlight Python %}

from PIL import Image, ImageTk

image=ImageTk.PhotoImage(Image.open("a.jpg"))

label=tkinter.Label(window, image=image)
label.pack()

{% endhighlight %}

- Tip : `Pillow`는 `pip install pillow`로 설치할 수 있습니다.

<br>
<br>

## 이미지 참조 유지

{% highlight Python %}

def show_image():
    image=tkinter.PhotoImage(file="a.png")
    label=tkinter.Label(window, image=image)
    label.image=image
    label.pack()

{% endhighlight %}

`PhotoImage` 객체는 **참조가 사라지면 가비지 컬렉션에 의해 삭제**되어, 위젯에 이미지가 표시되지 않고 빈 공간만 나타납니다.

예제와 같이 함수 안에서 이미지를 생성하는 경우, 함수가 종료되면 지역 변수 `image`가 사라지므로 `label.image=image`와 같이 **위젯의 속성에 이미지를 저장**하여 참조를 유지합니다.

- Tip : 이미지가 표시되지 않고 빈 위젯만 나타난다면, 오류가 발생하지 않더라도 **참조 유지 문제**를 가장 먼저 확인합니다.

<br>
<br>

## PhotoImage Method

|      이름      |               의미               |      설명      |
|:--------------:|:--------------------------------:|:----------------:|
| width() | 이미지 너비 반환 | 이미지의 너비(픽셀)를 반환 |
| height() | 이미지 높이 반환 | 이미지의 높이(픽셀)를 반환 |
| subsample(x, y) | 이미지 축소 | 가로 `x`배, 세로 `y`배 **축소**한 새로운 이미지 반환 `예) subsample(2, 2) → 1/2 크기` |
| zoom(x, y) | 이미지 확대 | 가로 `x`배, 세로 `y`배 **확대**한 새로운 이미지 반환 `예) zoom(2, 2) → 2배 크기` |

<br>

- Tip : `subsample`과 `zoom`은 **정수 배율**만 사용할 수 있으므로, 임의의 크기로 조절하려면 `Pillow`의 `Image.resize`를 사용합니다.

- Tip : `Label` 외에도 `Button`, `Canvas(create_image)` 등의 위젯에 이미지를 적용할 수 있으며, `compound` 매개변수를 사용하면 문자열과 이미지를 함께 표시할 수 있습니다.
