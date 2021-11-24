---
layout: post
title: "Python tkinter 강좌 : 제 22강 - PhotoImage"
tagline: "Python tkinter PhotoImage"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter PhotoImage
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

`tkinter.PhotoImage(경로)`을 사용하여 `위젯`에 표시할 `이미지`의 경로를 설정할 수 있습니다.

글꼴을 적용할 `위젯의 image 매개변수`에 사용합니다.

기본 경로는 현재 사용하고 있는 `프로젝트의 위치`가 기본 경로입니다.

- Tip : 프로젝트가 저장된 위치와 이미지의 위치가 `동일`하다면 **이미지 파일의 이름만 입력**
