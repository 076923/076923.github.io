---
layout: post
title: "Python tkinter 강좌 : 제 21강 - Font"
tagline: "Python tkinter Font"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Font
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-21/
comments: true
toc: true
---

## Font(글꼴)

`Font`를 이용하여 위젯들의 문자열에 `글꼴`을 설정할 수 있습니다.

<br>
<br>

## Font 사용

{% highlight Python %}

import tkinter
import tkinter.font

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

font=tkinter.font.Font(family="맑은 고딕", size=20, slant="italic")

label=tkinter.Label(window, text="파이썬 3.6", font=font)
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

import tkinter.font

{% endhighlight %}

<br>

{% highlight Python %}

font=tkinter.font.Font(family="맑은 고딕", size=20, slant="italic")

label=tkinter.Label(window, text="파이썬 3.6", font=font)
label.pack()

{% endhighlight %}

`tkinter.font.Font(매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 `위젯`에 표시할 `글꼴`을 설정할 수 있습니다.

`매개변수`를 사용하여 `글꼴`를 설정합니다.

글꼴을 적용할 `위젯의 font 매개변수`에 사용합니다.

<br>
<br>

## Font Parameter

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| family | 텍스트의 글꼴 | TkDefaultFont | 글꼴 이름 |
| size | 텍스트의 글꼴 크기 | 16 | 상수 |
| weight | 텍스트의 진하게 | normal | normal, bold |
| slant | 텍스트의 기울임 | roman | roamn, italic |
| underline | 텍스트의 밑줄 | False | Boolean |
| overstrike | 텍스트의 취소선 | False | Boolean |
