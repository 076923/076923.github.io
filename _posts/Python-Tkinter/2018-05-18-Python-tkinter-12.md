---
layout: post
title: "Python tkinter 강좌 : 제 12강 - 위젯 배치 : place"
tagline: "Python tkinter place"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter place, tkinter relx rely, tkinter anchor, tkinter 절대 위치 배치
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-12/
comments: true
toc: true
---

## place(절대 위치 배치)

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-12/1.webp" class="lazyload" width="100%" height="100%"/>

`place`를 이용하여 `위젯`들을 배치할 수 있습니다.

<br>
<br>

## place 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

b1=tkinter.Button(window, text="(50, 50)")
b2=tkinter.Button(window, text="(50, 100)")
b3=tkinter.Button(window, text="(100, 150)")
b4=tkinter.Button(window, text="(0, 200)")
b5=tkinter.Button(window, text="(0, 300)")
b6=tkinter.Button(window, text="(0, 300)")

b1.place(x=50, y=50)
b2.place(x=50, y=100, width=50, height=50)
b3.place(x=100, y=150, bordermode="inside")
b4.place(x=0, y=200, relwidth=0.5)
b5.place(x=0, y=300, relx=0.5)
b6.place(x=0, y=300, relx=0.5, anchor="s")

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

b1.place(x=50, y=50)
b2.place(x=50, y=100, width=50, height=50)
b3.place(x=100, y=150, bordermode="inside")
b4.place(x=0, y=200, relwidth=0.5)
b5.place(x=0, y=300, relx=0.5)
b6.place(x=0, y=300, relx=0.5, anchor="s")

{% endhighlight %}

`위젯이름.place(매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `위젯의 배치 속성`을 설정할 수 있습니다.

배치되는 우선 순위는 **가장 처음 선언한** `place`부터 배치됩니다.

`place`는 `절대 위치`로 배치되며, `크기`를 조정할 수 있습니다.

`place()`는 `pack()`, `grid()`와는 **같이 사용할 수 있습니다.**

`x`, `y`와 `relx`, `rely`를 함께 사용하면, `relx`, `rely`로 계산된 위치에 `x`, `y` 값이 **더해진 위치**에 배치됩니다.

- Tip : 좌표의 기준점은 `윈도우 창`의 **왼쪽 상단 모서리(0, 0)**이며, `y` 값이 커질수록 아래쪽으로 이동합니다.

- Tip : `place`는 창의 크기가 바뀌어도 위젯이 자동으로 재배치되지 않으므로, 창 크기에 따라 배치가 달라져야 한다면 `relx`, `rely`, `relwidth`, `relheight`를 사용합니다.

<br>
<br>

## place Parameter

|    이름   |       의미       | 기본값 |              속성             |
|:---------:|:----------------:|:------:|:-----------------------------:|
|     x     |    x좌표 배치    |    0   |              상수             |
|     y     |    y좌표 배치    |    0   |              상수             |
|    relx   |  x좌표 배치 비율 |    0   |             0 ~ 1             |
|    rely   |  y좌표 배치 비율 |    0   |             0 ~ 1             |
|   width   |    위젯의 너비   |    -   |              상수             |
|   height  |    위젯의 높이   |    -   |              상수             |
|  relwidth | 위젯의 너비 비율 |    -   |             0 ~ 1             |
| relheight | 위젯의 높이 비율 |    -   |             0 ~ 1             |
|   anchor  | 위젯의 기준 위치 |   nw   |   center, n, e, w, s, ne, nw, se, sw  |
| bordermode | 위젯의 테두리 포함 여부 | inside | inside, outside, ignore |

<br>

* `x`, `y`, `relx`, `rely` : 해당 구역으로 위젯을 `이동`시킵니다.

* `width`, `height`, `relwidth`, `relheight` : 위젯의 `크기`를 변경시킵니다. `(미설정 시 위젯의 기본 크기 사용)`

* `anchor` : `위젯의 기본 조정 위치`를 변경시킵니다. `(기본값 = 왼쪽 상단 모서리)`

* `bordermode` : 부모 위젯의 `테두리(border)`를 기준으로 좌표와 크기를 계산할지 설정합니다. `inside`는 테두리 안쪽을 기준으로 계산하며, `outside`는 테두리를 포함한 바깥쪽을 기준으로 계산합니다.

- Tip : `relx=0.5, rely=0.5, anchor="center"`로 설정하면 위젯을 윈도우 창의 **정중앙**에 배치할 수 있습니다.
