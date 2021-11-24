---
layout: post
title: "Python tkinter 강좌 : 제 11강 - 위젯 배치 : grid"
tagline: "Python tkinter pack"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter grid
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-11/
comments: true
toc: true
---

## grid(셀 단위 배치)

![1]({{ site.images }}/assets/posts/Python/Tkinter/lecture-11/1.png)

`grid`을 이용하여 `위젯`들을 배치할 수 있습니다.

<br>
<br>

## grid 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

b1=tkinter.Button(window, text="(0, 0)")
b2=tkinter.Button(window, text="(0, 1)", width=20)
b3=tkinter.Button(window, text="(0, 2)")

b4=tkinter.Button(window, text="(1, 0)")
b5=tkinter.Button(window, text="(1, 1)")
b6=tkinter.Button(window, text="(1, 3)")

b7=tkinter.Button(window, text="(2, 1)")
b8=tkinter.Button(window, text="(2, 2)")
b9=tkinter.Button(window, text="(2, 4)")

b1.grid(row=0, column=0)
b2.grid(row=0, column=1)
b3.grid(row=0, column=2)

b4.grid(row=1, column=0, rowspan=2)
b5.grid(row=1, column=1, columnspan=3)
b6.grid(row=1, column=3)

b7.grid(row=2, column=1, sticky="w")
b8.grid(row=2, column=2)
b9.grid(row=2, column=99)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

b1.grid(row=0, column=0)
b2.grid(row=0, column=1)
b3.grid(row=0, column=2)

b4.grid(row=1, column=0, rowspan=2)
b5.grid(row=1, column=1, columnspan=3)
b6.grid(row=1, column=3)

b7.grid(row=2, column=1, sticky="w")
b8.grid(row=2, column=2)
b9.grid(row=2, column=99)

{% endhighlight %}

`위젯이름.grid(매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `위젯의 배치 속성`을 설정할 수 있습니다.

배치되는 우선 순위는 **가장 처음 선언한** `grid`부터 배치됩니다.

`grid`의 `셀 단위`로 배치되며, `한 번에 여러 셀`을 **건너 뛰어 배치할 수 없습니다.**

`grid()`은 `pack()`과 **같이 사용될 수 없으며**, `place()`와는 **같이 사용할 수 있습니다.**

- Tip : `column`을 `99`로 위치시키더라도, 그 전 `grid` 배치에서 `최대 column의 값`이 `3`이 였으므로 자동적으로 `4`로 할당됩니다.

<br>
<br>

## grid Parameter

|    이름    |              의미              | 기본값 |            속성            |
|:----------:|:------------------------------:|:------:|:--------------------------:|
|     row    |             행 위치            |    0   |            상수            |
|   column   |             열 위치            |    0   |            상수            |
|   rowspan  |          행 위치 조정          |    1   |            상수            |
| columnspan |          열 위치 조정          |    1   |            상수            |
|   sticky   | 할당된 공간 내에서의 위치 조정 |    -   | n, e, s, w, nw, ne, sw, se |
|    ipadx   |  위젯에 대한 x 방향 내부 패딩  |    0   |            상수            |
|    ipady   |  위젯에 대한 y 방향 내부 패딩  |    0   |            상수            |
|    padx    |  위젯에 대한 x 방향 외부 패딩  |    0   |            상수            |
|    pady    |  위젯에 대한 y 방향 외부 패딩  |    0   |            상수            |

<br>

* `row`, `column` : 해당 구역으로 위젯을 `이동`시킵니다.

* `rowspan`, `columnspan` : 현재 배치된 구역에서 위치를 `조정`합니다.

* `sticky` : `현재 배치된 구역` 안에서 `특정 위치`로 이동시킵니다.
