---
layout: post
title: "Python tkinter 강좌 : 제 11강 - 위젯 배치 : grid"
tagline: "Python tkinter grid"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter grid, tkinter 위젯 배치, tkinter sticky, tkinter columnconfigure, tkinter rowconfigure
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-11/
comments: true
toc: true
---

## grid(셀 단위 배치)

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-11/1.webp" class="lazyload" width="100%" height="100%"/>

`grid`를 이용하여 `위젯`들을 `행(row)`과 `열(column)`로 이루어진 `격자(셀)` 단위로 배치할 수 있습니다.

- `pack 사용하기` : [10강 바로가기][10강]

- `place 사용하기` : [12강 바로가기][12강]

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

`grid`는 `셀 단위`로 배치되며, 위젯이 하나도 없는 `행`이나 `열`은 **크기가 0**이므로 화면에 나타나지 않습니다.

`grid()`는 `pack()`과 **같이 사용될 수 없으며**, `place()`와는 **같이 사용할 수 있습니다.**

- Tip : `column`을 `99`로 위치시키더라도, `4`부터 `98`까지의 열에는 위젯이 없어 너비가 `0`이 되므로 화면에서는 `3`열 바로 다음에 배치된 것처럼 표시됩니다.

- Tip : `rowspan=2`로 설정된 `b4`는 `1행`과 `2행`을 세로로 차지하며, `columnspan=3`으로 설정된 `b5`는 `1열`부터 `3열`까지 가로로 차지합니다.

- Tip : 같은 `윈도우 창(또는 프레임)` 안에서 `pack()`과 `grid()`를 함께 사용하면 `TclError`가 발생합니다.

<br>
<br>

## grid Parameter

|    이름    |              의미              | 기본값 |            속성            |
|:----------:|:------------------------------:|:------:|:--------------------------:|
|     row    |             행 위치            |    0   |            상수            |
|   column   |             열 위치            |    0   |            상수            |
|   rowspan  |        차지할 행의 개수        |    1   |            상수            |
| columnspan |        차지할 열의 개수        |    1   |            상수            |
|   sticky   | 할당된 공간 내에서의 위치 조정 |    -   | n, e, s, w, nw, ne, sw, se, nsew 등의 조합 |
|    ipadx   |  위젯에 대한 x 방향 내부 패딩  |    0   |            상수            |
|    ipady   |  위젯에 대한 y 방향 내부 패딩  |    0   |            상수            |
|    padx    |  위젯에 대한 x 방향 외부 패딩  |    0   |            상수            |
|    pady    |  위젯에 대한 y 방향 외부 패딩  |    0   |            상수            |

<br>

* `row`, `column` : 해당 구역으로 위젯을 `이동`시킵니다.

* `rowspan`, `columnspan` : 위젯이 `여러 행` 또는 `여러 열`을 차지하도록 셀을 `병합`합니다.

* `sticky` : `현재 배치된 구역` 안에서 `특정 위치`로 이동시킵니다. `"we"`, `"nsew"`처럼 마주보는 방향을 함께 설정하면 **해당 방향으로 위젯이 늘어나** 셀을 채웁니다.

- Tip : `sticky`를 설정하지 않으면 위젯은 셀의 `중앙`에 배치되며, `pack`의 `fill`과 같은 효과를 내려면 `sticky="nsew"`로 설정합니다.

- Tip : 윈도우 창의 크기가 변경될 때 셀의 크기도 함께 변경되게 하려면 `window.grid_columnconfigure(열, weight=1)`, `window.grid_rowconfigure(행, weight=1)`로 해당 `열`과 `행`에 `가중치(weight)`를 설정해야 합니다. 가중치가 `0`(기본값)인 행과 열은 창의 크기가 변경되어도 크기가 유지됩니다.

[10강]: https://076923.github.io/posts/Python-tkinter-10/
[12강]: https://076923.github.io/posts/Python-tkinter-12/
