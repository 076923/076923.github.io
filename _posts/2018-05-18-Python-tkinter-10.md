---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 10강 – 위젯 배치 : pack"
crawlertitle: "Python tkinter 강좌 : 제 10강 - 위젯 배치 : pack"
summary: "Python tkinter pack"
date: 2018-05-18
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### pack (상대 위치 배치) ###
----------

[![1]({{ site.images }}/Python/tkinter/ch10/1.png)]({{ site.images }}/Python/tkinter/ch10/1.png)

`pack`을 이용하여 `위젯`들을 배치할 수 있습니다.

<br>
<br>
### pack 사용 ###
----------
{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

b1=tkinter.Button(window, text="top")
b1_1=tkinter.Button(window, text="top-1")

b2=tkinter.Button(window, text="bottom")
b2_1=tkinter.Button(window, text="bottom-1")

b3=tkinter.Button(window, text="left")
b3_1=tkinter.Button(window, text="left-1")

b4=tkinter.Button(window, text="right")
b4_1=tkinter.Button(window, text="right-1")

b5=tkinter.Button(window, text="center", bg="red")

b1.pack(side="top")
b1_1.pack(side="top", fill="x")

b2.pack(side="bottom")
b2_1.pack(side="bottom", anchor="e")

b3.pack(side="left")
b3_1.pack(side="left", fill="y")

b4.pack(side="right")
b4_1.pack(side="right", anchor="s")

b5.pack(expand=True, fill="both")

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

b1.pack(side="top")
b1_1.pack(side="top", fill="x")

b2.pack(side="bottom")
b2_1.pack(side="bottom", anchor="e")

b3.pack(side="left")
b3_1.pack(side="left", fill="y")

b4.pack(side="right")
b4_1.pack(side="right", anchor="s")

b5.pack(expand=True, fill="both")

{% endhighlight %}


`위젯이름.pack(파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `위젯의 배치 속성`을 설정할 수 있습니다.

배치되는 우선 순위는 **가장 처음 선언한** `pack`부터 배치됩니다.

`pack`의 `파라미터`로 인하여 **위젯의 크기가 변경될 수 있습니다.**

`pack()`은 `grid()`와 **같이 사용될 수 없으며**, `place()`와는 **같이 사용할 수 있습니다.**

<br>
<br>
### pack Parameter ###
----------

|  이름  |             의미             | 기본값 |                속성                |
|:------:|:----------------------------:|:------:|:----------------------------------:|
|  side  |     특정 위치로 공간 할당    |   top  |      top, bottom, left, right      |
| anchor | 할당된 공간 내에서 위치 지정 | center | center, n, e, s, w, ne, nw, se, sw |
|  fill  |  할당된 공간에 대한 크기 맞춤  |  none  |          none, x, y, both          |
| expand |       미사용 공간 확보       |  False |               Boolean               |
|  ipadx | 위젯에 대한 x 방향 내부 패딩 |    0   |                상수                |
|  ipady | 위젯에 대한 y 방향 내부 패딩 |    0   |                상수                |
|  padx  | 위젯에 대한 x 방향 외부 패딩 |    0   |                상수                |
|  pady  | 위젯에 대한 y 방향 외부 패딩 |    0   |                상수                |

<br>

* `side` : 해당 구역으로 위젯을 `이동`시킨다.
* `anchor` : `현재 배치된 구역` 안에서 `특정 위치`로 이동시킨다.
* `fill` : 할당된 공간에 맞게 `크기가 변경`된다.
* `expand` : 할당되지 않은 `미사용 공간`을 모두 **현재 위젯의 할당된 공간으로 변경한다.**


