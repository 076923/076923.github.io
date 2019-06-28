---
layout: post
title: "Python tkinter 강좌 : 제 31강 – Separator"
tagline: "Python tkinter Separator"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Separator
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-31/
comments: true
---

## Separator(구분선) ##
----------

![1]({{ site.images }}/assets/images/Python/tkinter/ch31/1.png)
`Separator`를 이용하여 **위젯의 구역**을 나눌 수 있는 `구분선`을 생성할 수 있습니다.

<br>
<br>

## Separator 사용 ##
----------

{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")
window.resizable(False, False)

button1=tkinter.Button(window, width=10, height=5, text="1번")
button1.grid(row=0, column=0)

button2=tkinter.Button(window, width=10, height=5, text="2번")
button2.grid(row=0, column=2)

button3=tkinter.Button(window, width=10, height=5, text="3번")	
button3.grid(row=1, column=1)
		
button4=tkinter.Button(window, width=10, height=5, text="4번")
button4.grid(row=2, column=0)
		
button5=tkinter.Button(window, width=10, height=5, text="5번")
button5.grid(row=2, column=2)

s=tkinter.ttk.Separator(window, orient="vertical")	
s.grid(row=0,column=1, sticky='ns')

s2=tkinter.ttk.Separator(window, orient="horizontal")	
s2.grid(row=1,column=2, sticky='ew')

s3=tkinter.ttk.Separator(window, orient="vertical")
s3.grid(row=1,column=0, sticky='ns')

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

import tkinter.ttk

{% endhighlight %}

<br>

상단에 `import tkinter.ttk`를 사용하여 `ttk 모듈`을 포함시킵니다.

tkinter.ttk 함수의 사용방법은 `tkinter.ttk.*`를 이용하여 사용이 가능합니다.

<br>

{% highlight Python %}

button1=tkinter.Button(window, width=10, height=5, text="1번")
button1.grid(row=0, column=0)

button2=tkinter.Button(window, width=10, height=5, text="2번")
button2.grid(row=0, column=2)

button3=tkinter.Button(window, width=10, height=5, text="3번")	
button3.grid(row=1, column=1)
		
button4=tkinter.Button(window, width=10, height=5, text="4번")
button4.grid(row=2, column=0)
		
button5=tkinter.Button(window, width=10, height=5, text="5번")
button5.grid(row=2, column=2)

s=tkinter.ttk.Separator(window, orient="vertical")	
s.grid(row=0,column=1, sticky='ns')

s2=tkinter.ttk.Separator(window, orient="horizontal")	
s2.grid(row=1,column=2, sticky='ew')

s3=tkinter.ttk.Separator(window, orient="vertical")
s3.grid(row=1,column=0, sticky='ns')

{% endhighlight %}


`tkinter.ttk.Separator(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 `구분선의 속성`을 설정할 수 있습니다.

`구분선.grid()`의 파라미터에서 `sticky`의 값을 설정하여 할당된 공간 내에서의 `위치를 조정`할 수 있습니다.

`orient`의 속성이 `horizontal`일 경우 `sticky`의 속성은 고정적으로 `ew`입니다.

`orient`의 속성이 `vertical`일 경우 `sticky`의 속성은 고정적으로 `ns`입니다.

<br>
<br>

## Separator Parameter ##
----------

## 구분선 형태 설정 ##

|   이름   |                           의미                          |     기본값    |               속성                    |
|:--------:|:-------------------------------------------------------:|:-------------:|:-------------:|
|  orient |      구분선의 표시 방향   |       -     |    horizontal, vertical |


<br>
<br>

## 구분선 형식 설정 ##

|   이름   |                           의미                          |     기본값    |               속성                    |
|:--------:|:-------------------------------------------------------:|:-------------:|:-------------:|
|  cursor  |      구분선의 마우스 커서 모양                 |       -       |          [커서 속성](#reference-1)             |
|  class_  |      클래스 설정                 |       -       |      -    |  


<br>
<br>

## 구분선 동작 설정 ##


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |

<br>
<br>

## 참고 ##
----------

<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock, coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


