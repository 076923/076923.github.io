---
layout: post
title: "Python tkinter 강좌 : 제 29강 - Sizegrip"
tagline: "Python tkinter Sizegrip"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Sizegrip, tkinter ttk Sizegrip, tkinter winfo
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-29/
comments: true
toc: true
---

## Sizegrip(크기 조절)

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-29/1.webp" class="lazyload" width="100%" height="100%"/>

`Sizegrip`을 이용하여 **위젯의 크기를 조절**할 수 있는 `크기 조절`을 생성할 수 있습니다.

<br>
<br>

## Sizegrip 사용

{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

def Drag(event):

    x=sizegrip.winfo_x()+event.x
    y=sizegrip.winfo_y()+event.y
    sz_width=sizegrip.winfo_reqwidth()
    sz_height=sizegrip.winfo_reqheight()

    text["width"]=x-sz_width
    text["height"]=y-sz_height

    if x >= sz_width and y >= sz_height and x < window.winfo_width() and y < window.winfo_height():
        text.place(x=0, y=0, width=x, height=y)
        sizegrip.place(x=x-sz_width, y=y-sz_height)

text=tkinter.Text(window)
text.place(x=0, y=0)

sizegrip=tkinter.ttk.Sizegrip(window)
sizegrip.place(x=text.winfo_reqwidth()-sizegrip.winfo_reqwidth() , y=text.winfo_reqheight()-sizegrip.winfo_reqheight() )
sizegrip.bind("<B1-Motion>", Drag)

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

def Drag(event):

    x=sizegrip.winfo_x()+event.x
    y=sizegrip.winfo_y()+event.y
    sz_width=sizegrip.winfo_reqwidth()
    sz_height=sizegrip.winfo_reqheight()

    text["width"]=x-sz_width
    text["height"]=y-sz_height

    if x >= sz_width and y >= sz_height and x < window.winfo_width() and y < window.winfo_height():
        text.place(x=0, y=0, width=x, height=y)
        sizegrip.place(x=x-sz_width, y=y-sz_height)

text=tkinter.Text(window)
text.place(x=0, y=0)

sizegrip=tkinter.ttk.Sizegrip(window)
sizegrip.place(x=text.winfo_reqwidth()-sizegrip.winfo_reqwidth() , y=text.winfo_reqheight()-sizegrip.winfo_reqheight() )
sizegrip.bind("<B1-Motion>", Drag)

{% endhighlight %}

`tkinter.ttk.Sizegrip(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 `크기 조절의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `크기 조절의 속성`을 설정합니다.

`마우스 포인터`의 위치와 `위젯의 크기`에 대한 정보를 활용하여 `위젯의 크기`를 조절 할 수 있습니다.

`winfo_*`를 활용하여 `위젯에 대한 정보`를 확인할 수 있습니다.

`x 변수`와 `y 변수`는 마우스로 크기를 조절할 때 `해당 위젯의 좌표`를 의미합니다.

`sz_width 변수`와 `sz_height 변수`는 크기 조절 위젯의 `실제 크기`를 의미합니다.

`if문`을 활용하여 위젯의 크기를 조절할 때 `윈도우 창`을 벗어나지 않게 설정합니다.

`sizegrip.place()`의 좌표를 변경하여 `위젯의 내부`가 아닌 `위젯의 외부`로 위치를 이동시킬 수 있습니다. 

<br>
<br>

## 윈도우 창 크기 조절

{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

sizegrip=tkinter.ttk.Sizegrip(window)
sizegrip.pack(side="bottom", anchor="se")

window.mainloop()

{% endhighlight %}

`Sizegrip`의 **기본 용도**는 `윈도우 창`의 우측 하단에 배치하여 마우스로 `윈도우 창의 크기`를 조절하는 것입니다.

`윈도우 창`이 `resizable(True, True)`로 설정되어 있다면, 별도의 `이벤트 함수` 없이 드래그만으로 윈도우 창의 크기가 조절됩니다.

- Tip : 앞의 예제처럼 `Text` 위젯 등 **윈도우 창이 아닌 위젯**의 크기를 조절하려면 `bind`를 이용하여 직접 구현해야 합니다.

<br>
<br>

## winfo_ Method

|              이름             |             의미             |                                      속성                                     |
|:-----------------------------:|:----------------------------:|:-----------------------------------------------------------------------------:|
| winfo_atom("문자열   식별자") |          식별자 부여         |                         해당 위젯의 정수 식별자를 부여                        |
|  winfo_atomname(정수 식별자)  |          식별자 반환         |                        해당 위젯의 문자열 식별자를 확인                       |
|         winfo_cells()         |          컬러맵 반환         |                        해당 위젯의 컬러맵 셀의 수 반환                        |
|        winfo_children()       |        하위 위젯 반환        |                  해당 위젯에 포함되어 있는 하위 위젯들을 반환                 |
|         winfo_class()         |        클래스 명 반환        |                          해당 위젯의 클래스 명을 반환                         |
|      winfo_colormapfull()     |          컬러맵 확인         |                 해당 위젯에 컬러맵이 포함되어있다면 참 값 반환                |
|     winfo_containing(x, y)    |        경로 이름 반환        |                 해당 위젯에 x, y 위치의 위젯 경로 이름을 반환                 |
|         winfo_depth()         |        비트 깊이 반환        |                          해당 위젯의 비트 깊이를 반환                         |
|         winfo_exists()        |        존재 여부 반환        |                       해당 위젯이 존재한다면 참 값 반환                       |
|      winfo_fpixels(화소)      |         화소 값 반환         |                      거리 문자열을 화소 단위의 부동 소수점 값으로 변환하여 반환                     |
|        winfo_geometry()       |        위젯 설정 반환        |            해당 위젯의 width x height + x + y 형식의 위젯 설정 반환           |
|         winfo_width()         |        위젯 너비 반환        |                             해당 위젯의 너비 반환                             |
|         winfo_height()        |       위젯의 높이 반환       |                             해당 위젯의 높이 반환                             |
|           winfo_id()          |       고유 식별자 반환       |                      해당 위젯의 16진수 고유 식별자 반환                      |
|        winfo_interps()        |  디스플레이 인터프리터 반환  |              해당 위젯에 대한 디스플레이 Tcl 인터프리터 이름 반환             |
|        winfo_ismapped()       |           매핑 반환          |                     해당 위젯이 매핑되어있다면 참 값 반환                     |
|        winfo_manager()        |  지오메트리 매니저 이름 반환 |                   해당 위젯의 지오메트리 매니저의 값을 반환                   |
|          winfo_name()         |        위젯 이름 반환        |                             해당 위젯의 이름 반환                             |
|         winfo_parent()        |        상위 위젯 반환        |                       해당 위젯의 상위 위젯 이름을 반환                       |
|       winfo_pathname(id)      |        경로 이름 반환        |                        해당 id의 위젯의 경로 이름 반환                        |
|       winfo_pixels(화소)       |         화소 값 반환         |                         거리 문자열을 화소 단위의 정수 값으로 변환하여 반환                         |
|        winfo_pointerx()       |    마우스 포인터 x 값 반환   |                   해당 위젯에서 마우스 포인터 x 좌표 값 반환                  |
|        winfo_pointery()       |    마우스 포인터 y 값 반환   |                   해당 위젯에서 마우스 포인터 y 좌표 값 반환                  |
|       winfo_pointerxy()       | 마우스 포인터 (x, y) 값 반환 |                해당 위젯에서 마우스 포인터 (x, y) 좌표 값 반환                |
|        winfo_reqwidth()       |        위젯 너비 반환        |                          해당 위젯의 요청된 너비 반환                         |
|       winfo_reqheight()       |        위젯 높이 반환        |                          해당 위젯의 요청된 높이 반환                         |
|        winfo_rgb(color)       |        (r, g, b) 반환        |                       color에 해당하는 (r, g, b) 값 반환                      |
|         winfo_rootx()         |   위젯 좌측 상단 x 값 반환   |                      해당 위젯의 화면 기준 좌측 상단 x 좌표 값 반환                     |
|         winfo_rooty()         |   위젯 좌측 상단 y 값 반환   |                      해당 위젯의 화면 기준 좌측 상단 y 좌표 값 반환                     |
|         winfo_screen()        |      위젯 화면 이름 반환     |                           해당 위젯의 화면 이름 반환                          |
|      winfo_screencells()      |    위젯 화면 컬러맵 셀 수 반환    |                         해당 위젯의 화면 컬러맵 셀 수 반환                         |
|      winfo_screendepth()      |   위젯 화면 픽셀 깊이 반환   |                           해당 위젯의 화면 깊이 반환                          |
|     winfo_screenwidth()     |      화면 너비 반환     |                           해당 위젯이 표시된 화면(모니터)의 너비를 픽셀 단위로 반환                          |
|     winfo_screenheight()    |      화면 높이 반환     |                           해당 위젯이 표시된 화면(모니터)의 높이를 픽셀 단위로 반환                          |
|     winfo_screenmmwidth()     |      화면 너비 반환 (mm)     |                           해당 위젯이 표시된 화면의 너비를 밀리미터 단위로 반환                          |
|     winfo_screenmmheight()    |      화면 높이 반환 (mm)     |                           해당 위젯이 표시된 화면의 높이를 밀리미터 단위로 반환                          |
|      winfo_screenvisual()     |   위젯 화면 색상 모델 반환   |                        해당 위젯의 화면 색상 모델 반환                        |
|         winfo_server()        |     윈도우 서버 정보 반환    | 해당 위젯의 서버 (1)버전 (2)개정 번호 (3)공급 업체 (4)서버 릴리즈 번호를 반환 |
|        winfo_toplevel()       |       최상위 위젯 반환       |                          해당 위젯의 최상위 위젯 반환                         |
|        winfo_viewable()       |      위젯 매핑 여부 반환     |            해당 위젯이 최상위 윈도우 까지 매핑되어있다면 참 값 반환           |
|         winfo_visual()        |      위젯 색상 모델 반환     |                           해당 위젯의 색상 모델 반환                          |
|        winfo_visualid()       |      비주얼 식별자 반환      |                         해당 위젯의 비주얼 식별자 반환                        |
|    winfo_visualsavailable()   |       비주얼 목록 확인       |               해당 위젯에서 사용할 수 있는 모든 비주얼 목록 확인              |
|       winfo_vrootwidth()      |   위젯 가상 루트 너비 반환   |                        해당 위젯의 가상 루트 너비 반환                        |
|      winfo_vrootheight()      |   위젯 가상 루트 높이 반환   |                        해당 위젯의 가상 루트 높이 반환                        |
|           winfo_x()           |   위젯 좌측 상단 x 값 반환   |                 해당 위젯의 부모에서 좌측 상단 x 좌표 값 반환                 |
|           winfo_y()           |   위젯 좌측 상단 y 값 반환   |                 해당 위젯의 부모에서 좌측 상단 y 좌표 값 반환                 |

* 화소 형식 : "1i"(인치), "2.0c"(센티미터), "10m"(밀리미터), "12p"(포인트)

- Tip : `winfo_width()`, `winfo_height()`는 위젯이 **화면에 배치된 이후**의 실제 크기를 반환하므로, 배치 직후에는 `1`이 반환될 수 있습니다. 배치 전의 크기가 필요하다면 `winfo_reqwidth()`, `winfo_reqheight()`를 사용하거나 `window.update_idletasks()`를 먼저 호출합니다.

* r, g, b 범위 : 0~65535

* 색상 모델 반환 값 : truecolor, staticgray, staticcolor, pseudocolor, grayscale, directcolor

<br>
<br>

## Sizegrip Parameter

### 크기 조절 형식 설정

|   이름   |                           의미                          |     기본값    |               속성                    |
|:--------:|:-------------------------------------------------------:|:-------------:|:-------------:|
|  cursor  |      크기 조절의 마우스 커서 모양                 |       -       |          [커서 속성](#reference-1)             |
|  class_  |      클래스 설정                 |       -       |      -    |  

<br>

### 크기 조절 동작 설정

|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock, coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor
