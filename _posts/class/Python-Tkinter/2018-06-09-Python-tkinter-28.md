---
layout: post
title: "Python tkinter 강좌 : 제 28강 - Notebook"
tagline: "Python tkinter Notebook"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Notebook
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-28/
comments: true
---

## Notebook(탭 메뉴) ##
----------

![1]({{ site.images }}/assets/images/Python/tkinter/ch28/1.png)
`Notebook`을 이용하여 **페이지**를 나눌 수 있는 `탭 메뉴`를 생성할 수 있습니다.

<br>
<br>

## Notebook 사용 ##
----------
{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

notebook=tkinter.ttk.Notebook(window, width=300, height=300)
notebook.pack()

frame1=tkinter.Frame(window)
notebook.add(frame1, text="페이지1")

label1=tkinter.Label(frame1, text="페이지1의 내용")
label1.pack()

frame2=tkinter.Frame(window)
notebook.add(frame2, text="페이지2")

label2=tkinter.Label(frame2, text="페이지2의 내용")
label2.pack()

frame3=tkinter.Frame(window)
notebook.add(frame3, text="페이지4")

label3=tkinter.Label(frame3, text="페이지4의 내용")
label3.pack()

frame4=tkinter.Frame(window)
notebook.insert(2, frame4, text="페이지3")

label4=tkinter.Label(frame4, text="페이지3의 내용")
label4.pack()

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

notebook=tkinter.ttk.Notebook(window, width=300, height=300)
notebook.pack()

frame1=tkinter.Frame(window)
notebook.add(frame1, text="페이지1")

label1=tkinter.Label(frame1, text="페이지1의 내용")
label1.pack()

frame2=tkinter.Frame(window)
notebook.add(frame2, text="페이지2")

label2=tkinter.Label(frame2, text="페이지2의 내용")
label2.pack()

frame3=tkinter.Frame(window)
notebook.add(frame3, text="페이지4")

label3=tkinter.Label(frame3, text="페이지4의 내용")
label3.pack()

frame4=tkinter.Frame(window)
notebook.insert(2, frame4, text="페이지3")

label4=tkinter.Label(frame4, text="페이지3의 내용")
label4.pack()

{% endhighlight %}


`tkinter.ttk.Notebook(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 `탭 메뉴의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `탭 메뉴의 속성`을 설정합니다.

`프레임`을 이용하여 `탭 페이지`를 설정할 수 있습니다.

<br>
<br>

## Notebook Method ##
----------

## 탭 메뉴 메소드 ##

|              이름              |       의미       |                  설명                  |
|:------------------------------:|:----------------:|:--------------------------------------:|
|       add(frame, option)       |    페이지 추가   |         탭 메뉴의 페이지를 추가        |
| insert(tabname, frame, option) |    페이지 삽입   | 탭 메뉴의 tabname 위치에 페이지를 추가 |
|         forget(tabname)        |    페이지 삭제   |     탭 메뉴의 tabname 페이지를 삭제    |
|          hide(tabname)         |    페이지 숨김   |     탭 메뉴의 tabname 페이지를 숨김    |
|         select(tabname)        |    페이지 선택   |     탭 메뉴의 tabname 페이지를 선택    |
|         index(tabname)         | 페이지 위치 반환 | 탭 메뉴의 tabname 페이지의 위치를 반환 |
|          tab(tabname)          | 페이지 설정 반환 | 탭 메뉴의 tabname 페이지의 설정을 반환 |
|             tabs()             |    페이지 반환   |      탭 메뉴의 페이지 tabname 반환     |
|       enable_traversal()       |  키 바인딩 허용  |       탭 메뉴의 키 바인딩을 허용       |

* `tabname` : `index`로 사용하거나, `frame` 위젯의 **변수 이름**으로 사용
* `current` : 현재 선택된 탭 페이지의 메뉴, (문자열)
* `end` : 탭 메뉴의 탭 페이지 개수, (문자열)

<br>

* enable_traversal()

    - `Ctrl + Tab` 을 이용하여 다음 페이지로 이동 가능
    - `Shift + Ctrl + Tab`을 이용하여 이전 페이지로 이동 가능

<br>
<br>

## 탭 메뉴 Option ##

|    이름   |                             의미                             |                  속성                  |
|:---------:|:------------------------------------------------------------:|:--------------------------------------:|
|    text   |                   탭 페이지에 표시할 문자열                  |                 문자열                 |
| underline |   탭 페이지의 문자열 중 index에 해당되는 문자에 밑줄 표시    |                  상수                  |
|   sticky  |          탭 페이지의 할당된 공간 내에서의 위치 조정          |       n, e, w, s, ne, nw, se, sw       |
|   state   |                     탭 페이지의 상태 설정                    |        normal, disabled, hidden        |
|   image   |                탭 페이지에 포함할 임의 이미지                |                    -                   |
|  compound | 탭 페에지에 문자열과 이미지를 동시에 표시할 때 이미지의 위치 | none, center, left, right, top, bottom |

<br>
<br>

## Notebook Parameter ##
----------


## 탭 메뉴 형태 설정 ##

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | 탭 메뉴의 너비 | 0 | 상수 |
| height | 탭 메뉴의 높이 | 0 | 상수 |
| padding | 탭 메뉴의 여백 | 0 | 상수 |

<br>
<br>

## 탭 메뉴 형식 설정 ##

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |      탭 메뉴의 마우스 커서 모양                 |       -       |          [커서 속성](#reference-1)             |
|  class_  |      클래스 설정                 |       -       |      -    |  

<br>
<br>

## 탭 메뉴 동작 설정 ##


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |

<br>
<br>

## 참고 ##
----------

<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

