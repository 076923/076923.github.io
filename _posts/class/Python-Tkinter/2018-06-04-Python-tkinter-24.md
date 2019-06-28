---
layout: post
title: "Python tkinter 강좌 : 제 24강 – Toplevel"
tagline: "Python tkinter Toplevel"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Toplevel
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-24/
comments: true
---

## Toplevel(외부 윈도우) ##
----------

`Toplevel`을 이용하여 **다른 위젯들을 포함**하는 `외부 윈도우`를 생성할 수 있습니다.

<br>
<br>

## Toplevel 사용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

menubar=tkinter.Menu(window)

menu_1=tkinter.Menu(menubar, tearoff=0)
menu_1.add_command(label="하위 메뉴 1-1")
menu_1.add_command(label="하위 메뉴 1-2")
menu_1.add_separator()
menu_1.add_command(label="하위 메뉴 1-3")
menubar.add_cascade(label="상위 메뉴 1", menu=menu_1)

toplevel=tkinter.Toplevel(window, menu=menubar)
toplevel.geometry("320x200+820+100")

label=tkinter.Label(toplevel, text="YUN DAE HEE")
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

menubar=tkinter.Menu(window)

menu_1=tkinter.Menu(menubar, tearoff=0)
menu_1.add_command(label="하위 메뉴 1-1")
menu_1.add_command(label="하위 메뉴 1-2")
menu_1.add_separator()
menu_1.add_command(label="하위 메뉴 1-3")
menubar.add_cascade(label="상위 메뉴 1", menu=menu_1)

toplevel=tkinter.Toplevel(window, menu=menubar)
toplevel.geometry("320x200+820+100")

label=tkinter.Label(toplevel, text="YUN DAE HEE")
label.pack()

{% endhighlight %}


`tkinter.Toplevel(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `외부 윈도우의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `외부 윈도우의 속성`을 설정합니다.

`메인 윈도우 창`에도 동일하게 적용할 수 있습니다.

<br>
<br>

## Toplevel Method ##
----------


|            이름           |                          의미                          |
|:-------------------------:|:------------------------------------------------------:|
|       title("제목")       |               외부 윈도우 창의 제목 설정               |
|   resizable(너비, 높이)   |            외부 윈도우 창의 너비, 높이 설정            |
|    maxsize(너비, 높이)    |       외부 윈도우 창의 최대 너비, 최대 높이 설정       |
|    minsize(너비, 높이)    |       외부 윈도우 창의 최소 너비, 최소 높이 설정       |
|   aspect(n1, d1, n2, d2)  |  외부 윈도우 창의 너비, 높이 비율 제한 (n1/d1, n2/d2)  |
|          state()          | 외부 윈도우 창의 상태 반환 (normal, iconic, withdrawn) |
|         iconify()         |            외부 윈도우 창을 아이콘으로 변경            |
|         withdraw()        |             외부 윈도우 창을 화면에서 제거             |
|        deiconify()        |                  외부 윈도우 창을 복원                 |
| overrideredirect(Boolean) |           외부 윈도우 창의 상태 표시줄 유/무           |
|          frame()          |            외부 윈도우 창의 고유 식별자 반환           |
|           lift()          |             윈도우 겹침 순서를 맨 위로 이동            |
|          lower()          |            윈도우 겹침 순서를 맨 아래로 이동           |
|     transient(윈도우)     |        지정된 윈도우 창에 대해 임시 창으로 전환        |
|     sizefrom(컨트롤러)    |                   크기 컨트롤러 설정                   |
|   positionfrom(컨트롤러)  |                   위치 컨트롤러 설정                   |
|       group(윈도우)       |               관리 그룹에 윈도우 창 추가               |
|    protocol(이름, 함수)   |               호출될 함수를 콜백으로 등록              |


<br>
<br>

## Toplevel Parameter ##
----------


## 외부 윈도우 형태 설정 ##


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | [외부 윈도우의 너비](#reference-1) | 0 | 상수 |
| height | [외부 윈도우의 높이](#reference-1) | 0 | 상수 |
| relief | 외부 윈도우의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 외부 윈도우의 테두리 두께 | 0 | 상수 |
| background=bg | 외부 윈도우의 배경 색상 | SystemButtonFace | color |
| padx | 외부 윈도우의 테두리와 내용의 가로 여백 | 0 | 상수 |
| pady | 외부 윈도우의 테두리와 내용의 세로 여백 | 0 | 상수 |


<br>
<br>

## 외부 윈도우 형식 설정 ##


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |      외부 윈도우의 마우스 커서 모양                 |       -       |   [커서 속성](#reference-2)  |
|   class_  |           클래스 설정            | - |          -          |
|   visual   |           시각적 정보 설정            | - |          -          |
|   colormap |            256 색상을 지정하는 색상 맵 설정            | - |          new          |

<br>
<br>

## 외부 윈도우 하이라이트 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    외부 윈도우가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 외부 윈도우가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    외부 윈도우가 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>
<br>

## 외부 윈도우 동작 설정 ##


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    container  |   [응용 프로그램이 포함될 컨테이너로 사용](#reference-4)   | False |  Boolean |
|   menu |  외부 윈도우에 메뉴 부착   | -  |  Menu 위젯 |

<br>
<br>

## 참고 ##
----------

<a id="reference-1"></a>

* 내부에 위젯이 존재할 경우, `width`와 `height` 설정을 무시하고 `크기 자동 조절`


<a id="reference-2"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 외부 윈도우가이 선택되지 않았을 때에도 두께가 표시됨


<a id="reference-4"></a>

* `container`를 `True`로 설정하였을 경우, 외부 윈도우의 내부에 `위젯`이 포함되어 있지 않아야 함

