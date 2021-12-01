---
layout: post
title: "Python tkinter 강좌 : 제 19강 - LabelFrame"
tagline: "Python tkinter LabelFrame"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter LabelFrame
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-19/
comments: true
toc: true
---

## LabelFrame(라벨 프레임)

`LabelFrame`을 이용하여 **다른 위젯들을 포함**하고 `캡션`이 있는 `라벨 프레임`를 생성할 수 있습니다.

<br>
<br>

## LabelFrame 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

def check():
    label.config(text=RadioVariety_1.get())
    
labelframe=tkinter.LabelFrame(window, text="플랫폼 선택")
labelframe.pack()

RadioVariety_1=tkinter.StringVar()
RadioVariety_1.set("미선택")

radio1=tkinter.Radiobutton(labelframe, text="Python", value="가능", variable=RadioVariety_1, command=check)
radio1.pack()
radio2=tkinter.Radiobutton(labelframe, text="C/C++", value="부분 가능", variable=RadioVariety_1, command=check)
radio2.pack()
radio3=tkinter.Radiobutton(labelframe, text="JSON", value="불가능", variable=RadioVariety_1, command=check)
radio3.pack()
label=tkinter.Label(labelframe, text="None")
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def check():
    label.config(text=RadioVariety_1.get())
    
labelframe=tkinter.LabelFrame(window, text="플랫폼 선택")
labelframe.pack()

RadioVariety_1=tkinter.StringVar()
RadioVariety_1.set("미선택")

radio1=tkinter.Radiobutton(labelframe, text="Python", value="가능", variable=RadioVariety_1, command=check)
radio1.pack()
radio2=tkinter.Radiobutton(labelframe, text="C/C++", value="부분 가능", variable=RadioVariety_1, command=check)
radio2.pack()
radio3=tkinter.Radiobutton(labelframe, text="JSON", value="불가능", variable=RadioVariety_1, command=check)
radio3.pack()
label=tkinter.Label(labelframe, text="None")
label.pack()

{% endhighlight %}

`tkinter.LabelFrame(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `라벨 프레임의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `라벨 프레임의 속성`을 설정합니다.

`라벨 프레임` 안에 `위젯`이 포함되어 있지 않을 경우, **표시되지 않습니다.**

<br>
<br>

## LabelFrame Parameter

### 라벨 프레임 텍스트 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| text |    라벨 프레임에 표시할 문자열     |       -       | - |
| labelwidget |    라벨 프레임에 표시할 위젯     |       -       | - |
| labelanchor |    라벨 프레임의 문자열 위치 설정     |       nw       | n, e, w, s, ne, nw, se, sw |

<br>

### 라벨 프레임 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | [라벨 프레임의 너비](#reference-1) | 0 | 상수 |
| height | [라벨 프레임의 높이](#reference-1) | 0 | 상수 |
| relief | 라벨 프레임의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 라벨 프레임의 테두리 두께 | 2 | 상수 |
| background=bg | 라벨 프레임의 배경 색상 | SystemButtonFace | color |
| foreground=fg | 라벨 프레임의 문자열 색상 | SystemButtonFace | color |
| padx | 라벨 프레임의 테두리와 내용의 가로 여백 | 0 | 상수 |
| pady | 라벨 프레임의 테두리와 내용의 세로 여백 | 0 | 상수 |

<br>

### 라벨 프레임 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  font   |       라벨 프레임의 문자열 글꼴 설정              |    TkDefaultFont    |      font        |
|  cursor  |      라벨 프레임의 마우스 커서 모양                 |       -       |   [커서 속성](#reference-2)  |
|   class_   |           클래스 설정            | - |          -          |
|   visual   |           시각적 정보 설정            | - |          -          |
|   colormap |            256 색상을 지정하는 색상 맵 설정            | - |          new          |

<br>

### 라벨 프레임 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    라벨 프레임이가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 라벨 프레임이가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    라벨 프레임이가 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>

### 라벨 프레임 동작 설정

|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    container  |   [응용 프로그램이 포함될 컨테이너로 사용](#reference-4)   | False |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* 내부에 위젯이 존재할 경우, `width`와 `height` 설정을 무시하고 `크기 자동 조절`

<a id="reference-3"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-4"></a>

* `highlightbackground`를 설정하였을 경우, 라벨 프레임이 선택되지 않았을 때에도 두께가 표시됨

* `container`를 `True`로 설정하였을 경우, 라벨 프레임의 내부에 `위젯`이 포함되어 있지 않아야 함
