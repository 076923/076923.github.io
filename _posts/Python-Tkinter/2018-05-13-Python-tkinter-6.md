---
layout: post
title: "Python tkinter 강좌 : 제 6강 - Checkbutton"
tagline: "Python tkinter Checkbutton"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Checkbutton
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-6/
comments: true
toc: true
---

## Checkbutton(체크버튼)

`Checkbutton`을 이용하여 `옵션` 등을 **다중 선택**하기 위한 `체크버튼`을 생성할 수 있습니다.

<br>
<br>

## Checkbutton 사용

{% highlight Python %}
import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")
window.resizable(False, False)

def flash():
    checkbutton1.flash()

CheckVariety_1=tkinter.IntVar()
CheckVariety_2=tkinter.IntVar()

checkbutton1=tkinter.Checkbutton(window, text="O", variable=CheckVariety_1, activebackground="blue")
checkbutton2=tkinter.Checkbutton(window, text="△", variable=CheckVariety_2)
checkbutton3=tkinter.Checkbutton(window, text="X", variable=CheckVariety_2, command=flash)

checkbutton1.pack()
checkbutton2.pack()
checkbutton3.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def flash():
    checkbutton1.flash()

CheckVariety_1=tkinter.IntVar()
CheckVariety_2=tkinter.IntVar()

checkbutton1=tkinter.Checkbutton(window, text="O", variable=CheckVariety_1, activebackground="blue")
checkbutton2=tkinter.Checkbutton(window, text="△", variable=CheckVariety_2)
checkbutton3=tkinter.Checkbutton(window, text="X", variable=CheckVariety_2, command=flash)

checkbutton1.pack()
checkbutton2.pack()
checkbutton3.pack()

{% endhighlight %}

`tkinter.Checkbutton(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `체크버튼의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `체크버튼의 속성`을 설정합니다.

매개변수 중 `command`를 이용하여 `사용자 정의 함수 : flash()`을 실행시킬 수 있습니다.

<br>
<br>

## Checkbutton Method

|    이름    |      의미      |
|:----------:|:--------------:|
|  select()  |    체크 상태   |
| deselect() |    해제 상태   |
|  toggle()  |      토글      |
|  invoke()  | 체크 버튼 실행 |
|   flash()  |     깜빡임     |

- Tip : `toggle()` : 체크일 경우 해제되며, 해제일 경우 체크됨

- Tip : `invoke()` : 체크버튼을 클릭했을 때와 동일한 실행

- Tip : `flash()` : `normal` 상태 배경 색상과 `active` 상태 배경 색 사이에서 깜빡임

<br>
<br>

## Checkbutton Parameter

### 체크버튼 문자열 설정

|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     text     |            체크버튼에 표시할 문자열           |    -   |                  -                  |
| textvariable |     체크버튼에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    anchor    |     체크버튼안의 문자열 또는 이미지의 위치    | center | n, ne, e, se, s, sw, w, nw, center  |
|    justify   | 체크버튼의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |
|  wraplength  |           자동 줄내림 설정 너비           |    0   |                 상수                |

<br>

### 체크버튼 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            체크버튼의 너비           |         0        |                    상수                    |
|     height     |            체크버튼의 높이           |         0        |                    상수                    |
|     relief     |        체크버튼의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|     overrelief      |        체크버튼에 마우스를 올렸을 때 체크버튼의 테두리 모양        |       raised       | flat, groove, raised, ridge, solid, sunken |
|  background=bg |           체크버튼의 배경 색상        | SystemButtonFace |                    color                 |
|  foreground=fg |          체크버튼의 문자열 색상         | SystemButtonFace |                    color                   |
|  selectcolor |          체크버튼 상태의 배경 색       | SystemWindow |                    color                  |
|      padx      | 체크버튼의 테두리와 내용의 가로 여백 |         1        |                    상수                    |
|      pady      | 체크버튼의 테두리와 내용의 세로 여백 |         1        |                    상수                    |

<br>

### 체크버튼 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  bitmap  |                체크버튼에 포함할 기본 이미지                |       -       | info, warning, error, question,   questhead, hourglass, gray12, gray25, gray50, gray75 |
|   image  |                체크버튼에 포함할 임의 이미지                |       -       |                                            -                                           |
|   selectimage  |            [체크버튼의 체크 상태일 때 표시할 임의 이미지](#reference-1)    |       -       |                   -           |
| compound | 체크버튼에 문자열과 이미지를 동시에 표시할 때 이미지의 위치 |      none     |                         bottom, center, left, none, right, top                         |
|   font   |                체크버튼의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 체크버튼의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-2)                                   |

<br>

### 체크버튼 상태 설정

|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-3), active, disabled |
|  activebackground  |   active 상태일 때 체크버튼의 배경 색상   |  SystemButtonFace  |           color          |
|  activeforeground  |  active 상태일 때 체크버튼의 문자열 색상  |  SystemButtonText  |           color          |
| disabledforeground | disabeld 상태일 때 체크버튼의 문자열 색상 | SystemDisabledText |           color          |

<br>

### 체크버튼 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    체크버튼이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 체크버튼이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    체크버튼이 선택되었을 때 두께 [(두께 설정)](#reference-4)     |         0         | 상수 |

<br>

### 체크버튼 동작 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    command |    체크버튼이 active 상태일 때 실행하는 메서드(함수)   | - |  메서드, 함수 |
|    variable |    체크버튼의 상태를 저장할 제어 변수   | - |  tkinter.IntVar() |
|    onvalue |    체크버튼이 체크 상태일 때 연결된 제어 변수의 값  | True |  Boolean |
|    offvalue |    체크버튼이 해제 상태일 때 연결된 제어 변수의 값   | False |  Boolean |
|    indicatoron |    체크버튼의 위젯 일치화 여부 | True|  [Boolean](#reference-5) |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* `selectimage` 매개변수는 `image` 매개변수에 이미지가 포함되어 있어야 작동됨

<a id="reference-3"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-4"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)

<a id="reference-5"></a>

* `highlightbackground`를 설정하였을 경우, 체크버튼이 선택되지 않았을 때에도 두께가 표시됨

* `indicatoron`을 False로 설정할 경우, 체크 선택 부분이 사라지고 버튼처럼 변환됨
