---
layout: post
title: "Python tkinter 강좌 : 제 7강 – Radiobutton"
tagline: "Python tkinter Radiobutton"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Radiobutton
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-7/
comments: true
---

## Radiobutton(라디오버튼) ##
----------

`Radiobutton`을 이용하여 `옵션` 등을 **단일 선택**하기 위한 `라디오버튼`을 생성할 수 있습니다.

<br>
<br>

## Radiobutton 사용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")
window.resizable(False, False)

def check():
    label.config(text= "RadioVariety_1 = " + str(RadioVariety_1.get()) + "\n" +
                       "RadioVariety_2 = " + str(RadioVariety_2.get()) + "\n\n" +
                       "Total = "          + str(RadioVariety_1.get() + RadioVariety_2.get()))

RadioVariety_1=tkinter.IntVar()
RadioVariety_2=tkinter.IntVar()

radio1=tkinter.Radiobutton(window, text="1번", value=3, variable=RadioVariety_1, command=check)
radio1.pack()

radio2=tkinter.Radiobutton(window, text="2번(1번)", value=3, variable=RadioVariety_1, command=check)
radio2.pack()

radio3=tkinter.Radiobutton(window, text="3번", value=9, variable=RadioVariety_1, command=check)
radio3.pack()

label=tkinter.Label(window, text="None", height=5)
label.pack()

radio4=tkinter.Radiobutton(window, text="4번", value=12, variable=RadioVariety_2, command=check)
radio4.pack()

radio5=tkinter.Radiobutton(window, text="5번", value=15, variable=RadioVariety_2, command=check)
radio5.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def check():
    label.config(text= "RadioVariety_1 = " + str(RadioVariety_1.get()) + "\n" +
                       "RadioVariety_2 = " + str(RadioVariety_2.get()) + "\n\n" +
                       "Total = "          + str(RadioVariety_1.get() + RadioVariety_2.get()))

RadioVariety_1=tkinter.IntVar()
RadioVariety_2=tkinter.IntVar()

radio1=tkinter.Radiobutton(window, text="1번", value=3, variable=RadioVariety_1, command=check)
radio1.pack()

radio2=tkinter.Radiobutton(window, text="2번(1번)", value=3, variable=RadioVariety_1, command=check)
radio2.pack()

radio3=tkinter.Radiobutton(window, text="3번", value=9, variable=RadioVariety_1, command=check)
radio3.pack()

label=tkinter.Label(window, text="None", height=5)
label.pack()

radio4=tkinter.Radiobutton(window, text="4번", value=12, variable=RadioVariety_2, command=check)
radio4.pack()

radio5=tkinter.Radiobutton(window, text="5번", value=15, variable=RadioVariety_2, command=check)
radio5.pack()

{% endhighlight %}


`tkinter.Radiobutton(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `라디오버튼의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `라디오버튼의 속성`을 설정합니다.

파라미터 중 `command`를 이용하여 `사용자 정의 함수 : check()`을 실행시킬 수 있습니다.

파라미터 중 `value`를 이용하여 `라디오버튼의 값`을 설정할 수 있습니다. `value`의 값이 겹치는 경우 **같이 선택됩니다.**

파라미터 중 `variable`를 이용하여 `tkinter.IntVar()`의 그룹이 같을 경우 하나의 묶음으로 간주하며 'value'의 값이 저장됩니다.

`tkinter.IntVar()`에 저장된 `value` 값은 `변수이름.get()`을 통하여 불러올 수 있습니다.

<br>
<br>

## Radiobutton Method ##
----------

|    이름    |      의미      |
|:----------:|:--------------:|
|  select()  |    체크 상태   |
| deselect() |    해제 상태   |
|  invoke()  | 체크 버튼 실행 |
|   flash()  |     깜빡임     |

<br>

* Tip : `invoke()` : 라디오버튼을 클릭했을 때와 동일한 실행
* Tip : `flash()` : `normal` 상태 배경 색상과 `active` 상태 배경 색상 사이에서 깜빡임

<br>
<br>

## Radiobutton Parameter ##
----------

## 라디오버튼 문자열 설정 ##


|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     text     |            라디오버튼에 표시할 문자열           |    -   |                  -                  |
| textvariable |     라디오버튼에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    anchor    |     라디오버튼안의 문자열 또는 이미지의 위치    | center | n, ne, e, se, s, sw, w, nw, center  |
|    justify   | 라디오버튼의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |
|  wraplength  |           자동 줄내림 설정 너비           |    0   |                 상수                |

<br>
<br>

## 라디오버튼 형태 설정 ##


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            라디오버튼의 너비           |         0        |                    상수                    |
|     height     |            라디오버튼의 높이           |         0        |                    상수                    |
|     relief     |        라디오버튼의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|     overrelief      |        라디오버튼에 마우스를 올렸을 때 라디오버튼의 테두리 모양        |       raised       | flat, groove, raised, ridge, solid, sunken |
|  background=bg |           라디오버튼의 배경 색상        | SystemButtonFace |                    color                 |
|  foreground=fg |          라디오버튼의 문자열 색상         | SystemButtonFace |                    color                   |
|  selectcolor |          라디오버튼 상태의 배경 색상       | SystemWindow |                    color                   |
|      padx      | 라디오버튼의 테두리와 내용의 가로 여백 |         1        |                    상수                    |
|      pady      | 라디오버튼의 테두리와 내용의 세로 여백 |         1        |                    상수                    |

<br>
<br>

## 라디오버튼 형식 설정 ##


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  bitmap  |                라디오버튼에 포함할 기본 이미지                |       -       | info, warning, error, question,   questhead, hourglass, gray12, gray25, gray50, gray75 |
|   image  |                라디오버튼에 포함할 임의 이미지                |       -       |                                            -                                           |
|   selectimage  |            [라디오버튼의 체크 상태일 때 표시할 임의 이미지](#reference-1)    |       -       |                   -           |
| compound | 라디오버튼에 문자열과 이미지를 동시에 표시할 때 이미지의 위치 |      none     |                         bottom, center, left, none, right, top                         |
|   font   |                라디오버튼의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 라디오버튼의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-2)                                   |

<br>
<br>

## 라디오버튼 상태 설정 ##


|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-3), active, disabled |
|  activebackground  |   active 상태일 때 라디오버튼의 배경 색상   |  SystemButtonFace  |           color          |
|  activeforeground  |  active 상태일 때 라디오버튼의 문자열 색상  |  SystemButtonText  |           color          |
| disabledforeground | disabeld 상태일 때 라디오버튼의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

## 라디오버튼 하이라이트 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    라디오버튼이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 라디오버튼이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    라디오버튼이 선택되었을 때 두께 [(두께 설정)](#reference-4)     |         0         | 상수 |

<br>
<br>

## 라디오버튼 동작 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    command |    라디오버튼이 active 상태일 때 실행하는 메소드(함수)   | - |  메소드, 함수 |
|    variable |    라디오버튼의 상태를 저장할 제어 변수   | - |  tkinter.IntVar(), tkinter.StringVar() |
|    value |     라디오버튼이 가지고 있는 값 | - |  [variable의 속성](#reference-5) |
|    indicatoron |    라디오버튼의 위젯 일치화 여부 | True|  [Boolean](#reference-6) |

<br>
<br>

## 참고 ##
----------

<a id="reference-1"></a>

* `selectimage` 파라미터는 `image` 파라미터에 이미지가 포함되어 있어야 작동됨

<a id="reference-2"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-3"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)

<a id="reference-4"></a>

* `highlightbackground`를 설정하였을 경우, 라디오버튼이 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-5"></a>

* `variable`의 값이 `tkinter.StringVar()`일 경우, `value`에 `str` 속성을 할당할 수 있음

<a id="reference-6"></a>

* `indicatoron`을 False로 설정할 경우, 체크 선택 부분이 사라지고 버튼처럼 변환됨
