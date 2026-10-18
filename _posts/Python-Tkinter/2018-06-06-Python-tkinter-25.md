---
layout: post
title: "Python tkinter 강좌 : 제 25강 - Spinbox"
tagline: "Python tkinter Spinbox"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Spinbox, tkinter validatecommand, tkinter ttk Spinbox
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-25/
comments: true
toc: true
---

## Spinbox(수치 조정 기입창)

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-25/1.webp" class="lazyload" width="100%" height="100%"/>

`Spinbox`을 이용하여 **수치를 조정하고 입력**받는 `수치 조정 기입창`를 생성할 수 있습니다.

<br>
<br>

## Spinbox 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="숫자를 입력하세요.", height=3)
label.pack()

def value_check(self):
    label.config(text="숫자를 입력하세요.")
    valid = False
    if self.isdigit():
        if (int(self) <= 50 and int(self) >= 0):
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    label.config(text=str(self) + "를 입력하셨습니다.\n올바른 값을 입력하세요.")

validate_command=(window.register(value_check), '%P')
invalid_command=(window.register(value_error), '%P')

spinbox=tkinter.Spinbox(window, from_ = 0, to = 50, validate = 'all', validatecommand = validate_command, invalidcommand=invalid_command)
spinbox.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

label=tkinter.Label(window, text="숫자를 입력하세요.", height=3)
label.pack()

def value_check(self):
    label.config(text="숫자를 입력하세요.")
    valid = False
    if self.isdigit():
        if (int(self) <= 50 and int(self) >= 0):
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    label.config(text=str(self) + "를 입력하셨습니다.\n올바른 값을 입력하세요.")

validate_command=(window.register(value_check), '%P')
invalid_command=(window.register(value_error), '%P')

spinbox=tkinter.Spinbox(window, from_ = 0, to = 50, validate = 'all', validatecommand = validate_command, invalidcommand=invalid_command)
spinbox.pack()

{% endhighlight %}

`tkinter.Spinbox(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 `수치 조정 기입창의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `수치 조정 기입창의 속성`을 설정합니다.

`사용자 정의 함수`에 `래핑`하여 `콜백`을 포함하여야 `유효성 검사`를 실행할 수 있습니다.

`validate`를 이용하여 `유효성 검사`를 `반응 조건`을 설정합니다.

`validatecommand`를 이용하여 유효성을 검사하며, `invalidcommand`를 통하여 올바르지 않을 때의 함수를 실행합니다.

`window.register(함수)`로 Python 함수를 `Tcl 명령`으로 등록하고, `'%P'`와 같은 `치환 코드`를 튜플로 함께 전달하면 **값이 바뀔 때마다 함수의 매개변수로 전달**됩니다.

`value_check` 함수가 `True`를 반환하면 입력이 허용되고, `False`를 반환하면 입력이 취소된 뒤 `invalidcommand`의 함수가 실행됩니다.

- Tip : `spinbox.get()`의 반환값은 항상 `문자열(str)`이므로, 숫자로 계산하려면 `int()`나 `float()`로 변환해야 합니다.

- Tip : `Python 3.7` 이상에서는 `tkinter.ttk.Spinbox`도 사용할 수 있습니다. `ttk.Spinbox`는 테마가 적용된 모양을 가지며, `set(값)` 메서드로 값을 변경할 수 있지만 `background`, `buttonbackground` 등의 색상 매개변수는 지원하지 않습니다.

<br>
<br>

## Spinbox Method

### 수치 조정 기입창 메서드

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     insert(index, "문자열")    |    문자열 삽입   |       해당 `index` 위치에 `문자열`을 삽입       |
| delete(start_index, end_index) |    문자열 삭제   | `start_index`부터 `end_index`까지의 문자열 삭제 |
|   get()  |    문자열 반환   | 수치 조정 기입창의 현재 값을 `문자열`로 반환 |
|          index(index)          |    인덱스 반환   |      `end`, `insert` 등의 `index`에 해당하는 위치를 숫자로 반환     |
|          icursor(index)          |    키보드 커서 이동   |      해당 `index` 위치로 키보드 커서를 이동     |
|           identify(x, y)         | 요소 식별 |   `x, y` 위치의 위젯 요소 반환 (`entry`, `buttonup`, `buttondown`, `''`)  |
|           invoke(button)         | 버튼 실행 |  `buttonup` 또는 `buttondown` 실행  |

<br>

### 수치 조정 기입창 콜백

| 이름 |           의미          |                               설명                               |
|:----:|:-----------------------:|:----------------------------------------------------------------:|
|  %d  |      액션 코드 반환     | `삭제=0`, `삽입=1`, `그 외(포커스 이동, textvariable 변경 등)=-1` |
|  %i  | 텍스트 수정 인덱스 반환 |    `삽입 또는 삭제=해당 index`, `포커스인 또는 포커스 아웃=-1`   |
|  %P  |      텍스트 부여 값     |               변경이 허용될 경우 텍스트에 부여될 값 (변경 후 전체 값)              |
|  %s  |  텍스트가 변경되기 전 값  |                 텍스트가 변경되기 전의 전체 값                 |
|  %S  |  삽입 또는 삭제되는 문자열  |                  이번 변경에서 삽입되거나 삭제되는 문자열                  |
|  %v  |     validate 현재 값    |          수치 조정 기입창의 validate 매개변수의 현재 값          |
|  %V  |        콜백 확인        |               어떤 호출로 인하여 콜백되었는지 확인 (`focusin`, `focusout`, `key`, `forced`)               |
|  %W  |       위젯의 이름       |                     호출된 위젯의 이름을 확인                    |

- Tip : 치환 코드로 전달되는 값은 모두 `문자열`입니다. `%d`의 `0`, `1`, `-1`도 `"0"`, `"1"`, `"-1"`로 전달됩니다.

<br>
<br>

## Spinbox Parameter

### 수치 조정 기입창 텍스트 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| textvariable | 수치 조정 기입창에 표시할 문자열을 가져올 변수 | - | - |
| justify | 수치 조정 기입창의 문자열 정렬 방법 | left | center, left, right |

<br>

### 수치 조정 기입창 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | 수치 조정 기입창의 너비 | 20 | 상수 |
| relief | 수치 조정 기입창의 테두리 모양 | sunken | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 수치 조정 기입창의 테두리 두께 | 1 | 상수 |
| background=bg | 수치 조정 기입창의 배경 색상 | SystemWindow | color |
| foreground=fg | 수치 조정 기입창의 문자열 색상 | SystemWindowText | color |
| insertwidth | 수치 조정 기입창의 키보드 커서 너비 | 2 | 상수 |
| insertborderwidth | 수치 조정 기입창의 키보드 커서 테두리 두께 | 0 | 상수 |
| insertbackground | 수치 조정 기입창의 키보드 커서 색상 | SystemWindowText | color |
| selectborderwidth | 수치 조정 기입창의 문자열 블록처리 테두리 두께 | 0 | 상수 |
| selectbackground | 수치 조정 기입창의 문자열 블록처리 배경 색상 | SystemHighlight | color |
| selectforeground | 수치 조정 기입창의 문자열 블록처리 문자열 색상 | SystemHighlightText | color |
| buttonbackground | 수치 조정 기입창의 수치 조정 버튼 배경 색상 | SystemButtonFace | color |

<br>

### 수치 조정 기입창 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  font   |       수치 조정 기입창의 문자열 글꼴 설정              |    TkTextFont    |      font        |
|  cursor  |      수치 조정 기입창의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |
|  buttoncursor |    수치 조정 기입창의 수치 조정 버튼 마우스 커서 모양              |       -       |                                    [커서 속성](#reference-1)                                   |
|   xscrollcommand  |          수치 조정 기입창의 가로스크롤 위젯 적용            | - |          Scrollbar위젯.set |
|   from_  |           수치 조정 기입창의 최솟값           | 0 |          상수          |
|   to   |           수치 조정 기입창의 최댓값         | 0 |          상수          |
|   increment |            수치 조정 기입창의 수치 간격            | 1 |          상수          |
|   values |            수치 조정 기입창의 사용자 정의 수치 값            | - |          [list, tuple 등](#reference-2)            |
|   format_ |            [수치 조정 기입창의 수치 표시 형식](#reference-7)            | - |          "%너비.소수자릿수f"          |
|  exportselection |     수치 조정 기입창의 선택 항목 여부 설정   |    True    |      Boolean        |

<br>

### 수치 조정 기입창 상태 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    state  |    상태 설정  | normal  | normal, readonly, disabled  |
|    activebackground |    active 상태일 때 수치 조정 기입창의 배경 색상 | SystemButtonFace |  color  |
|  readonlybackground |   readonly 상태일 때 수치 조정 기입창의 배경 색상   |  SystemButtonFace  |           color          |
|  disabledbackground |  disabled 상태일 때 수치 조정 기입창의 배경 색상  |  SystemButtonFace  |           color          |
| disabledforeground | disabled 상태일 때 수치 조정 기입창의 문자열 색상 | SystemDisabledText |           color          |

- Tip : `state="readonly"`로 설정하면 키보드로 직접 입력할 수 없고, **수치 조정 버튼으로만** 값을 변경할 수 있습니다.

<br>

### 수치 조정 기입창 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    수치 조정 기입창이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 수치 조정 기입창이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    수치 조정 기입창이 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>

### 수치 조정 기입창 동작 설정

|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    command |    수치 조정 기입창에서 수치가 조정될 때 실행하는 메서드(함수)   | - |  메서드, 함수 |
|    wrap  |  수치 조정 기입창에서 최댓값을 초과하면 최솟값으로, 최솟값 미만이면 최댓값으로 순환  | False |  Boolean |
|    insertontime |    기입창의 키보드 커서 깜빡임이 보이는 시간   | 600  |  상수(ms) |
| insertofftime | 기입창의 키보드 커서 깜빡임이 보이지 않는 시간  |  300 |  상수(ms) |
| repeatdelay | [수치 조정 버튼이 눌러진 상태에서 command 실행까지의 대기 시간](#reference-4)   |  400 |  상수(ms) |
|  repeatinterval |    [수치 조정 버튼이 눌러진 상태에서 command 실행의 반복 시간](#reference-5)    |         100         | 상수(ms) |
|    validate |    수치 조정 기입창의 유효성 검사 실행 조건  | none |  [none, focus, focusin, focusout, key, all](#reference-6) |
|    validatecommand |   유효성 검사 평가 함수  | - |  함수 |
|    invalidcommand |    validateCommand가 False를 반환 할 때 실행할 함수 | - |  함수  |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-3"></a>

* `values`를 `[5, 10, 50, 100]`으로 설정하였을 경우, `목록의 순서`로 증감

<a id="reference-4"></a>

* `highlightbackground`를 설정하였을 경우, 수치 조정 기입창이 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-5"></a>

* `repeatdelay=100` 일 경우, **누르고 있기 시작한 0.1초 후**에 `command`가 실행됨

<a id="reference-6"></a>

* `repeatdelay=1000`, `repeatinterval=100` 일 경우, **1초 후에 command가 실행되며 0.1초마다 버튼을 뗄 때까지** `command`가 계속 실행됨

* validate 매개변수

    - `none` : 수치 조정 기입창의 유효성 검사 실행하지 않음
    - `focus` : 수치 조정 기입창이 포커스를 받거나 잃을 때 validateCommand 실행
    - `focusin` : 수치 조정 기입창이 포커스를 받을 때 validateCommand 실행
    - `focusout` : 수치 조정 기입창이 포커스를 잃을 때 validateCommand 실행
    - `key` : 수치 조정 기입창이 수정될 경우 validateCommand 실행
    - `all` : 수치 조정 기입창의 모든 validate에 대해 validateCommand 실행

<a id="reference-7"></a>

* `format_="%5.2f"`로 설정하였을 경우, 수치가 `전체 5자리, 소수점 둘째 자리`까지 표시됨 (`format="%5.2f"`로도 설정 가능)
