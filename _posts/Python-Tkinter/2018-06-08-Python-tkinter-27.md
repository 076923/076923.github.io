---
layout: post
title: "Python tkinter 강좌 : 제 27강 - Progressbar"
tagline: "Python tkinter Progressbar"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Progressbar, tkinter ttk Progressbar, tkinter 진행 표시줄
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-27/
comments: true
toc: true
---

## Progressbar(프로그래스 바)

`Progressbar`을 이용하여 **현재 진행 상황**을 표시하는 `프로그래스 바`를 생성할 수 있습니다.

<br>
<br>

## Progressbar 사용

{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

progressbar=tkinter.ttk.Progressbar(window, maximum=100, mode="indeterminate")
progressbar.pack()

progressbar.start(50)

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

progressbar=tkinter.ttk.Progressbar(window, maximum=100, mode="indeterminate")
progressbar.pack()

progressbar.start(50)

{% endhighlight %}

`tkinter.ttk.Progressbar(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 `프로그래스 바의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `프로그래스 바의 속성`을 설정합니다.

`mode="indeterminate"`는 **작업량을 알 수 없을 때** 사용하며, `start()`를 호출하면 표시기가 좌우로 반복 이동합니다.

<br>
<br>

## Progressbar Method

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     start(ms)    |   시작  |      프로그래스 바가 `밀리 초`마다 `step()`만큼 움직임 (생략 시 50ms)       |
| step(value) |    값 증가   | 현재 표시되는 값에서 `value`만큼 증가 (생략 시 1.0) |
| stop() |    종료  | 프로그래스 바 작동 종료 |

- Tip : `step()`으로 증가한 값이 `maximum`을 넘으면 `0`부터 다시 시작합니다. 예를 들어 `maximum=100`일 때 `step(150)`을 실행하면 값은 `50`이 됩니다.

<br>

## determinate 모드 사용

{% highlight Python %}

progressbar=tkinter.ttk.Progressbar(window, maximum=100, mode="determinate")
progressbar.pack()

progressbar["value"]=30

{% endhighlight %}

`mode="determinate"`는 **작업량을 알고 있을 때** 사용하며, `maximum`에 대한 `value`의 비율만큼 표시기가 채워집니다.

`프로그래스 바["value"]=값` 또는 `프로그래스 바.config(value=값)`으로 현재 값을 직접 설정할 수 있습니다.

- Tip : 반복문 안에서 값을 갱신할 때는 `window.update_idletasks()`를 호출해야 화면에 즉시 반영됩니다. 오래 걸리는 작업은 `window.after()`나 `스레드`로 분리해야 윈도우 창이 멈추지 않습니다.

<br>
<br>

## Progressbar Parameter

### 프로그래스 바 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| length | 프로그래스 바의 길이 (`orient` 방향의 크기) | 100 | 상수 |
| orient | 프로그래스 바의 표시 방향 | horizontal | vertical, horizontal |
| mode | 프로그래스 바의 표시 스타일 | determinate | [determinate , indeterminate](#reference-1)    |

<br>

### 프로그래스 바 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |      프로그래스 바의 마우스 커서 모양                 |       -       |          [커서 속성](#reference-2)             |
|  class_  |      클래스 설정                 |       -       |      -    |  
|   maximum |          프로그래스 바의 최댓값 설정 | 100 |   상수 |
|   value |  프로그래스 바의 현재값을 설정 | 0 |    상수        |
|   variable |  프로그래스 바의 현재값을 가져올 변수 | - |    IntVar, DoubleVar        |
|   phase |  indeterminate 모드에서 애니메이션 단계를 나타내는 값 (`start()` 호출 시 자동 증가) | 0 |    상수        |

<br>

### 프로그래스 바 동작 설정

|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* mode 매개변수

    - `determinate` : 표시기가 처음부터 끝까지 채워짐
    - `indeterminate` : 표시기가 처음부터 끝까지 반복 이동

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor
