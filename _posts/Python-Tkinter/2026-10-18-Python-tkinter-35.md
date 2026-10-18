---
layout: post
title: "Python tkinter 강좌 : 제 35강 - 주기 실행"
tagline: "Python tkinter after"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter after, tkinter after_cancel, tkinter timer, tkinter 타이머, tkinter 시계, tkinter 애니메이션, tkinter update, tkinter mainloop, tkinter threading
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-35/
comments: true
toc: true
---

## 주기 실행(after)

`tkinter`의 `mainloop()`는 **이벤트를 기다리며 화면을 갱신하는 반복문**입니다.

이 반복문 안에서 `time.sleep()`이나 긴 `while` 반복문을 실행하면 이벤트 처리가 멈춰 **창이 응답하지 않는 상태**가 됩니다.

`after()`는 함수를 **지정한 시간 뒤에 한 번 실행하도록 예약**하는 메서드로, `mainloop()`를 멈추지 않고 시계, 타이머, 애니메이션처럼 **주기적으로 반복되는 작업**을 구현할 때 사용합니다.

- `mainloop 이해하기` : [1강 바로가기][1강]

<br>
<br>

## after 사용

{% highlight Python %}

import tkinter
import time

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="", font=("", 40))
label.pack(expand=True)

def clock():
    label.config(text=time.strftime("%H:%M:%S"))
    window.after(1000, clock)

clock()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def clock():
    label.config(text=time.strftime("%H:%M:%S"))
    window.after(1000, clock)

{% endhighlight %}

`위젯이름.after(시간, 함수)`를 이용하여 `시간(밀리초)`이 지난 뒤에 `함수`를 **한 번** 실행하도록 예약할 수 있습니다.

`after()`는 함수를 예약만 하고 **즉시 반환**되므로, 이후의 코드와 `mainloop()`는 멈추지 않고 계속 실행됩니다.

`clock` 함수는 현재 시각을 라벨에 표시한 뒤, 자기 자신을 `1초 뒤`에 다시 예약합니다.

함수가 실행될 때마다 다음 실행을 다시 예약하므로, 결과적으로 **1초마다 반복**되는 시계가 됩니다.

<br>

{% highlight Python %}

clock()

{% endhighlight %}

예약은 함수 안에서 이루어지므로, `mainloop()` 전에 함수를 **한 번 직접 호출**해 반복을 시작합니다.

<br>

- Tip : `after()`에 전달하는 함수에는 괄호를 붙이지 않습니다. `window.after(1000, clock())`으로 작성하면 `clock`이 즉시 실행되고, 반환값인 `None`이 예약됩니다.

- Tip : 함수에 매개변수를 전달하려면 `window.after(1000, 함수, 매개변수1, 매개변수2, ...)`처럼 시간과 함수 뒤에 **인수를 나열**합니다. 람다 함수를 사용하는 방법은 [33강 바로가기][33강]를 참고합니다.

- Tip : 예약된 시간은 **최소 대기 시간**입니다. 다른 이벤트를 처리하느라 바쁘면 예약보다 늦게 실행될 수 있으므로, 정확한 시각이 필요하면 `time` 모듈로 실제 시각을 읽어 표시합니다.

<br>
<br>

## after_cancel 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

count = 10
after_id = None

label=tkinter.Label(window, text=str(count), font=("", 40))
label.pack(expand=True)

def countdown():
    global count, after_id
    if count > 0:
        count -= 1
        label.config(text=str(count))
        after_id = window.after(1000, countdown)
    else:
        label.config(text="종료")
        after_id = None

def start():
    global count
    if after_id is None:
        count = 10
        label.config(text=str(count))
        countdown()

def stop():
    global after_id
    if after_id is not None:
        window.after_cancel(after_id)
        after_id = None

button1=tkinter.Button(window, text="시작", width=10, command=start)
button2=tkinter.Button(window, text="정지", width=10, command=stop)
button1.pack(side="left", expand=True)
button2.pack(side="left", expand=True)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

after_id = window.after(1000, countdown)

{% endhighlight %}

`after()`는 예약을 구분하는 **식별자(ID)**를 반환합니다.

예약을 취소하려면 이 식별자를 보관해 두어야 하므로, 전역 변수 `after_id`에 저장합니다.

<br>

{% highlight Python %}

window.after_cancel(after_id)

{% endhighlight %}

`위젯이름.after_cancel(식별자)`를 이용하여 아직 실행되지 않은 **예약을 취소**할 수 있습니다.

`정지` 버튼을 누르면 다음 예약이 취소되어 카운트다운이 멈추고, `시작` 버튼을 누르면 처음부터 다시 시작합니다.

<br>

- Tip : `start` 함수에서 `after_id`가 `None`일 때만 시작하도록 확인하지 않으면, 버튼을 여러 번 누를 때마다 **반복이 중복으로 예약**되어 카운트다운이 빨라집니다.

- Tip : 이미 실행된 예약의 식별자를 `after_cancel`에 전달해도 오류는 발생하지 않습니다.

<br>
<br>

## 애니메이션 적용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

canvas=tkinter.Canvas(window, width=640, height=400, bg="white")
canvas.pack()

ball = canvas.create_oval(0, 180, 40, 220, fill="red", outline="")
dx = 5

def move():
    global dx
    x1, y1, x2, y2 = canvas.coords(ball)
    if x2 >= 640 or x1 <= 0:
        dx = -dx
    canvas.move(ball, dx, 0)
    window.after(16, move)

move()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def move():
    global dx
    x1, y1, x2, y2 = canvas.coords(ball)
    if x2 >= 640 or x1 <= 0:
        dx = -dx
    canvas.move(ball, dx, 0)
    window.after(16, move)

{% endhighlight %}

`move` 함수는 공의 현재 좌표를 읽어 벽에 닿으면 이동 방향을 반대로 바꾸고, `canvas.move()`로 공을 `dx`만큼 이동시킵니다.

`16밀리초`마다 반복하면 초당 약 60번 화면이 갱신되어 **부드러운 애니메이션**이 됩니다.

- `Canvas 사용하기` : [15강 바로가기][15강]

<br>

- Tip : 반복 간격을 짧게 설정할수록 부드러워지지만, 실행 시간이 간격보다 길어지면 화면이 끊기므로 반복 함수 안의 작업은 **짧게** 유지합니다.

<br>
<br>

## after Method

| 메서드 | 의미 | 반환값 |
|:---:|:---:|:---:|
| after(ms, func, *args) | ms 밀리초 뒤에 func(*args)를 한 번 실행 | 예약 식별자 |
| after(ms) | ms 밀리초 동안 대기 (이벤트 처리 중단) | None |
| after_idle(func, *args) | 처리할 이벤트가 없을 때 func(*args)를 실행 | 예약 식별자 |
| after_cancel(id) | 예약 취소 | None |
| update() | 대기 중인 모든 이벤트와 화면 갱신을 즉시 처리 | None |
| update_idletasks() | 화면 갱신만 즉시 처리 (사용자 입력은 처리하지 않음) | None |

<br>

- Tip : `after(ms)`처럼 함수 없이 호출하면 `time.sleep()`과 같이 **창이 멈추므로**, 주기 실행에는 반드시 함수를 함께 전달합니다.

- Tip : 긴 반복문 안에서 화면을 갱신해야 한다면 반복마다 `update_idletasks()`를 호출합니다. `update()`는 사용자 입력까지 처리해 함수가 **재진입**될 수 있으므로 신중하게 사용합니다.

<br>
<br>

## 긴 작업 처리

{% highlight Python %}

import tkinter
import threading
import time

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="대기 중", font=("", 20))
label.pack(expand=True)

def work():
    time.sleep(3)
    window.after(0, label.config, {"text": "작업 완료"})

def start():
    label.config(text="작업 중...")
    threading.Thread(target=work, daemon=True).start()

button=tkinter.Button(window, text="시작", width=10, command=start)
button.pack(expand=True)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def work():
    time.sleep(3)
    window.after(0, label.config, {"text": "작업 완료"})

{% endhighlight %}

파일 다운로드나 계산처럼 **수 초 이상 걸리는 작업**은 `after()`로 나누기 어려우므로, `threading` 모듈로 별도의 스레드에서 실행합니다.

작업 중에도 `mainloop()`는 계속 동작하므로 창이 멈추지 않습니다.

<br>

`tkinter` 위젯은 **메인 스레드에서만** 조작할 수 있으므로, 작업 스레드에서 `label.config()`를 직접 호출하면 오류가 발생하거나 프로그램이 비정상 종료될 수 있습니다.

`window.after(0, 함수, 매개변수)`로 예약하면 함수가 **메인 스레드에서 실행**되므로, 작업 스레드에서 안전하게 위젯을 변경할 수 있습니다.

- `스레드 사용하기` : [31강 바로가기][31강]

<br>

- Tip : `daemon=True`로 설정하면 창을 닫을 때 작업 스레드가 남아 있어도 프로그램이 **함께 종료**됩니다.

- Tip : 진행률을 표시하려면 작업 스레드에서 `window.after(0, progressbar.step)`을 주기적으로 예약하거나, `queue` 모듈로 값을 전달하고 메인 스레드에서 `after()`로 주기적으로 꺼내어 표시합니다.

[1강]: https://076923.github.io/posts/Python-tkinter-1/
[15강]: https://076923.github.io/posts/Python-tkinter-15/
[33강]: https://076923.github.io/posts/Python-tkinter-33/
[31강]: https://076923.github.io/posts/Python-31/
