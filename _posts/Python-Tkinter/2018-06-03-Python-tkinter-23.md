---
layout: post
title: "Python tkinter 강좌 : 제 23강 - Bind"
tagline: "Python tkinter Bind"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Bind, tkinter event, tkinter unbind, tkinter MouseWheel
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-23/
comments: true
toc: true
---

## Bind(이벤트 실행)

`Bind`를 이용하여 위젯들의 `이벤트`와 실행할 `함수`를 설정할 수 있습니다.

<br>
<br>

## Bind 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

width=1

def drawing(event):
    if width>0:
        x1=event.x-1
        y1=event.y-1
        x2=event.x+1
        y2=event.y+1
        canvas.create_oval(x1, y1, x2, y2, fill="blue", width=width)

def scroll(event):
    global width
    if event.delta==120:
        width+=1
    if event.delta==-120:
        width-=1
    label.config(text=str(width))

canvas=tkinter.Canvas(window, relief="solid", bd=2)
canvas.pack(expand=True, fill="both")
canvas.bind("<B1-Motion>", drawing)
canvas.bind("<MouseWheel>", scroll)

label=tkinter.Label(window, text=str(width))
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

width=1

def drawing(event):
    if width>0:
        x1=event.x-1
        y1=event.y-1
        x2=event.x+1
        y2=event.y+1
        canvas.create_oval(x1, y1, x2, y2, fill="blue", width=width)

def scroll(event):
    global width
    if event.delta==120:
        width+=1
    if event.delta==-120:
        width-=1
    label.config(text=str(width))

canvas=tkinter.Canvas(window, relief="solid", bd=2)
canvas.pack(expand=True, fill="both")
canvas.bind("<B1-Motion>", drawing)
canvas.bind("<MouseWheel>", scroll)

label=tkinter.Label(window, text=str(width))
label.pack()

window.mainloop()

{% endhighlight %}

`위젯.bind("이벤트", 함수)`를 사용하여 `위젯`의 `이벤트`가 작동할 때 실행할 `함수`를 설정할 수 있습니다.

`함수`는 `이벤트 객체(event)`를 **첫 번째 매개변수**로 전달받으므로, `drawing(event)`처럼 매개변수를 하나 선언해야 합니다.

`서로 다른 이벤트`를 `bind`하면 `모두 적용`됩니다.

`같은 이벤트`를 다시 `bind`하면 **기존 함수가 새 함수로 대체**되며, `위젯.bind("이벤트", 함수, add="+")`로 설정하면 기존 함수를 유지한 채 **함수를 추가**합니다.

`위젯.unbind("이벤트")`를 사용하여 설정된 `이벤트`를 해제할 수 있습니다.

`event.delta`를 이용하여 `마우스 휠`의 `delta`값을 확인할 수 있습니다.

`event.delta`의 값이 `120`일 경우 `업 스크롤`이며, `-120`일 경우 `다운 스크롤`입니다.

- Tip : `delta`의 값은 `Windows` 기준이며, `macOS`에서는 `1`, `-1`과 같이 작은 값이 전달되고 `Linux`에서는 `<Button-4>`, `<Button-5>` 이벤트가 발생합니다. 운영체제에 관계없이 동작하게 하려면 `event.delta > 0`과 같이 **부호**로 판단합니다.

<br>
<br>

## Event Attribute

|     이름     |                         의미                        |
|:------------:|:---------------------------------------------------:|
|  event.x, event.y  |   이벤트가 발생한 위젯 내부의 마우스 좌표   |
|  event.widget  |   이벤트가 발생한 위젯   |
|  event.num  |   눌린 마우스 버튼 번호 (1, 2, 3)   |
|  event.delta  |   마우스 휠의 이동 값   |
|  event.char  |   입력된 키의 문자   |
|  event.keysym  |   입력된 키의 이름 (`Return`, `Escape`, `a` 등)   |

<br>
<br>

## Bind Event

### Button

|     이름     |             의미             |
|:------------:|:----------------------------:|
|  \<Button-1>  |  마우스 왼쪽 버튼을 누를 때  |
|  \<Button-2>  |   마우스 휠 버튼을 누를 때   |
|  \<Button-3>  | 마우스 오른쪽 버튼을 누를 때 |
|  \<Button-4>  |           스크롤 업 (Linux)          |
|  \<Button-5>  |          스크롤 다운 (Linux)         |
| \<MouseWheel> |        마우스 휠 이동 (Windows, macOS)        |

<br>

### Motion

|     이름    |                   의미                  |
|:-----------:|:---------------------------------------:|
| \<Motion> |  마우스가 움직일 때  |
| \<B1-Motion> |  마우스 왼쪽 버튼을 누르면서 움직일 때  |
| \<B2-Motion> |   마우스 휠 버튼을 누르면서 움직일 때   |
| \<B3-Motion> | 마우스 오른쪽 버튼을 누르면서 움직일 때 |

<br>

### Release

|        이름       |            의미            |
|:-----------------:|:--------------------------:|
| \<ButtonRelease-1> |  마우스 왼쪽 버튼을 뗄 때  |
| \<ButtonRelease-2> |   마우스 휠 버튼을 뗄 때   |
| \<ButtonRelease-3> | 마우스 오른쪽 버튼을 뗄 때 |

<br>

### Double Click

|        이름       |                 의미                |
|:-----------------:|:-----------------------------------:|
| \<Double-Button-1> |  마우스 왼쪽 버튼을 더블 클릭할 때  |
| \<Double-Button-2> |   마우스 휠 버튼을 더블 클릭할 때   |
| \<Double-Button-3> | 마우스 오른쪽 버튼을 더블 클릭할 때 |

<br>

### Widget Operation

|    이름    |                    의미                    |
|:----------:|:------------------------------------------:|
|   \<Enter>  |   위젯 안으로 마우스 포인터가 들어왔을 때  |
|   \<Leave>  |    위젯 밖으로 마우스 포인터가 나갔을 때   |
|  \<FocusIn> |  위젯 안으로 Tab 키를 이용하여 들어왔을 때 |
| \<FocusOut> | 위젯 밖으로 Tab 키를 이용하여 나갔을 때    |
| \<Configure> | 위젯의 모양이 수정되었을 때   |

<br>

### Key Input

|   이름   |             의미             |
|:--------:|:----------------------------:|
|   \<Key>  |    특정 키가 입력되었을 때   |
| \<Return> |   Enter 키가 입력되었을 때   |
|  \<Cancel>  | Break 키가 입력되었을 때 |
|  \<Pause>  | Pause 키가 입력되었을 때 |
|  \<BackSpace>  | 백스페이스 키가 입력되었을 때 |
|  \<Caps_Lock>  | 캡스 락 키가 입력되었을 때 |
|  \<Escape>  | 이스케이프 키가 입력되었을 때 |
|  \<Home>  | Home 키가 입력되었을 때 |
|  \<End>  | End 키가 입력되었을 때 |
|  \<Print> |  Print 키가 입력되었을 때  |
|  \<Insert> |  Insert 키가 입력되었을 때  |
|  \<Delete> |  Delete 키가 입력되었을 때  |
|  \<Prior> |  Page Up 키가 입력되었을 때  |
|  \<Next> |  Page Down 키가 입력되었을 때  |
|  \<Tab> |  Tab 키가 입력되었을 때  |
|  \<space> |  스페이스 바가 입력되었을 때  |
|  \<Up>  | 윗쪽 방향키가 입력되었을 때 |
|  \<Down>  | 아랫쪽 방향키가 입력되었을 때 |
|  \<Right>  | 오른쪽 방향키가 입력되었을 때 |
|  \<Left>  | 왼쪽 방향키가 입력되었을 때 |

- Tip : `<Key>` 이벤트 입력 시, `<s>`, `<0>`, `<F1>`, `<F4>` 등 특정 문자가 입력되었을 때도 가능하며, `공백`은 `< >`가 아닌 `<space>`로 **표기해야 합니다.**

- Tip : `<Key>` 이벤트 입력 시, 키 이벤트가 할당된 컨트롤에 `*.focus_set()`을 추가해 **포커스**를 할당합니다. 키 이벤트는 **포커스를 가진 위젯**에만 전달됩니다.

<br>

### Assistant Key Input

|   이름   |             의미             |
|:--------:|:----------------------------:|
|   \<Shift-Key>  |   Shift + 특정 키가 입력되었을 때   |
| \<Control-Key> |   Ctrl + 특정 키가 입력되었을 때   |
|  \<Alt-Key> |  Alt + 특정 키가 입력되었을 때  |

- Tip : `<Control-s>`, `<Alt-F4>`처럼 `Key` 대신 특정 키를 조합할 수 있습니다.
