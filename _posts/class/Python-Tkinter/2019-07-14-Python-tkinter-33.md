---
layout: post
title: "Python tkinter 강좌 : 제 33강 - 매개변수 전달"
tagline: "Python tkinter command parameter"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, Python tkinter command, Python tkinter parameter
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-33/
comments: true
---

## 매개변수 전달(Command parameter) ##
----------

`tkinter`의 `command`에 매개변수를 전달할 수 있습니다.

`람다(lambda)` 함수를 사용해 함수에 여러 매개변수를 전달할 수 있습니다.

<br>    

`람다 사용하기` : [22강 바로가기][22강]

<br>
<br>

## 람다 함수 적용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

def command_args(argument1, argument2, argument3):
    global arg1
    arg1 = argument1 * 2
    print(argument1, argument2, argument3)

arg1 = 1
arg2 = "alpha"
arg3 = "beta"

button = tkinter.Button(window, width=25, height=10, text="버튼", command=lambda: command_args(arg1, arg2, arg3))
button.pack(expand=True, anchor="center")

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def command_args(argument1, argument2, argument3):
    global arg1
    arg1 = argument1 * 2
    print(argument1, argument2, argument3)

{% endhighlight %}

<br>

`command_args` 함수를 정의하고, 세 가지의 매개변수를 적용합니다.

`arg1` 변수는 **전역 변수**로 설정합니다.

* Tip : `전역 변수(global)`를 사용하면, 함수 밖에서 선언된 `arg1`을 변경할 수 있습니다.

<br>

{% highlight Python %}

button = tkinter.Button(window, width=25, height=10, text="버튼", command=lambda: command_args(arg1, arg2, arg3))

{% endhighlight %}

`command`에 `람다 함수(lambda)`를 적용합니다.

`command=lambda: 함수(매개변수1, 매개변수2, 매개변수3, ...)`으로 설정합니다.

람다 함수를 사용하면 `n` 개 이상의 매개변수를 전달할 수 있습니다.

<br>
<br>

## 클래스에서 함수 적용 ##
----------

{% highlight Python %}

import tkinter

class windows_tkinter:
    def __init__(self, window):
        self.window = window
        self.window.title("YUN DAE HEE")
        self.window.geometry("640x400+100+100")
        self.window.resizable(True, True)

        self.arg1 = 1
        self.arg2 = "alpha"
        self.arg3 = "beta"
        self.__main__()

    def command_args(self, argument1, argument2, argument3):
        print(argument1, argument2, argument3)
        self.arg1 = argument1 * 2

    def __main__(self):
        button = tkinter.Button(self.window, width=25, height=10, text="버튼", command=lambda: self.command_args(self.arg1, self.arg2, self.arg3))
        button.pack(expand=True, anchor="center")

if __name__ == '__main__':    
    window = tkinter.Tk()
    windows_tkinter(window)
    window.mainloop()

{% endhighlight %}

<br>

`command_args` 함수를 정의하고, 세 가지의 매개변수를 적용합니다.

클래스에서 함수 사용법과 동일하게 `self`는 생략해서 전달합니다.

* Tip : 클래스로 구현하면 `전역 변수(global)`를 사용하지 않아도 `arg1`을 변경할 수 있습니다.

<br>
<br>

## Result ##
----------

1 alpha beta<br>
2 alpha beta<br>
4 alpha beta<br>
8 alpha beta<br>
16 alpha beta<br>
32 alpha beta<br>
64 alpha beta<br>
128 alpha beta<br>
...

[22강]: https://076923.github.io/posts/Python-21/