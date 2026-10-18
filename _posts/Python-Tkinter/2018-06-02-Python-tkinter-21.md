---
layout: post
title: "Python tkinter 강좌 : 제 21강 - Font"
tagline: "Python tkinter Font"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Font, tkinter 글꼴, tkinter font family size, tkinter font families, tkinter 글꼴 크기 변경
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-21/
comments: true
toc: true
---

## Font(글꼴)

`Font`를 이용하여 위젯들의 문자열에 `글꼴`을 설정할 수 있습니다.

<br>
<br>

## Font 사용

{% highlight Python %}

import tkinter
import tkinter.font

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

font=tkinter.font.Font(family="맑은 고딕", size=20, slant="italic")

label=tkinter.Label(window, text="파이썬 3.6", font=font)
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

import tkinter.font

{% endhighlight %}

<br>

{% highlight Python %}

font=tkinter.font.Font(family="맑은 고딕", size=20, slant="italic")

label=tkinter.Label(window, text="파이썬 3.6", font=font)
label.pack()

{% endhighlight %}

`tkinter.font.Font(매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 `위젯`에 표시할 `글꼴`을 설정할 수 있습니다.

`매개변수`를 사용하여 `글꼴`을 설정합니다.

글꼴을 적용할 `위젯의 font 매개변수`에 사용합니다.

하나의 `Font` 객체를 여러 위젯에 적용한 뒤, `font.config(size=30)`과 같이 글꼴을 변경하면 **적용된 모든 위젯의 글꼴이 함께 변경**됩니다.

- Tip : `tkinter.font`는 `import tkinter`만으로는 사용할 수 없으며, `import tkinter.font`를 별도로 작성해야 합니다.

- Tip : `font` 매개변수에는 `Font` 객체 대신 `font=("맑은 고딕", 20, "italic")`과 같은 **튜플**이나 `font="맑은 고딕 20 italic"`과 같은 **문자열**도 사용할 수 있습니다. 단, 문자열 형식은 글꼴 이름에 공백이 있으면 `font="{맑은 고딕} 20 italic"`처럼 중괄호로 감싸야 합니다.

- Tip : `size`를 양수로 설정하면 `포인트(pt)` 단위, 음수로 설정하면 `픽셀(px)` 단위로 적용됩니다. `예) size=-20 → 20px`

- Tip : `tkinter.font.families()`로 현재 시스템에서 사용할 수 있는 **글꼴 이름의 목록**을 확인할 수 있으며, 존재하지 않는 글꼴을 설정하면 오류 없이 기본 글꼴로 대체됩니다.

<br>
<br>

## Font Parameter

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| name | 글꼴 객체의 이름 | - | 문자열 |
| family | 텍스트의 글꼴 | [플랫폼 기본 글꼴](#reference-1) | 글꼴 이름 |
| size | 텍스트의 글꼴 크기 | [플랫폼 기본 크기](#reference-1) | 상수 |
| weight | 텍스트의 진하게 | normal | normal, bold |
| slant | 텍스트의 기울임 | roman | roman, italic |
| underline | 텍스트의 밑줄 | False | Boolean |
| overstrike | 텍스트의 취소선 | False | Boolean |

<br>

## Font Method

|      이름      |               의미               |      설명      |
|:--------------:|:--------------------------------:|:----------------:|
| config(매개변수) | 글꼴 설정 변경 | 글꼴의 속성을 변경하며, 적용된 모든 위젯에 반영 |
| actual() | 실제 글꼴 반환 | 실제로 적용된 글꼴의 속성을 `dict` 형식으로 반환 |
| measure("문자열") | 문자열 너비 반환 | 해당 글꼴로 표시했을 때 문자열의 너비(픽셀)를 반환 |

<br>

<a id="reference-1"></a>

### 참고

* `family`와 `size`의 기본값은 운영 체제에 따라 다르며, `Windows`에서는 `굴림`, `12`가 사용됨

* `font.actual()`로 현재 환경에서 실제로 적용된 글꼴과 크기를 확인할 수 있음
