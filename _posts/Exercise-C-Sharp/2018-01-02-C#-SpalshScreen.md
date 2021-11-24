---
layout: post
title: "C# 예제 : Splash Screen"
tagline: "C# Create Splash Screen"
image: /assets/images/csharp.svg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ["C# : Exercise"]
keywords: C#, Visual Studio, Splash Screen
ref: Exercise-C#
category: Exercise
permalink: /exercise/C-spalshscreen/
comments: true
toc: true
---

## Splash Screen

![0]({{ site.images }}/assets/posts/Exercise/C-Sharp/C-Sharp/spalshscreen/0.gif)

Splash Screen이란 프로그램을 시작했을 때, **로딩 중에 표시되는 이미지를 의미**합니다.

**프로그램에 대한 소개**나 **로딩 진행률** 등을 표시합니다. 또한 이 Splash Screen을 이용하여 `로그인창`, `로딩창`, `시작폼` 등에 응용할 수 있습니다.

<br>
<br>

## 항목 추가

![1]({{ site.images }}/assets/posts/Exercise/C-Sharp/C-Sharp/spalshscreen/1.png)

`프로젝트(P)` → `새 항목 추가(W)`를 눌러 `Splash Screen`이 될 `Form`을 추가합니다.

<br>

![2]({{ site.images }}/assets/posts/Exercise/C-Sharp/C-Sharp/spalshscreen/2.png)

`Windows Form`을 선택 한 후 이름을 `SplashForm.cs`으로 변경하고 추가합니다.

<br>
<br>

## 전체 코드

### Program.cs

![3]({{ site.images }}/assets/posts/Exercise/C-Sharp/C-Sharp/spalshscreen/3.png)

`솔루션 탐색기`에서 `Program.cs`을 더블클릭하여 `프로그램 주 진입점`으로 이동합니다.

<br>

{% highlight C# %}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Project
{
    static class Program
    {
        /// <summary>
        /// 해당 응용 프로그램의 주 진입점입니다.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            SplashForm SplashForm = new SplashForm();
            Application.Run(SplashForm);

            Application.Run(new Form1());
        }
    }
}

{% endhighlight %}

위와 같이 코드를 수정합니다. `SplashForm`이 종료될 경우 `Form1`이 실행되게 됩니다.

<br>

### SplashForm.cs

{% highlight C# %}

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;

namespace Project
{
    public partial class SplashForm : Form
    {
        delegate void ProgressDelegate(int i);
        delegate void CloseDelegate();

        public SplashForm()
        {
            InitializeComponent();
        }

        private void SplashForm_Load(object sender, EventArgs e)
        {
            Thread loading = new Thread(Thread);
            loading.Start();
        }

        private void Step(int i)
        {
            progressBar1.Value = i;
        }

        private void FormClose()
        {
            this.Close();
        }

        private void Thread()
        {
            for(int i=0; i<=100; i++)
            {
                this.Invoke(new ProgressDelegate(Step), i);
                System.Threading.Thread.Sleep(50);
            }
            System.Threading.Thread.Sleep(1000);
            this.Invoke(new CloseDelegate(FormClose));
        }
    }
}

{% endhighlight %}

SplashForm에 `progressBar`를 추가합니다.

`SplashForm`이 로드될 때 `loading`으로 선언된 쓰레드가 실행됩니다.

사용자 정의 함수 `Thread`로 넘어가게 되며, `Step`이라는 사용자 정의 함수가 100회 실행됩니다.

잠시 대기 한 후 `SplashForm`을 종료합니다. `Program.cs`에서 정의 된 순서대로 폼이 순차적으로 실행됩니다.

