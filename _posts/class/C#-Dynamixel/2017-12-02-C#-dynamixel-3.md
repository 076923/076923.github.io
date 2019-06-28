---
layout: post
title: "C# Dynamixel 강좌 : 제 3강 - Dynamixel Serial 통신"
tagline: "C# Using Dynamixel"
image: /assets/images/robotis.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['C#-Dynamixel']
keywords: C#, Visual Studio, Dynamixel, Using Dynamixel
ref: C#-Dynamixel
category: posts
permalink: /posts/C-dynamixel-3/
comments: true
---

## Dynamixel & C# ##
----------

`Dynamixel`과 `C#`에서 `Serial 통신`하여 액추에이터를 제어할 수 있습니다.

<br>
<br>

## Main ##
----------

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
using System.IO.Ports;

namespace Dynamixel
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        static SerialPort serial;
        
        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                if (serial == null)
                {
                    serial = new SerialPort("COM10", 1000000);
                    serial.Open();
                }
            }
            catch
            {
                MessageBox.Show("Serial Port가 없습니다.");
            }     
        }
            
        private static void Dynamixel(byte ID, int position, byte speed)
        {
            byte length = 7;
            
            byte pos_l = (byte)(position & 0xff);
            byte pos_h = (byte)((position & 0xff00) >> 8);

            byte speed_l = (byte)(speed & 0xff);
            byte speed_h = (byte)((speed & 0xff00) >> 8);

            byte checkSum = (byte)(~((ID + length + 0x03 + 0x1E + pos_h + pos_l + speed_h + speed_l) & 0xff));
            byte[] buffer = { 0xFF, 0xFF, ID, length, 0x03, 0x1E, pos_l, pos_h, speed_l, speed_h, checkSum };
            serial.Write(buffer, 0, buffer.Length);
        }
    }
}

{% endhighlight %}

<br>
<br>

## Main Code ##
----------

{% highlight C# %}

using System.IO.Ports;

{% endhighlight %}

`Serial 통신`을 위하여 `namespace`에 `IO.Ports`를 선언합니다.

<br>
<br>

{% highlight C# %}

static SerialPort serial;

{% endhighlight %}

`Dynamixel`과의 통신을 위하여 `serial`이라는 변수를 선언합니다.

<br>
<br>

{% highlight C# %}

private void Form1_Load(object sender, EventArgs e)
{
    try
    {
        if (serial == null)
        {
            serial = new SerialPort("COM10", 1000000);
            serial.Open();
        }
    }
    catch
    {
        MessageBox.Show("Serial 포트가 없습니다.");
    }     
}

{% endhighlight %}

`Form1` 로드 시 `serial`의 포트를 설정합니다. `Dynamixel Wizard` 또는 `장치관리자`에서 **포트 번호를 확인할 수 있습니다.**

<br>
<br>

{% highlight C# %}

private static void Dynamixel(byte ID, int position, byte speed)
{
    byte length = 7;
    
    byte pos_l = (byte)(position & 0xff);
    byte pos_h = (byte)((position & 0xff00) >> 8);

    byte speed_l = (byte)(speed & 0xff);
    byte speed_h = (byte)((speed & 0xff00) >> 8);

    byte checkSum = (byte)(~((ID + length + 0x03 + 0x1E + pos_h + pos_l + speed_h + speed_l) & 0xff));
    byte[] buffer = { 0xFF, 0xFF, ID, length, 0x03, 0x1E, pos_l, pos_h, speed_l, speed_h, checkSum };
    serial.Write(buffer, 0, buffer.Length);
}

{% endhighlight %}

위의 코드를 활용하여 `Dynamixel`의 `목표 위치`와 `속도`를 제어할 수 있습니다.

<br>
<br>

{% highlight C# %}

Dynamixel(ID, POSITION, SPEED);

{% endhighlight %}

Dynamixel이 작동할 부분에 위의 `사용자 정의 함수`를 이용해 제어할 수 있습니다.

`POSITION`은 `0~2047` 또는 `0~4095` 등의 범위를 가지며, `SPEED`는 `0~2047` 등의 값을 가집니다.

추가적인 제어가 필요하다면 `checkSum`과 `buffer`를 수정하여 제어가 가능합니다.

<br>

* Tip : `Dynamixel`의 종류마다 `POSITION`과 `SPEED` 등의 값이 다릅니다. `ROBOTIS E-Manual`을 통해 값을 확인이 가능합니다.

<br>

`ROBOTIS e-Manual` : [바로가기][e-manual]

[e-manual]: http://support.robotis.com/ko/

