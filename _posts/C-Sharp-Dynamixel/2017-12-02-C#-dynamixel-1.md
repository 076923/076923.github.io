---
layout: post
title: "C# Dynamixel 강좌 : 제 1강 - Dynamixel 회로도"
tagline: "C# Dynamixel Circuit"
image: /assets/images/robotis.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Dynamixel']
keywords: C#, Visual Studio, Dynamixel, Dynamixel Circuit
ref: C#-Dynamixel
category: C#
permalink: /posts/C-dynamixel-1/
comments: true
toc: true
---

## Dynamixel

<img data-src="{{ site.images }}/assets/posts/C-Sharp/Dynamixel/lecture-1/1.webp" class="lazyload" width="100%" height="100%"/>

`ROBOTIS`사의 액추에이터 `Dynamxiel`을 `C#`과 `Serial 통신`하여 제어할 수 있습니다.

`RX-64`와 `CH340 USB to RS485 커넥터 어댑터`를 사용합니다.

이 외의 다이나믹셀 `RX-24`, `RX-64` 등과 `RS485 통신 모듈`이라면 동일하게 적용이 가능합니다.

<br>
<br>

## Dynamixel Circuit

<img data-src="{{ site.images }}/assets/posts/C-Sharp/Dynamixel/lecture-1/2.webp" class="lazyload" width="100%" height="100%"/>

`CH340 USB to RS485 커넥터 어댑터`을 컴퓨터 또는 노트북의 `USB Port`에 부착하고 점퍼선을 `녹색 부분`에 연결합니다.

`D+`, `D-`의 위치를 숙지합니다. 해당 위치는 `녹색 부분`에 적혀있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/C-Sharp/Dynamixel/lecture-1/3.webp" class="lazyload" width="100%" height="100%"/>

`Dynamixel`의 `Pinout`을 확인하여 점퍼선을 이용해 다이나믹셀과 연결합니다.

`Dynamixel`은 좌, 우의 `Pinout`이 다르니 `주의`합니다.

`VDD`는 `양극(+)`를 의미하고, `GND`는 `음극(-), 접지`를 의미합니다.

전원은 `Power Supply` 또는 `SMPS`의 `12V`를 사용합니다. 

<br>

<img data-src="{{ site.images }}/assets/posts/C-Sharp/Dynamixel/lecture-1/4.webp" class="lazyload" width="100%" height="100%"/>

위와 같은 회로도가 나오게 되며, `Dynamixel`을 여러개 부착해 **하나의 전원과 하나의 통신 모듈**로 연결된 모든 `Dynamixel`을 제어할 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/C-Sharp/Dynamixel/lecture-1/5.webp" class="lazyload" width="100%" height="100%"/>

`다이나믹셀`끼리의 연결은 위와 같이 연결합니다.
