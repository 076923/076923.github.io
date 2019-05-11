---
bg: "robotis.png"
layout: post
comments: true
title:  "C# Dynamixel 강좌 : 제 2강 - Dynamixel 설정"
crawlertitle: "C# Dynamixel 강좌 : 제 2강 - Dynamixel 설정"
summary: "C# Dynamixel Setting"
date: 2017-12-02
categories: posts
tags: ['C#-Dynamixel']
author: 윤대희
star: true
---

### Dynamixel Setting ###
----------
[![1]({{ site.images }}/C/dynamixel/ch2/1.PNG)]({{ site.images }}/C/dynamixel/ch2/1.PNG) 
`Dynamixel`을 설정하기 위해서 로보티즈의 로봇 전용 소프트웨어 `RoboPlus` 프로그램을 설치합니다. 설치 후, `전문가`탭에서 `Dynamixel Wizard`를 실행시킵니다.


`RoboPlus 설치 바로가기` : [ROBOTIS 고객지원 다운로드][roboplus]

<br>
[![2]({{ site.images }}/C/dynamixel/ch2/2.PNG)]({{ site.images }}/C/dynamixel/ch2/2.PNG) 
프로그램 상단에서 `포트 설정` 후 `포트 연결` 버튼을 누릅니다.


<br>
[![3]({{ site.images }}/C/dynamixel/ch2/3.PNG)]({{ site.images }}/C/dynamixel/ch2/3.PNG) 
`DXL 1.0`에 체크합니다. 그 후, `상세 검색`을 선택한 다음 `검색 시작` 버튼을 눌러 `Dynamixel`을 검색합니다.

<br>
[![4]({{ site.images }}/C/dynamixel/ch2/4.PNG)]({{ site.images }}/C/dynamixel/ch2/4.PNG)
검색 된 `Dynamixel`을 눌러 정보 창을 띄웁니다. `ID`, `통신 속도` 등 초기 설정을 할 수 있으며 테스트 또한 가능합니다.

<br>
[![5]({{ site.images }}/C/dynamixel/ch2/5.PNG)]({{ site.images }}/C/dynamixel/ch2/5.PNG)
`ID`를 `1`, `통신 속도`를 `1(1000000bps)`로 적용합니다. 이 외 사용할 성능에 맞는 `최대 토크`, `PID Gain` 등을 설정합니다.


`ID`는 사용할 `Dynamixel`의 고유 식별 번호이며, `통신 속도`는 `C#`과의 `Serial 통신 속도`입니다.

<br>
### Manual ###
----------
`ROBOTIS e-Manual` : [바로가기][e-manual]
<br>
`ROBOTIS 튜토리얼` : [바로가기][tutorial]

[roboplus]: http://www.robotis.com/service/downloadpage.php?cate=software
[e-manual]: http://support.robotis.com/ko/
[tutorial]: http://www.robotis.com/model/board.php?bo_table=tutorial_vod&page=4
