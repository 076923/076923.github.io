---
layout: post
title: "Python Pytorch 강좌 : 제 5강 - 최적화(Optimization)"
tagline: "Python PyTorch Optimization"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Optimization
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-5/
comments: true
toc: true
plotly: true
---

## 최적화(Optimization)

`최적화(Optimization)`란 `목적 함수(Objective Function)`의 결괏값을 최적화 시키는 변수를 찾는 알고리즘을 의미합니다.

앞선 [제 4강 - 손실 함수(Loss Function)][4강]에서 `인공 신경망(Neural Network)`은 오찻값을 최소화 시켜 정확도를 높이는 방법으로 학습이 진행되는 것을 확인했습니다.

즉, `손실 함수(Loss Function)`의 값이 최소가 되는 변수를 찾는다면 새로운 데이터에 대해 보다 정교한 `예측(predict)`을 할 수 있게 합니다.

그러므로 실젯값과 예측값의 차이를 계산하여 오차를 최소로 줄일 수 있는 **가중치(Weight)**와 **편향(Bias)**을 계산하게 됩니다.

<br>
<br>

## 경사 하강법(Gradient Descent)

$$ H(x) = 0.879x - 0.436 $$

<br>

<table>
  <thead>
    <tr>
      <th style="text-align: center">X</th>
      <th style="text-align: center">Y(실젯값)</th>
      <th style="text-align: center">예측값</th>
      <th style="text-align: center">오차(손실)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">1</td>
      <td style="text-align: center">0.94</td>
      <td style="text-align: center">0.443</td>
      <td style="text-align: center">0.497</td>
    </tr>
    <tr>
      <td style="text-align: center">2</td>
      <td style="text-align: center">1.98</td>
      <td style="text-align: center">1.322</td>
      <td style="text-align: center">0.658</td>
    </tr>
    <tr>
      <td style="text-align: center">3</td>
      <td style="text-align: center">2.88</td>
      <td style="text-align: center">2.201</td>
      <td style="text-align: center">0.679</td>
    </tr>
    <tr>
      <td style="text-align: center">4</td>
      <td style="text-align: center">3.92</td>
      <td style="text-align: center">0.84</td>
      <td style="text-align: center">0.84</td>
    </tr>
    <tr>
      <td style="text-align: center">5</td>
      <td style="text-align: center">3.96</td>
      <td style="text-align: center">3.959</td>
      <td style="text-align: center">0.001</td>
    </tr>
    <tr>
      <td style="text-align: center">6</td>
      <td style="text-align: center">4.55</td>
      <td style="text-align: center">4.838</td>
      <td style="text-align: center">-0.288</td>
    </tr>
    <tr>
      <td style="text-align: center">7</td>
      <td style="text-align: center">5.64</td>
      <td style="text-align: center">5.717</td>
      <td style="text-align: center">-0.077</td>
    </tr>
    <tr>
      <td style="text-align: center">8</td>
      <td style="text-align: center">6.3</td>
      <td style="text-align: center">6.596</td>
      <td style="text-align: center">-0.296</td>
    </tr>
    <tr>
      <td style="text-align: center">9</td>
      <td style="text-align: center">7.44</td>
      <td style="text-align: center">7.475</td>
      <td style="text-align: center">-0.035</td>
    </tr>
    <tr>
      <td style="text-align: center">10</td>
      <td style="text-align: center">9.1</td>
      <td style="text-align: center">8.354</td>
      <td style="text-align: center">0.746</td>
    </tr>
    <tr>
      <td style="text-align: center">…</td>
      <td style="text-align: center">…</td>
      <td style="text-align: center">…</td>
      <td style="text-align: center">…</td>
    </tr>
  </tbody>
</table>

위 데이터는 이미 최적의 **가중치(Weight)**와 **편향(Bias)**을 계산한 결과입니다.

그러므로 `비용 함수(Cost Function)`를 활용하여 해당 가중치와 편향에 대한 `비용(Cost)`을 계산했을 때 **다른 가중치(Weight)와 편향(Bias) 값보다 가장 작은 값이 나오게 됩니다.**

즉, 가중치(Weight)와 편향(Bias)의 값이 `0.879`과 `-0.436`에서 멀어질수록 `비용(Cost)`이 커지게됩니다.

만약 몇개의 `가중치(Weight)`에 따른 `비용(Cost)`의 값을 2차원 그래프로 표현한다면 다음과 같이 그려질 수 있습니다.

<center>
<div id="gradientPlot1" style="width:100%; max-width:600px"></div>
<script>
var weight=[-.938,-.5685,.878,1.28,1.649,2.019,2.758],cost=[7437.748047,4691.095215,1.37318,412.835571,1448.183594,3113.864502,8336.227539],g_data=[{x:weight,y:cost,mode:"lines",line:{color:"rgb(210, 210, 210)"}}],g_layout={showlegend:!1,xaxis:{title:"Weight"},yaxis:{title:"Cost"}};Plotly.newPlot("gradientPlot1",g_data,g_layout);
</script>
</center>

그래프에서 확인할 수 있듯이 **최적의 가중치(Weight)에서는 비용(Cost)이 가장 작아지며, 최적의 가중치에서 멀어질수록 비용(Cost)이 커지는 것을 확인할 수 있습니다.**

이를 수학적으로 바라본다면, 가중치와 비용의 그래프에서 `기울기(Gradient)`가 0에 가까워질 때 최적의 가중치(weight)를 갖게 되는것을 알 수 있습니다.

위 그래프를 $$ x^2 $$로 근사한 다음, 특정 지점에서 어떠한 기울기를 갖는지 확인해보도록 하겠습니다.

<center>
<div id="gradientPlot2" style="width:100%; max-width:600px"></div>
<script>
for(var x1Values=[],y1Values=[],x2Values=[-8,-6,-4],y2Values=[60,36,12],x3Values=[-5,0,5],y3Values=[0,0,0],x4Values=[8,6,4],y4Values=[60,36,12],x=-10;x<=10;x+=.1)x1Values.push(x),y1Values.push(eval("x*x"));var datad=[{x:x1Values,y:y1Values,mode:"lines",line:{color:"rgb(210, 210, 210)"},name:"함수: x²"},{x:x2Values,y:y2Values,mode:"lines",line:{color:"rgb(255, 55, 55)"},name:"기울기1"},{x:x3Values,y:y3Values,mode:"lines",line:{color:"rgb(164, 194, 244)"},name:"기울기2"},{x:x4Values,y:y4Values,mode:"lines",line:{color:"rgb(83, 244, 32)"},name:"기울기3"}],layoutd={showlegend:!0,legend:{x:1,xanchor:"right",y:1},xaxis:{title:"Weight"},yaxis:{title:"Cost"}};Plotly.newPlot("gradientPlot2",datad,layoutd);
</script>
</center>

위 그래프 처럼 `기울기2`의 형태를 갖는 지점을 찾기 위해서는 초깃값($$ W_{0} $$)을 설정하고 적합한 가중치를 찾을 수 있을 때까지 반복하게 됩니다.

다음과 같은 공식을 통해, 어느 지점에서 시작하더라도 `극값(Extremum Value)`을 찾을 수 있게 연산이 진행됩니다.

<br>
<br>

## 학습률(Learning Rate)

$$
\begin{multline}
\shoveleft W_{0} = \text{Initial Value}\\
\shoveleft W_{i+1} = W_{i} - \alpha \nabla f(W_{i})
\end{multline}
$$

새로운 `가중치(Weight)`를 구하기 위해서 $$ \alpha $$를 적용하는데 이 값은 `간격(Step Size)`으로 `학습률(Learning Rate)`을 의미합니다.

초깃값($$ W_{0} $$)을 임의의 값으로 설정해주듯이 이 학습률도 임의의 값으로 설정하게 됩니다.

다음 그래프들은 학습률에 따른 가중치 최적값을 찾아가는 방식을 보여줍니다.

<br>

### 학습률이 적절할 때

<center>
<div id="stepPlot1" style="width:100%; max-width:600px"></div>
<script>
var n1=0,x1=[-6],y1=[36],z1=[-1],t1=[-6],dt1=.03;let fps1,eachNthFrame1,frameCount1;fps1=3,eachNthFrame1=Math.round(1e3/fps1/16.66),frameCount1=eachNthFrame1;for(var x1Values1=[],y1Values1=[],x=-10;x<=10;x+=.1)x1Values1.push(x),y1Values1.push(eval("x*x"));function compute1(e){for(e=0;e<n1;e++)x1[e]=x1[e]-dt1*(2*x1[e]),y1[e]=x1[e]**2}function update1(){n1%75==0&&(x1=[-6],y1=[36]),frameCount1===eachNthFrame1&&(frameCount1=0,compute1(),Plotly.animate("stepPlot1",{data:[{x:x1,y:y1}]},{transition:{duration:0},frame:{duration:0,redraw:!1}}),n1++),frameCount1++,requestAnimationFrame(update1)}Plotly.newPlot("stepPlot1",[{x:x1,y:z1,mode:"markers",marker:{color:"rgb(255, 34, 22)"}},{x:x1Values1,y:y1Values1,mode:"lines",line:{color:"rgb(210, 210, 210)"}}],{xaxis:{range:[-10,10],title:"Weight"},yaxis:{range:[0,60],title:"Cost"},showlegend:!1}),requestAnimationFrame(update1);
</script>
</center>

### 학습률이 낮을 때

<center>
<div id="stepPlot2" style="width:100%; max-width:600px"></div>
<script>
var n2=0,x2=[-6],y2=[36],z2=[-1],t2=[-6],dt2=.003;let fps2,eachNthFrame2,frameCount2;fps2=3,eachNthFrame2=Math.round(1e3/fps2/16.66),frameCount2=eachNthFrame2;for(var x1Values2=[],y1Values2=[],x=-10;x<=10;x+=.1)x1Values2.push(x),y1Values2.push(eval("x*x"));function compute2(e){for(e=0;e<n2;e++)x2[e]=x2[e]-dt2*(2*x2[e]),y2[e]=x2[e]**2}function update2(){n2%75==0&&(x2=[-6],y2=[36]),frameCount2===eachNthFrame2&&(frameCount2=0,compute2(),Plotly.animate("stepPlot2",{data:[{x:x2,y:y2}]},{transition:{duration:0},frame:{duration:0,redraw:!1}}),n2++),frameCount2++,requestAnimationFrame(update2)}Plotly.newPlot("stepPlot2",[{x:x2,y:z2,mode:"markers",marker:{color:"rgb(255, 34, 22)"}},{x:x1Values2,y:y1Values2,mode:"lines",line:{color:"rgb(210, 210, 210)"}}],{xaxis:{range:[-10,10],title:"Weight"},yaxis:{range:[0,60],title:"Cost"},showlegend:!1}),requestAnimationFrame(update2);
</script>
</center>

### 학습률이 높을 때

<center>
<div id="stepPlot3" style="width:100%; max-width:600px"></div>
<script>
var n3=0,x3=[-6],y3=[36],z3=[-1],t3=[-6],dt3=.02;let fps3,eachNthFrame3,frameCount3;fps3=3,eachNthFrame3=Math.round(1e3/fps3/16.66),frameCount3=eachNthFrame3;for(var x1Values3=[],y1Values3=[],x=-10;x<=10;x+=.1)x1Values3.push(x),y1Values3.push(eval("x*x"));function compute3(e){for(e=0;e<n3;e++)x3[e]=x3[e]-dt3*(2*x3[e]),y3[e]=x3[e]**2,z3[e]=-1*z3[e],t3[e]=x3[e]*z3[e]}function update3(){n3%75==0&&(x3=[-6],y3=[36],z3=[-1],t3=[-6]),frameCount3===eachNthFrame3&&(frameCount3=0,compute3(),Plotly.animate("stepPlot3",{data:[{x:t3,y:y3}]},{transition:{duration:0},frame:{duration:0,redraw:!1}}),n3++),frameCount3++,requestAnimationFrame(update3)}Plotly.newPlot("stepPlot3",[{x:x3,y:z3,mode:"markers",marker:{color:"rgb(255, 34, 22)"}},{x:x1Values3,y:y1Values3,mode:"lines",line:{color:"rgb(210, 210, 210)"}}],{xaxis:{range:[-10,10],title:"Weight"},yaxis:{range:[0,60],title:"Cost"},showlegend:!1}),requestAnimationFrame(update3);
</script>
</center>

### 학습률이 너무 높을 때

<center>
<div id="stepPlot4" style="width:100%; max-width:600px"></div>
<script>
var n4=0,x4=[-6],y4=[36],z4=[-1],t4=[-6],dt4=.01;let fps4,eachNthFrame4,frameCount4;fps4=3,eachNthFrame4=Math.round(1e3/fps4/16.66),frameCount4=eachNthFrame4;for(var x1Values4=[],y1Values4=[],x=-10;x<=10;x+=.1)x1Values4.push(x),y1Values4.push(eval("x*x"));function compute4(e){for(e=0;e<n4;e++)x4[e]=x4[e]+dt4*(2*x4[e]),y4[e]=x4[e]**2,z4[e]=-1*z4[e],t4[e]=x4[e]*z4[e]}function update4(){n4%20==0&&(x4=[-6],y4=[36],z4=[-1],t4=[-6]),frameCount4===eachNthFrame4&&(frameCount4=0,compute4(),Plotly.animate("stepPlot4",{data:[{x:t4,y:y4}]},{transition:{duration:0},frame:{duration:0,redraw:!1}}),n4++),frameCount4++,requestAnimationFrame(update4)}Plotly.newPlot("stepPlot4",[{x:x4,y:z4,mode:"markers",marker:{color:"rgb(255, 34, 22)"}},{x:x1Values4,y:y1Values4,mode:"lines",line:{color:"rgb(210, 210, 210)"}}],{xaxis:{range:[-10,10],title:"Weight"},yaxis:{range:[0,60],title:"Cost"},showlegend:!1}),requestAnimationFrame(update4);
</script>
</center>

<br>
<br>

## 최적화의 문제

위 그래프에서 확인할 수 있듯이 초깃값이나 학습률을 너무 낮거나 높게 잡는다면 **최적의 가중치(Weight)를 찾는 데 오랜 시간이 걸리거나, 발산하여 값을 찾지 못할 수 있습니다.**

하지만, 위 그래프에서는 학습률을 가장 낮게 잡고 많은 연산을 하면 시간은 오래 걸리겠지만, 최적의 가중치를 찾을 수 있는 것처럼 보입니다.

만약, `가중치(Weight)`와 `비용(Cost)`이 다음과 같은 그래프의 형태를 지니게 된다면 다음과 같은 이유로 최적화된 값을 찾을 수 없게됩니다.

<center>
<div id="minimumPlot" style="width:100%; max-width:600px"></div>
<script>
var cx=[];var cy=[];for(var x=-1;x<=2.5;x+=0.01){cx.push(x);cy.push(eval("x**5-2*x**4-3*x**3+6*x**2+x"))}
var c_data=[{x:cx,y:cy,mode:"lines",name:"Weight-Cost"},{x:[-0.079],y:[-0.04],mode:"markers",marker:{color:'rgb(255, 44, 17)',size:10},name:"Global Minimum"},{x:[1.836],y:[1.631],mode:"markers",marker:{color:'rgb(17, 255, 44)',size:10},name:"Local Minimum"}];var c_layout={xaxis:{title:"Weight",autorange:!1,showgrid:!0,zeroline:!1,showline:!0,autotick:!0,ticks:'',showticklabels:!1,range:[-1.5,3,0.01]},yaxis:{title:"Cost",autorange:!1,showgrid:!0,zeroline:!1,showline:!0,autotick:!0,ticks:'',showticklabels:!1,range:[-0.3,5,0.01]}};Plotly.newPlot("minimumPlot",c_data,c_layout)
</script>
</center>

기울기가 0이 되는 지점인 `극값(Extremum Value)`은 `최댓값`(Global Maximum), `최솟값(Global Minimum)`, `극댓값(Local Maximum)`, `극솟값(Local Minimum)`으로 구분할 수 있습니다.

초기 가중치나 학습률을 설정할 때, 시작점이 적절하지 않거나 학습률이 너무 낮다면 `최솟값(Global Minimum)`이 아닌, `극솟값(Local Minimum)`에서 가중치가 결정될 수 있습니다.

또한 `안장점(Saddle Point)`이 존재하는 함수에서도 적절한 가중치를 찾을 수 없게됩니다.

안장점은 다음 그래프 처럼 말의 안장 모양처럼 생긴 그래프를 의미하며, 특정 방향에서 바라볼 경우 극댓값이 되지만 다른 방향에서 보면 극솟값이 되는 지점을 의미합니다.

<center>
<div id="saddlePlot" style="width:100%; max-width:600px"></div>
<script>
function getrandom(num){var value=[];for(i=0;i<=num;i++){var rand=Math.random()*2-1;value.push(rand)}
return value}
var x=getrandom(1000);var y=getrandom(1000);var z=[];for(var i=0;i<=1000;i++){z.push(x[i]**2-y[i]**2)}
var trace={opacity:0.5,color:'rgba(80,127,255,0.7)',type:'mesh3d',x:x,y:y,z:z,scene:"scene"};var point={x:[0],y:[0],z:[0],marker:{color:'rgb(255, 34, 22)'},mode:'markers',type:'scatter',name:'Saddle Point',};var s_layout={showlegend:!0,legend:{x:1,xanchor:'right',y:1}};Plotly.newPlot('saddlePlot',[trace,point],s_layout)
</script>
</center>

`최적화(Optimization)` 알고리즘은 `경사 하강법(Gradient Descent)`처럼 `목적 함수(Objective Function)`가 최적의 값을 찾아갈 수 있도록 최적화되게끔 하는 알고리즘입니다.

어떤 최적화 알고리즘을 사용하느냐에 따라 모델의 정확도가 달라지게 됩니다.

최적화 알고리즘은 경사 하강법 이외에도 `모멘텀(Momentum)`, `Adagrad(Adaptive Gradient)`, `Adam(Adaptive Moment Estimation)` 등이 있습니다.



[4강]: https://076923.github.io/posts/Python-pytorch-4/
