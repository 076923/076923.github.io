const glittersElement = document.querySelector(".glitters");

glittersElement.appendChild(Shape.star({className: "star s1", side: 5, strokeWidth: 10, stroke: "#ffa", fill: "#ffa", width: 100, strokeLinejoin: "round"}));

glittersElement.appendChild(Shape.star({className: "star s2", side: 5, strokeWidth: 10, stroke: "#ffc", fill: "#ffc", width: 100, strokeLinejoin: "round"}));

glittersElement.appendChild(Shape.star({className: "star s3", side: 5, strokeWidth: 10, stroke: "#ffd", fill: "#ffd", width: 100, strokeLinejoin: "round"}));

const crewsElement = document.querySelector(".crews");
const crewHTML = `
  <div class="crew">
    <div class="body">
    <div class="head">
      <div class="ear"></div>
      <div class="face-container">
        <div class="eye left"></div>
        <div class="eye right"></div>
        <div class="mouth"></div>
      </div>
    <div class="helmet"></div>
    </div>
      <div class="controller"></div>
      <div class="bag"></div>
      <div class="arm left"></div>
      <div class="arm right"></div>
      <div class="leg left"></div>
      <div class="leg right"></div>
    </div>
  </div>`;
const crews = [];
for (let i = 0; i < 12; ++i) {
  crews[i] = crewHTML;
}
crewsElement.innerHTML = crews.join("");

const crewBodyScene = new Scene({
  ".crew .arm.right": {
    0.1: {
      transform: "translate(0px) rotate(0deg)",
    },
    0.9: {
      transform: "translate(-35px) rotate(180deg)",
    },
    1: {},
  },
  ".crew .arm.left": {
    0.1: {
      transform: "translate(0px) rotate(180deg)",
    },
    0.9: {
      transform: "translate(35px) rotate(0deg)",
    },
    1: {},
  },
  ".crew .leg.right": {
    0.1: {
      transform: "translate(0px) rotate(-80deg)",
    },
    0.9: {
      transform: "translate(-23px) rotate(30deg)",
    },
    1: {},
  },
  ".crew .leg.left": {
    0.1: {
      transform: "translate(0px) rotate(30deg)",
    },
    0.9: {
      transform: "translate(23px) rotate(-80deg)",
    },
    1: {},
  },
}, {
  duration: 1,
  iterationCount: "infinite",
  direction: "alternate",
  selector: true,
  easing: "ease-in-out",
});

const crewKeyframes = {
  0: {
    transform: {
      translate: "-50%, 20px",
      rotate: "-35deg",
      translateY: "-1000px",
      scale: 0.5,
      rotate2: "-8deg",
      translateY2: "0px",
    },
  },
};

for (let i = 1; i <= 24; ++i) {
  crewKeyframes[i] = {
    transform: {
      rotate: `${-35 + i * 2.35}deg`,
    },
  };
}
for (let i = 20; i <= 24; ++i) {
   crewKeyframes[i].transform.translateY2 = `${25 * (20 - i)}px`;
}
for (let i = 22; i <= 24; ++i) {
  crewKeyframes[i].opacity = 1 - (i - 22) / 2;
}

const scene = new Scene({
  ".crew": i => {
    return {
      keyframes: crewKeyframes,
      options: {
        delay: i * 2,
        iterationCount: "infinite",
      },
    }; 
  },
  "crewbody": crewBodyScene,
}, { 
  selector: true,
  easing: "ease-in-out",
  playSpeed: 2,
});

scene.playCSS();