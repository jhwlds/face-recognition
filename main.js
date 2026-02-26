// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");

const hudSmileEl = document.getElementById("hud-smile");
const hudLookAwayEl = document.getElementById("hud-lookaway");
const hudLookAwayFlagEl = document.getElementById("hud-lookaway-flag");
const appleDialogEl = document.getElementById("apple-dialog");

let faceLandmarker;
let runningMode = "IMAGE"; // "IMAGE" | "VIDEO"
let enableWebcamButton;
let webcamRunning = false;

const videoWidth = 480;

// Cache landmark index sets for look-away (perf).
const LEFT_EYE_IDX = connectionIndices(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
const RIGHT_EYE_IDX = connectionIndices(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
const LEFT_IRIS_IDX = connectionIndices(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS);
const RIGHT_IRIS_IDX = connectionIndices(
  FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS
);

async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1,
  });

  demosSection.classList.remove("invisible");
}

createFaceLandmarker();

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detect them
// upon click.
********************************************************************/

const imageContainers = document.getElementsByClassName("detectOnClick");
for (const imageContainer of imageContainers) {
  imageContainer.children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
  if (!faceLandmarker) {
    console.log("Wait for faceLandmarker to load before clicking!");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await faceLandmarker.setOptions({ runningMode });
  }

  // Remove all landmarks drawn before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
  for (let i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode.removeChild(n);
  }

  const faceLandmarkerResult = faceLandmarker.detect(event.target);

  const canvas = document.createElement("canvas");
  canvas.setAttribute("class", "canvas");
  canvas.setAttribute("width", `${event.target.naturalWidth}px`);
  canvas.setAttribute("height", `${event.target.naturalHeight}px`);
  canvas.style.left = "0px";
  canvas.style.top = "0px";
  canvas.style.width = `${event.target.width}px`;
  canvas.style.height = `${event.target.height}px`;

  event.target.parentNode.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  const drawingUtils = new DrawingUtils(ctx);

  for (const landmarks of faceLandmarkerResult.faceLandmarks) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#C0C0C070", lineWidth: 1 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      { color: "#FF3030" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      { color: "#FF3030" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      { color: "#30FF30" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      { color: "#30FF30" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      { color: "#E0E0E0" }
    );
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: "#E0E0E0",
    });
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      { color: "#FF3030" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
      { color: "#30FF30" }
    );
  }

  drawBlendShapes(imageBlendShapes, faceLandmarkerResult.faceBlendshapes);
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

function enableCam() {
  if (!faceLandmarker) {
    console.log("Wait! faceLandmarker not loaded yet.");
    return;
  }

  const labelSpan = enableWebcamButton.querySelector(".mdc-button__label");
  if (webcamRunning === true) {
    webcamRunning = false;
    if (labelSpan) labelSpan.innerText = "ENABLE WEBCAM";
    else enableWebcamButton.innerText = "ENABLE WEBCAM";
  } else {
    webcamRunning = true;
    if (labelSpan) labelSpan.innerText = "DISABLE WEBCAM";
    else enableWebcamButton.innerText = "DISABLE WEBCAM";
  }

  const constraints = { video: true };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// ------------------------------------------------------------------
// Scores + stability state
// ------------------------------------------------------------------

const SCORE_TAU_MS = 120;
// More sensitive look-away tuning.
const LOOK_EYE_T = 0.2;
const LOOK_EYE_MAX = 0.55;
const LOOK_HEAD_T = 0.1;
const LOOK_HEAD_MAX = 0.35;
const LOOK_ON = 0.42;
const LOOK_OFF = 0.28;

const DIALOG_HOLD_MS = 350;
const DIALOG_MIN_SHOW_MS = 900;

const state = {
  tPrev: performance.now(),
  // smoothed scores
  smile: 0,
  lookAwayScore: 0,
  lookingAway: false,
  // dialog stability
  dialogCurrent: "Apologize properly.",
  dialogSinceMs: performance.now(),
  dialogCandidate: null,
  dialogCandidateSinceMs: 0,
  // DOM caching
  hudLastText: {
    smile: "",
    look: "",
    lookFlag: "",
    dialog: "",
  },
  // blendshape list throttle
  lastBlendshapeDomMs: 0,
};

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function emaAlpha(dtMs, tauMs) {
  const dt = Math.max(0, dtMs);
  const tau = Math.max(1, tauMs);
  return 1 - Math.exp(-dt / tau);
}

function blendshapesToMap(categories) {
  const out = Object.create(null);
  if (!categories || !categories.length) return out;
  for (const shape of categories) {
    const key = shape.categoryName;
    if (!key) continue;
    out[key] = typeof shape.score === "number" ? shape.score : 0;
  }
  return out;
}

function getCat(cat, key) {
  const v = cat && typeof cat[key] === "number" ? cat[key] : 0;
  return v;
}

function computeSmile(cat) {
  const l = getCat(cat, "mouthSmileLeft");
  const r = getCat(cat, "mouthSmileRight");
  return clamp01((l + r) * 0.5);
}

function connectionIndices(connections) {
  const set = new Set();
  if (!connections) return set;
  for (const c of connections) {
    if (!c) continue;
    if (Array.isArray(c) && c.length >= 2) {
      set.add(c[0]);
      set.add(c[1]);
    } else if (typeof c.start === "number" && typeof c.end === "number") {
      set.add(c.start);
      set.add(c.end);
    }
  }
  return set;
}

function bboxFromIndices(landmarks, indicesSet) {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  let count = 0;
  for (const idx of indicesSet) {
    const p = landmarks[idx];
    if (!p) continue;
    const x = p.x;
    const y = p.y;
    if (typeof x !== "number" || typeof y !== "number") continue;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    count++;
  }
  if (count === 0) return null;
  return { minX, minY, maxX, maxY };
}

function centerFromIndices(landmarks, indicesSet) {
  let sx = 0,
    sy = 0,
    n = 0;
  for (const idx of indicesSet) {
    const p = landmarks[idx];
    if (!p) continue;
    if (typeof p.x !== "number" || typeof p.y !== "number") continue;
    sx += p.x;
    sy += p.y;
    n++;
  }
  if (!n) return null;
  return { x: sx / n, y: sy / n };
}

function lookAwayMagForEye(landmarks, eyeConnections, irisConnections) {
  const eyeBox = bboxFromIndices(landmarks, eyeConnections);
  const irisCenter = centerFromIndices(landmarks, irisConnections);
  if (!eyeBox || !irisCenter) return 0;

  const midX = (eyeBox.minX + eyeBox.maxX) * 0.5;
  const midY = (eyeBox.minY + eyeBox.maxY) * 0.5;
  const halfW = Math.max(1e-6, (eyeBox.maxX - eyeBox.minX) * 0.5);
  const halfH = Math.max(1e-6, (eyeBox.maxY - eyeBox.minY) * 0.5);

  const dx = (irisCenter.x - midX) / halfW;
  const dy = (irisCenter.y - midY) / halfH;

  const mag = Math.sqrt(dx * dx + dy * dy);
  if (!Number.isFinite(mag)) return 0;
  return mag;
}

function safePoint(landmarks, idx) {
  const p = landmarks && landmarks[idx];
  if (!p) return null;
  if (typeof p.x !== "number" || typeof p.y !== "number") return null;
  return p;
}

function hypot2(a, b) {
  return Math.sqrt(a * a + b * b);
}

function headYawProxy(landmarks) {
  // Common FaceMesh indices: nose tip(1), left eye outer(33), right eye outer(263)
  const NOSE_TIP = 1;
  const LEFT_EYE_OUTER = 33;
  const RIGHT_EYE_OUTER = 263;

  const n = safePoint(landmarks, NOSE_TIP);
  const l = safePoint(landmarks, LEFT_EYE_OUTER);
  const r = safePoint(landmarks, RIGHT_EYE_OUTER);
  if (!n || !l || !r) return 0;

  // 2D asymmetry proxy (works even if z is missing/unreliable).
  const dL = hypot2(n.x - l.x, n.y - l.y);
  const dR = hypot2(n.x - r.x, n.y - r.y);
  const yaw2d = Math.abs(dL - dR) / (dL + dR + 1e-6);

  // If z exists, add a depth-based proxy too (often reacts strongly to yaw).
  let yawZ = 0;
  if (typeof l.z === "number" && typeof r.z === "number") {
    const eyeDx = Math.abs(l.x - r.x) + 1e-6;
    yawZ = Math.abs(l.z - r.z) / eyeDx;
  }

  // Take the stronger signal; scale 2D a bit to be comparable.
  return Math.max(yawZ, yaw2d * 1.25);
}

function computeLookAwayCombined(landmarks) {
  if (!landmarks || !landmarks.length) {
    return { lookAwayScore: 0, eyeScore: 0, headScore: 0 };
  }

  const leftMag = lookAwayMagForEye(landmarks, LEFT_EYE_IDX, LEFT_IRIS_IDX);
  const rightMag = lookAwayMagForEye(landmarks, RIGHT_EYE_IDX, RIGHT_IRIS_IDX);
  const eyeRaw = Math.max(leftMag, rightMag);
  const eyeScore = clamp01((eyeRaw - LOOK_EYE_T) / (LOOK_EYE_MAX - LOOK_EYE_T));

  const headRaw = headYawProxy(landmarks);
  const headScore = clamp01(
    (headRaw - LOOK_HEAD_T) / (LOOK_HEAD_MAX - LOOK_HEAD_T)
  );

  const lookAwayScore = Math.max(eyeScore, headScore);
  return { lookAwayScore, eyeScore, headScore };
}

function updateLookingAwayBool(prev, lookAwayScore) {
  if (prev) return lookAwayScore > LOOK_OFF;
  return lookAwayScore > LOOK_ON;
}

function pickDialogLine(facePresent, lookingAway, smile) {
  if (!facePresent) return "I can't see your face…";
  if (lookingAway) return "Why are you looking away? Are you ignoring me?";
  if (smile > 0.35) return "You're smiling? I'm angry right now.";
  return "Apologize properly.";
}

function updateDialogStably(nowMs, nextLine) {
  if (state.dialogCurrent === nextLine) {
    state.dialogCandidate = null;
    return state.dialogCurrent;
  }

  if (state.dialogCandidate !== nextLine) {
    state.dialogCandidate = nextLine;
    state.dialogCandidateSinceMs = nowMs;
    return state.dialogCurrent;
  }

  const candidateHeldMs = nowMs - state.dialogCandidateSinceMs;
  const currentShownMs = nowMs - state.dialogSinceMs;

  if (candidateHeldMs >= DIALOG_HOLD_MS || currentShownMs >= DIALOG_MIN_SHOW_MS) {
    state.dialogCurrent = nextLine;
    state.dialogSinceMs = nowMs;
    state.dialogCandidate = null;
  }

  return state.dialogCurrent;
}

function setTextIfChanged(el, text) {
  if (!el) return;
  if (el.textContent !== text) el.textContent = text;
}

function updateHUD(smile, lookAwayScore, lookingAway) {
  const smileText = smile.toFixed(2);
  const lookText = lookAwayScore.toFixed(2);
  const flagText = lookingAway ? "YES" : "NO";

  if (state.hudLastText.smile !== smileText) {
    state.hudLastText.smile = smileText;
    setTextIfChanged(hudSmileEl, smileText);
  }
  if (state.hudLastText.look !== lookText) {
    state.hudLastText.look = lookText;
    setTextIfChanged(hudLookAwayEl, lookText);
  }
  if (state.hudLastText.lookFlag !== flagText) {
    state.hudLastText.lookFlag = flagText;
    setTextIfChanged(hudLookAwayFlagEl, flagText);
  }
}

let lastVideoTime = -1;
let results = undefined;

async function predictWebcam() {
  const radio = video.videoHeight / video.videoWidth;
  video.style.width = `${videoWidth}px`;
  video.style.height = `${videoWidth * radio}px`;
  canvasElement.style.width = `${videoWidth}px`;
  canvasElement.style.height = `${videoWidth * radio}px`;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceLandmarker.setOptions({ runningMode });
  }

  const startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = faceLandmarker.detectForVideo(video, startTimeMs);
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  const nowMs = performance.now();
  const dtMs = nowMs - state.tPrev;
  state.tPrev = nowMs;

  let facePresent = false;
  let rawSmile = 0;
  let rawLookAwayScore = 0;

  if (results && results.faceLandmarks && results.faceLandmarks.length) {
    facePresent = true;

    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: "#30FF30" }
      );

      const { lookAwayScore } = computeLookAwayCombined(landmarks);
      rawLookAwayScore = Math.max(rawLookAwayScore, lookAwayScore);
    }
  }

  if (results && results.faceBlendshapes && results.faceBlendshapes.length) {
    const categories = results.faceBlendshapes[0]?.categories;
    const cat = blendshapesToMap(categories);
    rawSmile = computeSmile(cat);
  }

  if (!facePresent) {
    rawSmile = 0;
    rawLookAwayScore = 0;
  }

  const a = emaAlpha(dtMs, SCORE_TAU_MS);
  state.smile = state.smile + a * (rawSmile - state.smile);
  state.lookAwayScore =
    state.lookAwayScore + a * (rawLookAwayScore - state.lookAwayScore);

  state.smile = clamp01(state.smile);
  state.lookAwayScore = clamp01(state.lookAwayScore);

  state.lookingAway = updateLookingAwayBool(state.lookingAway, state.lookAwayScore);

  updateHUD(state.smile, state.lookAwayScore, state.lookingAway);

  const nextLine = pickDialogLine(
    facePresent,
    state.lookingAway,
    state.smile
  );
  const stableLine = updateDialogStably(nowMs, nextLine);
  if (state.hudLastText.dialog !== stableLine) {
    state.hudLastText.dialog = stableLine;
    setTextIfChanged(appleDialogEl, stableLine);
  }

  // Keep existing blendshape list output "as-is", but throttle DOM rebuild.
  if (results && results.faceBlendshapes) {
    if (nowMs - state.lastBlendshapeDomMs > 90) {
      state.lastBlendshapeDomMs = nowMs;
      drawBlendShapes(videoBlendShapes, results.faceBlendshapes);
    }
  }

  canvasCtx.restore();

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function drawBlendShapes(el, blendShapes) {
  if (!el) return;
  if (!blendShapes || !blendShapes.length) {
    return;
  }

  let htmlMaker = "";
  blendShapes[0].categories.map((shape) => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${
          shape.displayName || shape.categoryName
        }</span>
        <span class="blend-shapes-value" style="width: calc(${
          +shape.score * 100
        }% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
  });

  el.innerHTML = htmlMaker;
}

