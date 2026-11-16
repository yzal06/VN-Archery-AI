import { ArcheryVisionAnalyzer, SKELETON_EDGES } from "./archery_analyzer.js";

const MODEL_PATH = "./models/yolo11n-pose.onnx";
const INPUT_SIZE = 320;
const PERSON_CONF_THRESHOLD = 0.45;

const statusEl = document.getElementById("status");
const toggleCamBtn = document.getElementById("toggleCamBtn");
const facingSelect = document.getElementById("facingSelect");
const bowSideSelect = document.getElementById("bowSideSelect");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const mPhase = document.getElementById("mPhase");
const mBow = document.getElementById("mBow");
const mDraw = document.getElementById("mDraw");
const mTilt = document.getElementById("mTilt");
const mTremor = document.getElementById("mTremor");
const mFps = document.getElementById("mFps");

// Offscreen canvas for 320x320 letterbox preprocessing
const prepCanvas = document.createElement("canvas");
prepCanvas.width = INPUT_SIZE;
prepCanvas.height = INPUT_SIZE;
const prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

let session = null;
let stream = null;
let running = false;
let isBusy = false;
let lastFrameTime = performance.now();

const analyzer = new ArcheryVisionAnalyzer({ bowSide: bowSideSelect.value });

const PHASE_COLORS = {
  Idle: "#cbd5e1",
  Draw: "#facc15",
  "Aim (Hold)": "#4ade80",
  Release: "#fb7185",
};

async function initModel() {
  try {
    statusEl.textContent = "Đang khởi tạo ONNX Runtime...";
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgpu", "wasm"],
    });
    statusEl.textContent = "Model sẵn sàng! Nhấn 'Bật Camera'.";
    toggleCamBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Lỗi tải model: ${err.message}`;
  }
}

async function startCamera() {
  stopCamera();
  const facingMode = facingSelect.value;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    running = true;
    toggleCamBtn.textContent = "Tắt Camera";
    statusEl.textContent = "Đang nhận diện...";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Không mở được camera (cần HTTPS trên điện thoại): ${err.message}`;
  }
}

function stopCamera() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  toggleCamBtn.textContent = "Bật Camera";
}

function preprocessFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = Math.floor((INPUT_SIZE - nw) / 2);
  const padY = Math.floor((INPUT_SIZE - nh) / 2);

  prepCtx.fillStyle = "rgb(114, 114, 114)";
  prepCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  prepCtx.drawImage(video, 0, 0, vw, vh, padX, padY, nw, nh);

  const imgData = prepCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0; i < planeSize; i++) {
    const idx = i * 4;
    floatData[i] = imgData[idx] / 255.0;
    floatData[planeSize + i] = imgData[idx + 1] / 255.0;
    floatData[2 * planeSize + i] = imgData[idx + 2] / 255.0;
  }

  return {
    tensor: new ort.Tensor("float32", floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    padX,
    padY,
  };
}

function extractBestPersonKeypoints(outputTensor, scale, padX, padY) {
  // YOLO11n-pose output shape: [1, 56, 2100]
  const data = outputTensor.data;
  const numAnchors = outputTensor.dims[2];

  let bestIdx = -1;
  let bestScore = PERSON_CONF_THRESHOLD;
  for (let i = 0; i < numAnchors; i++) {
    const score = data[4 * numAnchors + i];
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  if (bestIdx === -1) return null;

  const keypoints = [];
  for (let k = 0; k < 17; k++) {
    const base = (5 + k * 3) * numAnchors + bestIdx;
    const x = (data[base] - padX) / scale;
    const y = (data[base + numAnchors] - padY) / scale;
    const conf = data[base + 2 * numAnchors];
    keypoints.push([x, y, conf]);
  }
  return keypoints;
}

function drawSkeleton(keypoints, mirror) {
  ctx.save();
  if (mirror) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (keypoints) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#38bdf8";
    for (const [i, j] of SKELETON_EDGES) {
      if (keypoints[i][2] > 0.35 && keypoints[j][2] > 0.35) {
        ctx.beginPath();
        ctx.moveTo(keypoints[i][0], keypoints[i][1]);
        ctx.lineTo(keypoints[j][0], keypoints[j][1]);
        ctx.stroke();
      }
    }

    for (let k = 0; k < keypoints.length; k++) {
      if (keypoints[k][2] > 0.35) {
        ctx.fillStyle = k <= 10 && k >= 5 ? "#facc15" : "#22c55e";
        ctx.beginPath();
        ctx.arc(keypoints[k][0], keypoints[k][1], 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

function updateMetricsUI(metrics, fps) {
  mFps.textContent = fps.toFixed(1);
  if (!metrics) {
    mPhase.textContent = "Chưa thấy đủ khớp";
    mPhase.style.color = "#94a3b8";
    return;
  }
  mPhase.textContent = metrics.phase;
  mPhase.style.color = PHASE_COLORS[metrics.phase] || "#f8fafc";
  mBow.textContent = `${metrics.bowElbowAngle.toFixed(1)}°`;
  mDraw.textContent = `${metrics.drawElbowAngle.toFixed(1)}°`;
  mTilt.textContent = `${metrics.shoulderTiltDeg.toFixed(1)}°`;
  mTremor.textContent = metrics.bowHandSteadiness.toFixed(4);
}

async function loop() {
  if (!running) return;
  if (!isBusy && video.readyState >= 2) {
    isBusy = true;
    try {
      const { tensor, scale, padX, padY } = preprocessFrame();
      const outputs = await session.run({ [session.inputNames[0]]: tensor });
      const outTensor = outputs[session.outputNames[0]];
      const keypoints = extractBestPersonKeypoints(outTensor, scale, padX, padY);
      const metrics = keypoints ? analyzer.analyzeFrame(keypoints) : null;

      const now = performance.now();
      const fps = 1000 / Math.max(now - lastFrameTime, 1);
      lastFrameTime = now;

      drawSkeleton(keypoints, facingSelect.value === "user");
      updateMetricsUI(metrics, fps);
    } catch (err) {
      console.error(err);
    } finally {
      isBusy = false;
    }
  }
  requestAnimationFrame(loop);
}

toggleCamBtn.addEventListener("click", () => {
  if (running) stopCamera();
  else startCamera();
});

facingSelect.addEventListener("change", () => {
  if (running) startCamera();
});

bowSideSelect.addEventListener("change", () => {
  analyzer.setBowSide(bowSideSelect.value);
});

initModel();
