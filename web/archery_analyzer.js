/**
 * JavaScript port of ArcheryVisionAnalyzer (src/vision_pose_pipeline.py).
 * Computes bow/draw elbow angles, shoulder tilt, bow-hand steadiness,
 * SmoothDiff-inspired wrist speed, and archery phase classification.
 */

export const KP = {
  NOSE: 0,
  L_SHOULDER: 5,
  R_SHOULDER: 6,
  L_ELBOW: 7,
  R_ELBOW: 8,
  L_WRIST: 9,
  R_WRIST: 10,
};

export const SKELETON_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function computeAngleDeg(a, b, c) {
  const bax = a[0] - b[0];
  const bay = a[1] - b[1];
  const bcx = c[0] - b[0];
  const bcy = c[1] - b[1];
  const normBa = Math.hypot(bax, bay);
  const normBc = Math.hypot(bcx, bcy);
  if (normBa < 1e-6 || normBc < 1e-6) return 0;
  const cosVal = Math.max(-1, Math.min(1, (bax * bcx + bay * bcy) / (normBa * normBc)));
  return (Math.acos(cosVal) * 180) / Math.PI;
}

export class ArcheryVisionAnalyzer {
  constructor({ bowSide = "left", smoothWindow = 8, minConf = 0.35 } = {}) {
    this.bowSide = bowSide.toLowerCase();
    this.smoothWindow = smoothWindow;
    this.minConf = minConf;
    this.bowWristHistory = [];
    this.drawWristHistory = [];
    this.drawSpeedHistory = [];
    this.lastPhase = "Idle";
  }

  setBowSide(side) {
    this.bowSide = side.toLowerCase();
    this.bowWristHistory = [];
    this.drawWristHistory = [];
    this.drawSpeedHistory = [];
    this.lastPhase = "Idle";
  }

  _sideIndices() {
    if (this.bowSide === "left") {
      return {
        bow: [KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST],
        draw: [KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST],
      };
    }
    return {
      bow: [KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST],
      draw: [KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST],
    };
  }

  analyzeFrame(keypoints) {
    // keypoints: Array of 17 items: [x, y, conf]
    if (!keypoints || keypoints.length < 13) return null;

    const { bow, draw } = this._sideIndices();
    const required = [KP.NOSE, ...bow, ...draw];
    for (const idx of required) {
      if (keypoints[idx][2] < this.minConf) return null;
    }

    const lSh = keypoints[KP.L_SHOULDER];
    const rSh = keypoints[KP.R_SHOULDER];
    const shoulderWidth = Math.hypot(rSh[0] - lSh[0], rSh[1] - lSh[1]);
    if (shoulderWidth < 10) return null;

    const bowElbowAngle = computeAngleDeg(keypoints[bow[0]], keypoints[bow[1]], keypoints[bow[2]]);
    const drawElbowAngle = computeAngleDeg(keypoints[draw[0]], keypoints[draw[1]], keypoints[draw[2]]);

    let shoulderTiltDeg = Math.abs((Math.atan2(rSh[1] - lSh[1], rSh[0] - lSh[0]) * 180) / Math.PI);
    if (shoulderTiltDeg > 90) shoulderTiltDeg = 180 - shoulderTiltDeg;

    const bowWristNorm = [keypoints[bow[2]][0] / shoulderWidth, keypoints[bow[2]][1] / shoulderWidth];
    const drawWristNorm = [keypoints[draw[2]][0] / shoulderWidth, keypoints[draw[2]][1] / shoulderWidth];
    const noseNorm = [keypoints[KP.NOSE][0] / shoulderWidth, keypoints[KP.NOSE][1] / shoulderWidth];

    this.bowWristHistory.push(bowWristNorm);
    if (this.bowWristHistory.length > this.smoothWindow * 2) this.bowWristHistory.shift();

    this.drawWristHistory.push(drawWristNorm);
    if (this.drawWristHistory.length > this.smoothWindow * 2) this.drawWristHistory.shift();

    let bowHandSteadiness = 0;
    if (this.bowWristHistory.length >= 3) {
      const n = this.bowWristHistory.length;
      const meanX = this.bowWristHistory.reduce((s, p) => s + p[0], 0) / n;
      const meanY = this.bowWristHistory.reduce((s, p) => s + p[1], 0) / n;
      const stdX = Math.sqrt(this.bowWristHistory.reduce((s, p) => s + (p[0] - meanX) ** 2, 0) / n);
      const stdY = Math.sqrt(this.bowWristHistory.reduce((s, p) => s + (p[1] - meanY) ** 2, 0) / n);
      bowHandSteadiness = (stdX + stdY) / 2;
    }

    let speed = 0;
    if (this.drawWristHistory.length >= 2) {
      const prev = this.drawWristHistory[this.drawWristHistory.length - 2];
      const curr = this.drawWristHistory[this.drawWristHistory.length - 1];
      speed = Math.hypot(curr[0] - prev[0], curr[1] - prev[1]);
    }
    this.drawSpeedHistory.push(speed);
    if (this.drawSpeedHistory.length > this.smoothWindow) this.drawSpeedHistory.shift();

    const drawSmoothSpeed =
      this.drawSpeedHistory.reduce((s, v) => s + v, 0) / this.drawSpeedHistory.length;

    const wristToFace = Math.hypot(drawWristNorm[0] - noseNorm[0], drawWristNorm[1] - noseNorm[1]);
    const phase = this._classifyPhase(bowElbowAngle, drawElbowAngle, wristToFace, drawSmoothSpeed);

    return {
      bowElbowAngle,
      drawElbowAngle,
      shoulderTiltDeg,
      bowHandSteadiness,
      drawSmoothSpeed,
      phase,
    };
  }

  _classifyPhase(bowAngle, drawAngle, wristToFace, smoothSpeed) {
    const isBowExtended = bowAngle > 140;
    const isDrawAnchored = drawAngle < 110 && wristToFace < 1.35;

    let phase = "Idle";
    if (isBowExtended && isDrawAnchored) {
      phase = smoothSpeed < 0.03 ? "Aim (Hold)" : "Draw";
    } else if ((this.lastPhase === "Aim (Hold)" || this.lastPhase === "Draw") && smoothSpeed >= 0.08) {
      phase = "Release";
    }

    this.lastPhase = phase;
    return phase;
  }
}
