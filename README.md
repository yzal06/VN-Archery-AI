# Archery AI Baseline (Wearable Sensor + YOLO-Pose Vision + Web ONNX)

Baseline implementation inspired by:
**"A Machine Learning–Based Multimodal Framework for Wearable Sensor-Based Archery Action Recognition and Stress Estimation"** (arXiv:2511.14057v1).

## Structure
- `src/config.py`: Hyperparameters from the paper (20 Hz sampling, 4s motion window, 30s PPG window, 70:30 split).
- `src/motion_pipeline.py`: Wearable accelerometer pipeline (`Total Acc`, `SmoothDiff`, `MotionLSTM`, `PQD`).
- `src/stress_pipeline.py`: Wearable PPG + HRV pipeline (`Bandpass`, `RR intervals`, 10 HRV features, `MLPClassifier`).
- `src/vision_pose_pipeline.py`: Camera/YOLO-Pose alternative computing bow/draw elbow angles, shoulder tilt, bow-hand steadiness, SmoothDiff-style wrist speed, and phase classification.
- `webcam_yolo_demo.py`: Real-time laptop webcam test using `yolo11n-pose.pt`.
- `web/`: Static web app running **`yolo11n-pose.onnx` (`320x320`, ~11.1 MB)** directly in the browser via `onnxruntime-web` (works on mobile phone cameras over HTTPS).

## 1. Laptop Webcam Test (Python)
```powershell
pip install -r requirements.txt
python webcam_yolo_demo.py --bow-side left
```

## 2. Web ONNX Test (Laptop & Mobile via GitHub Pages)
- **Local laptop test (`localhost`):**
  ```powershell
  python -m http.server 8000 --directory web
  ```
  Open `http://localhost:8000` in your browser.
- **Mobile phone test (requires HTTPS):**
  1. Push this repository to GitHub (`.gitignore` excludes `*.pt`, keeping `web/models/yolo11n-pose.onnx`).
  2. In GitHub repo **Settings → Pages**, deploy from branch `main` (root `/` or `/web` — if root `/`, open `https://<username>.github.io/<repo>/web/`).
  3. Open that HTTPS URL on your phone browser and allow camera access.

## 3. Run & Test with Docker
- **Run ML baseline verification and start the Web ONNX server (`http://localhost:8000`):**
  ```powershell
  docker compose up --build
  ```
  *(If using Docker Compose v1, run `docker-compose up --build`.)*
- **Run only the Python baseline test (`main_demo.py`):**
  ```powershell
  docker run --rm vn-archery-ai:latest python main_demo.py
  ```
