# Face recognition (MediaPipe Face Landmarker demo)

A small, static web demo that runs **MediaPipe Tasks Vision – FaceLandmarker** in the browser, draws face landmarks on a webcam stream, and shows a simple HUD:

- **Smile score** (from blendshapes)
- **Look-away score** (iris position + a lightweight head-yaw proxy)
- A short “dialog” line that changes based on those signals

This project is plain HTML/CSS/JS and loads dependencies via CDN.

## Quick start

Because `main.js` is an ES module and uses remote WASM/model assets, you should run it via a local web server (opening `index.html` directly often fails due to browser security/CORS rules).

From the project directory:

```bash
cd face-recognition
npx serve .
```

Then open the printed `http://localhost:...` URL and click **ENABLE WEBCAM**.

## How it works

- **FaceLandmarker** is loaded from `@mediapipe/tasks-vision@0.10.3` (CDN)
- The FaceLandmarker model (`face_landmarker.task`) is fetched at runtime
- Landmarks are drawn to `#output_canvas` over the mirrored webcam video
- Blendshapes are listed in the right panel

Key logic lives in `main.js`.

## Tuning

You can adjust responsiveness and thresholds in `main.js`:

- **Smoothing**: `SCORE_TAU_MS`
- **Look-away**: `LOOK_EYE_*`, `LOOK_HEAD_*`, `LOOK_ON`, `LOOK_OFF`
- **Dialog stability**: `DIALOG_HOLD_MS`, `DIALOG_MIN_SHOW_MS`

## Troubleshooting

- **Webcam button does nothing**: make sure you granted camera permission for `localhost`.
- **No camera on Safari / mobile**: camera APIs may require HTTPS; `localhost` typically works, but remote hosting should be HTTPS.
- **Blank page / console errors**: run via a local server (see Quick start), not by double-clicking `index.html`.

## Credits & license

This is based on the MediaPipe FaceLandmarker web example and includes code headers from the MediaPipe Authors (Apache License 2.0). Check file headers in `index.html`, `main.js`, and `style.css` for details.