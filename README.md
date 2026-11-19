# ClassMonitor AI

A Streamlit classroom-monitoring demo. YOLOv8m detects people, cell phones, books, and laptops. Activity labels are estimated from detected objects and person-box geometry. Analysis counts, class details, and manually confirmed attendance are stored locally in `classmonitor.sqlite3`. Uploaded photos and videos are not saved to the database.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open PowerShell in this project folder and create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies and start the app:

   ```powershell
   python -m pip install -r requirements.txt

   ```

The `yolov8m.pt` model is downloaded by Ultralytics on first use if it is not already beside `app.py`; the `attendance.xlsx` roster must remain beside the app. The video demo analyzes only its first frame. Activity labels are heuristic estimates, not confirmed behavior assessments. Talking is estimated from nearby person boxes and does not verify facing direction. Student numbers are assigned in detection order, and named Present/Absent report statuses must be manually confirmed before downloading the Excel report.

Phone detections take priority over Sleeping and are assigned to the nearest detected person. Sleeping requires closed-eye and head-down cues in two consecutive frames, so it is not labeled for the single-image capture and upload modes.

The Dashboard offers **Take Photo** using Streamlit's camera capture, plus **Photo Upload** and **Video Upload** (the video demo analyzes only its first frame).

Face registration and matching use the official [OpenCV Zoo YuNet model (MIT)](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace model (Apache-2.0)](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). YuNet is downloaded to `face_models` when the app starts; if it is unavailable, face registration falls back to OpenCV's Haar Cascade detector. If neither detector is available, the app continues and lets you save uploaded face photos without detection. SFace is downloaded when face matching is needed; if unavailable, Haar detection and color-histogram matching are used where possible while YOLO detection and activity analysis continue. Registered face photos stay in the local `known_faces` folder. Face matching aligns and compares face-only crops, displays a small green face box, and splits multiple detected faces inside YOLO person boxes before duplicate filtering, which then compares face boxes where available. After duplicate filtering, SFace assigns names one-to-one across the remaining faces using eligible scores above the 0.60 cosine-similarity threshold (higher is closer); close top scores do not force a face to be marked unknown. The sidebar shows the closest scores for unknown faces and lists registered students and photo counts. Use **Add More Photo for Same Student** or **Re-Register Face** to append photos without replacing existing registrations. Displayed match percentages are approximate similarity scores, not calibrated probabilities. Matches are approximate and must not be treated as verified identity or attendance records.

Voice alerts for sleeping, phone use, and talking activities use browser speech. Turn on **Voice Alert ON** and click **Test Voice** once to enable sound in the browser. **Stop Voice** stops speech without clearing the analysis; **SPEAK AGAIN** speaks the last saved analysis again.

Use the sidebar to select a class or add classes. The Dashboard saves analysis summaries to the shared local history; Analytics charts that history, Reports reads student names/rolls from `attendance.xlsx`, and Settings shows class details.
