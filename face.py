import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = APP_DIR / "known_faces"
FACE_MODELS_DIR = APP_DIR / "face_models"
YUNET_MODEL_PATH = FACE_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = FACE_MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
HAAR_CASCADE_NAME = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    f"{HAAR_CASCADE_NAME}"
)
SFACE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
FACE_MODEL_INPUT_SIZE = (320, 320)
FACE_MATCH_THRESHOLD = 0.60
HISTOGRAM_FACE_MATCH_THRESHOLD = 0.90
FACE_ENGINE_LOCK = threading.RLock()


def ensure_face_model(model_path: Path, model_url: str, minimum_size: int):
    temporary_path = None
    try:
        if model_path.is_file() and model_path.stat().st_size >= minimum_size:
            return
        if model_path.exists():
            model_path.unlink()
        FACE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            with tempfile.NamedTemporaryFile(
                dir=FACE_MODELS_DIR,
                prefix=f"{model_path.name}.",
                suffix=".download",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                with urllib.request.urlopen(model_url, timeout=60) as response:
                    shutil.copyfileobj(response, temporary_file)
            if temporary_path.stat().st_size < minimum_size:
                temporary_path.unlink()
                temporary_path = None
                if attempt == 0:
                    continue
                raise RuntimeError(f"Downloaded model is too small: {model_path.name}.")
            temporary_path.replace(model_path)
            temporary_path = None
            return
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            f"Could not download {model_path.name} from OpenCV Zoo: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

class FaceRecognitionUnavailableError(RuntimeError):
    pass

class HaarCascadeFaceDetector:
    def __init__(self):
        cascade_path = Path(cv2.data.haarcascades) / HAAR_CASCADE_NAME
        if not cascade_path.is_file():
            cascade_path = FACE_MODELS_DIR / HAAR_CASCADE_NAME
            ensure_face_model(cascade_path, HAAR_CASCADE_URL, 100_000)
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError("OpenCV's Haar Cascade face detector is unavailable.")

@st.cache_resource
def load_face_detector():
    try:
        ensure_face_model(YUNET_MODEL_PATH, YUNET_MODEL_URL, 100_000)
        detector = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL_PATH),
            "",
            FACE_MODEL_INPUT_SIZE,
            0.7,
            0.3,
            5000,
        )
    except (RuntimeError, cv2.error):
        detector = None
    if detector is None:
        try:
            return HaarCascadeFaceDetector()
        except (RuntimeError, cv2.error, OSError):
            return None
    return detector

@st.cache_resource
def load_face_models():
    detector = load_face_detector()
    try:
        with st.spinner("Downloading recognition model..."):
            ensure_face_model(SFACE_MODEL_PATH, SFACE_MODEL_URL, 1_000_000)
    except RuntimeError as error:
        raise FaceRecognitionUnavailableError(str(error)) from error
    try:
        recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")
    except cv2.error as error:
        raise FaceRecognitionUnavailableError(
            f"OpenCV could not load the face-recognition model: {error}"
        ) from error
    if recognizer is None:
        raise FaceRecognitionUnavailableError(
            "OpenCV could not initialize the face recognizer."
        )
    return detector, recognizer

def detect_face_rows(detector, image_bgr):
    if detector is None:
        return []
    image_height, image_width = image_bgr.shape[:2]
    if image_height == 0 or image_width == 0:
        return []

    if not hasattr(detector, "setInputSize"):
        cascade = getattr(detector, "cascade", detector)
        if not hasattr(cascade, "detectMultiScale"):
            raise RuntimeError("OpenCV face detector does not support face detection.")
        grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(grayscale, 1.1, 5)
        return [
            np.array([x, y, width, height, *([0.0] * 10), 1.0], dtype=np.float32)
            for x, y, width, height in boxes
        ]

    input_width, input_height = FACE_MODEL_INPUT_SIZE
    scale = min(input_width / image_width, input_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    offset_x = (input_width - resized_width) // 2
    offset_y = (input_height - resized_height) // 2
    padded = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    padded[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized

    with FACE_ENGINE_LOCK:
        detector.setInputSize(FACE_MODEL_INPUT_SIZE)
        _, faces = detector.detect(padded)
    if faces is None:
        return []

    mapped_faces = []
    for face in faces:
        mapped = face.astype(np.float32, copy=True)
        mapped[0] = (mapped[0] - offset_x) / scale
        mapped[1] = (mapped[1] - offset_y) / scale
        mapped[2] /= scale
        mapped[3] /= scale
        for x_index in (4, 6, 8, 10, 12):
            mapped[x_index] = (mapped[x_index] - offset_x) / scale
        for y_index in (5, 7, 9, 11, 13):
            mapped[y_index] = (mapped[y_index] - offset_y) / scale
        mapped[0] = np.clip(mapped[0], 0, image_width - 1)
        mapped[1] = np.clip(mapped[1], 0, image_height - 1)
        mapped_faces.append(mapped)
    return mapped_faces

def known_face_signature():
    if not KNOWN_FACES_DIR.is_dir():
        return ()
    supported_extensions = {".jpg", ".jpeg", ".png"}
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(KNOWN_FACES_DIR.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.suffix.lower() in supported_extensions
    )

def canonical_face_name(file_path: str | Path) -> str:
    stem = Path(file_path).stem
    return re.sub(r"__re_[0-9a-f]+_[123]$", "", stem, flags=re.IGNORECASE)

@st.cache_data(show_spinner=False)
def load_known_face_embeddings(signature):
    detector, recognizer = load_face_models()
    embeddings = []
    for file_path, _, _ in signature:
        try:
            image_bytes = np.fromfile(file_path, dtype=np.uint8)
        except OSError as error:
            raise RuntimeError(
                f"Could not read the registered face image: {file_path}"
            ) from error
        try:
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(
                    f"Could not read the registered face image: {file_path}"
                )
            faces = detect_face_rows(detector, image)
            if len(faces) != 1:
                raise RuntimeError(
                    f"The registered image must contain exactly one face: {file_path}"
                )
            with FACE_ENGINE_LOCK:
                aligned = recognizer.alignCrop(image, faces[0])
                embeddings.append(
                    (canonical_face_name(file_path), recognizer.feature(aligned))
                )
        except cv2.error as error:
            raise RuntimeError(
                f"Could not process the registered face image {file_path}: {error}"
            ) from error
    return embeddings

def detected_face_crop(image_bgr, face):
    image_height, image_width = image_bgr.shape[:2]
    x, y, width, height = (int(round(float(value))) for value in face[:4])
    x1 = max(0, min(image_width, x))
    y1 = max(0, min(image_height, y))
    x2 = max(x1, min(image_width, x + width))
    y2 = max(y1, min(image_height, y + height))
    return image_bgr[y1:y2, x1:x2]

def align_detected_face(recognizer, image_bgr, face):
    face_crop = detected_face_crop(image_bgr, face)
    if not face_crop.size:
        return None
    image_height, image_width = image_bgr.shape[:2]
    crop_x = max(0, min(image_width, int(round(float(face[0])))))
    crop_y = max(0, min(image_height, int(round(float(face[1])))))
    local_face = face.astype(np.float32, copy=True)
    local_face[0] = 0
    local_face[1] = 0
    local_face[2] = face_crop.shape[1]
    local_face[3] = face_crop.shape[0]
    for x_index in (4, 6, 8, 10, 12):
        local_face[x_index] -= crop_x
    for y_index in (5, 7, 9, 11, 13):
        local_face[y_index] -= crop_y
    return recognizer.alignCrop(face_crop, local_face)

def registered_face_photo_counts():
    counts = {}
    for file_path, _, _ in known_face_signature():
        name = canonical_face_name(file_path)
        key = name.casefold()
        if key in counts:
            counts[key][1] += 1
        else:
            counts[key] = [name, 1]
    return sorted(counts.values(), key=lambda item: item[0].casefold())

def ranked_face_scores(scores_by_name):
    scores_by_identity = {}
    for name, score in scores_by_name.items():
        key = name.casefold()
        current = scores_by_identity.get(key)
        if current is None:
            scores_by_identity[key] = (name, score)
        elif score > current[1]:
            scores_by_identity[key] = (current[0], score)
    return sorted(
        scores_by_identity.values(),
        key=lambda item: item[1],
        reverse=True,
    )

def maximum_weight_assignment(weights):
    row_count = len(weights)
    if not row_count:
        return []
    column_count = len(weights[0])
    if row_count > column_count:
        raise ValueError("Assignment requires at least as many columns as faces.")

    row_potentials = [0.0] * (row_count + 1)
    column_potentials = [0.0] * (column_count + 1)
    matched_rows = [0] * (column_count + 1)
    previous_columns = [0] * (column_count + 1)

    for row_index in range(1, row_count + 1):
        matched_rows[0] = row_index
        current_column = 0
        minimum_costs = [float("inf")] * (column_count + 1)
        visited_columns = [False] * (column_count + 1)
        while True:
            visited_columns[current_column] = True
            current_row = matched_rows[current_column]
            delta = float("inf")
            next_column = 0
            for column_index in range(1, column_count + 1):
                if visited_columns[column_index]:
                    continue
                cost = (
                    -weights[current_row - 1][column_index - 1]
                    - row_potentials[current_row]
                    - column_potentials[column_index]
                )
                if cost < minimum_costs[column_index]:
                    minimum_costs[column_index] = cost
                    previous_columns[column_index] = current_column
                if minimum_costs[column_index] < delta:
                    delta = minimum_costs[column_index]
                    next_column = column_index
            for column_index in range(column_count + 1):
                if visited_columns[column_index]:
                    row_potentials[matched_rows[column_index]] += delta
                    column_potentials[column_index] -= delta
                else:
                    minimum_costs[column_index] -= delta
            current_column = next_column
            if matched_rows[current_column] == 0:
                break
        while True:
            previous_column = previous_columns[current_column]
            matched_rows[current_column] = matched_rows[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assigned_columns = [-1] * row_count
    for column_index in range(1, column_count + 1):
        if matched_rows[column_index]:
            assigned_columns[matched_rows[column_index] - 1] = column_index - 1
    return assigned_columns

def assign_faces_one_to_one(faces, threshold, score_metric):
    ranked_scores = []
    names_by_identity = {}
    for face in faces:
        ranked = ranked_face_scores(face.get("scores_by_name", {}))
        ranked_scores.append(ranked)
        for name, _ in ranked:
            names_by_identity.setdefault(name.casefold(), name)
        face["match_scores"] = [
            {"name": name, "score": score}
            for name, score in ranked[:3]
        ]
        face["match_metric"] = score_metric
        face["match_threshold"] = threshold
        face["name"] = "Unknown Student"
        face["match_confidence"] = None

    if not faces or not names_by_identity:
        return

    identity_keys = list(names_by_identity)
    weights = []
    for ranked in ranked_scores:
        scores_by_key = {name.casefold(): score for name, score in ranked}
        weights.append(
            [
                scores_by_key.get(key, -1.0)
                if scores_by_key.get(key, -1.0) >= threshold
                else -1.0
                for key in identity_keys
            ]
            + [threshold - 1e-6] * len(faces)
        )

    for face_index, column_index in enumerate(maximum_weight_assignment(weights)):
        if column_index < 0 or column_index >= len(identity_keys):
            continue
        score = weights[face_index][column_index]
        if score < threshold:
            continue
        faces[face_index]["name"] = names_by_identity[identity_keys[column_index]]
        faces[face_index]["match_confidence"] = score

def face_color_histogram(face_bgr):
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram, alpha=1, norm_type=cv2.NORM_L1)

def prepare_person_face_candidates(image_bgr, people, detector):
    image_height, image_width = image_bgr.shape[:2]
    for person in people:
        x1, y1, x2, y2 = person["box"]
        x1, x2 = max(0, x1), min(image_width, x2)
        y1, y2 = max(0, y1), min(image_height, y2)
        person_crop = image_bgr[y1:y2, x1:x2]
        candidates = []
        if person_crop.size:
            for face in detect_face_rows(detector, person_crop):
                face_crop = detected_face_crop(person_crop, face)
                if not face_crop.size:
                    continue
                face_x = x1 + int(round(float(face[0])))
                face_y = y1 + int(round(float(face[1])))
                face_width = int(round(float(face[2])))
                face_height = int(round(float(face[3])))
                candidates.append(
                    {
                        "face_box": [
                            max(0, face_x),
                            max(0, face_y),
                            min(image_width, face_x + face_width),
                            min(image_height, face_y + face_height),
                        ],
                        "face_crop": face_crop,
                        "face_row": face,
                        "source_crop": person_crop,
                        "name": "Unknown Student",
                        "match_confidence": None,
                        "match_scores": [],
                    }
                )
        person["face_candidates"] = candidates
        person["name"] = "Unknown Student"
        person["match_confidence"] = None

def match_people_with_histograms(people, signature, detector):
    known_histograms = []
    for file_path, _, _ in signature:
        try:
            image_bytes = np.fromfile(file_path, dtype=np.uint8)
        except OSError as error:
            raise RuntimeError(
                f"Could not read the registered face image: {file_path}"
            ) from error
        known_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if known_image is None:
            raise RuntimeError(f"Could not read the registered face image: {file_path}")
        known_faces = detect_face_rows(detector, known_image)
        if not known_faces:
            continue
        known_face = max(known_faces, key=lambda face: face[2] * face[3])
        known_crop = detected_face_crop(known_image, known_face)
        if known_crop.size:
            known_histograms.append(
                (canonical_face_name(file_path), face_color_histogram(known_crop))
            )

    for person in people:
        histogram = face_color_histogram(person["face_crop"])
        scores_by_name = {}
        for name, known_histogram in known_histograms:
            score = float(
                cv2.compareHist(
                    histogram,
                    known_histogram,
                    cv2.HISTCMP_CORREL,
                )
            )
            scores_by_name[name] = max(
                scores_by_name.get(name, float("-inf")),
                score,
            )
        person["scores_by_name"] = scores_by_name
    assign_faces_one_to_one(
        people,
        HISTOGRAM_FACE_MATCH_THRESHOLD,
        "color-histogram correlation",
    )

def match_people_to_known_faces(people):
    detector = load_face_detector()
    if detector is None:
        for person in people:
            person["name"] = "Unknown Student"
            person["match_confidence"] = None
            person["match_scores"] = []
        return

    matchable_faces = [person for person in people if person.get("face_crop") is not None]
    if not matchable_faces:
        return
    signature = known_face_signature()
    if not signature:
        for person in people:
            person["name"] = "Unknown Student"
            person["match_confidence"] = None
            person["match_scores"] = []
        return

    if not hasattr(detector, "setInputSize"):
        try:
            match_people_with_histograms(matchable_faces, signature, detector)
        except (RuntimeError, cv2.error) as error:
            st.warning(f"Fallback face matching is unavailable: {error}")
        return

    try:
        gallery = load_known_face_embeddings(signature)
        if not gallery:
            return
        detector, recognizer = load_face_models()
    except FaceRecognitionUnavailableError:
        st.warning(
            "Face recognition is unavailable; using Haar and histogram matching when possible."
        )
        try:
            match_people_with_histograms(matchable_faces, signature, detector)
        except (RuntimeError, cv2.error) as error:
            st.warning(f"Fallback face matching is unavailable: {error}")
        return

    try:
        for person in matchable_faces:
            with FACE_ENGINE_LOCK:
                aligned = align_detected_face(
                    recognizer,
                    person["source_crop"],
                    person["face_landmarks"],
                )
                if aligned is None:
                    person["scores_by_name"] = {}
                    continue
                feature = recognizer.feature(aligned)
                scores_by_name = {}
                for name, known_feature in gallery:
                    score = float(
                        recognizer.match(
                            feature,
                            known_feature,
                            cv2.FaceRecognizerSF_FR_COSINE,
                        )
                    )
                    scores_by_name[name] = max(
                        scores_by_name.get(name, float("-inf")),
                        score,
                    )
            person["scores_by_name"] = scores_by_name
        assign_faces_one_to_one(
            matchable_faces,
            FACE_MATCH_THRESHOLD,
            "SFace cosine similarity",
        )
    except cv2.error as error:
        raise RuntimeError(f"OpenCV face matching failed: {error}") from error

def save_known_face(name: str, image_bgr):
    clean_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip()).strip(" .")
    if not clean_name:
        raise ValueError("Enter a valid name for the registered face.")
    if clean_name.split(".", maxsplit=1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        clean_name = f"_{clean_name}"

    detector = load_face_detector()
    if detector is not None:
        try:
            faces = detect_face_rows(detector, image_bgr)
        except cv2.error as error:
            raise RuntimeError(
                f"Could not detect a face in the uploaded photo: {error}"
            ) from error
        if len(faces) != 1:
            raise ValueError(
                "The uploaded photo must contain exactly one detectable face."
            )

    try:
        success, encoded = cv2.imencode(".jpg", image_bgr)
    except cv2.error as error:
        raise RuntimeError(f"Could not encode the face photo: {error}") from error
    if not success:
        raise RuntimeError("Could not encode the face photo as a JPEG.")
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = KNOWN_FACES_DIR / f"{clean_name}.jpg"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=KNOWN_FACES_DIR,
            prefix=f"{clean_name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(encoded.tobytes())
        temporary_path.replace(image_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    load_known_face_embeddings.clear()
    return clean_name

def save_additional_face(name: str, image_bgr) -> str:
    clean_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip()).strip(" .")
    if not clean_name:
        raise ValueError("Enter a valid name for the registered face.")
    if clean_name.split(".", maxsplit=1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        clean_name = f"_{clean_name}"

    detector = load_face_detector()
    if detector is not None:
        try:
            faces = detect_face_rows(detector, image_bgr)
        except cv2.error as error:
            raise RuntimeError(
                f"Could not detect a face in the uploaded photo: {error}"
            ) from error
        if len(faces) != 1:
            raise ValueError(
                "The uploaded photo must contain exactly one detectable face."
            )

    try:
        success, encoded = cv2.imencode(".jpg", image_bgr)
    except cv2.error as error:
        raise RuntimeError(f"Could not encode the face photo: {error}") from error
    if not success:
        raise RuntimeError("Could not encode the face photo as JPEG.")

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    registration_id = time.time_ns()
    image_path = KNOWN_FACES_DIR / (
        f"{clean_name}__re_{registration_id:x}_1.jpg"
    )
    while image_path.exists():
        registration_id += 1
        image_path = KNOWN_FACES_DIR / (
            f"{clean_name}__re_{registration_id:x}_1.jpg"
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=KNOWN_FACES_DIR,
            prefix=f"{clean_name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(encoded.tobytes())
        temporary_path.rename(image_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    load_known_face_embeddings.clear()
    return clean_name

def save_re_registered_faces(name: str, images_bgr: list[np.ndarray]) -> str:
    clean_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip()).strip(" .")
    if not clean_name:
        raise ValueError("Enter a valid name for the registered face.")
    if clean_name.split(".", maxsplit=1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        clean_name = f"_{clean_name}"
    if len(images_bgr) != 3:
        raise ValueError("Upload exactly three clear, front-facing photos.")

    detector = load_face_detector()
    if detector is not None:
        try:
            for image_bgr in images_bgr:
                if len(detect_face_rows(detector, image_bgr)) != 1:
                    raise ValueError(
                        "Each re-registration photo must contain exactly one detectable face."
                    )
        except cv2.error as error:
            raise RuntimeError(
                f"Could not detect a face in a re-registration photo: {error}"
            ) from error

    encoded_images = []
    try:
        for image_bgr in images_bgr:
            success, encoded = cv2.imencode(".jpg", image_bgr)
            if not success:
                raise RuntimeError("Could not encode a re-registration photo as JPEG.")
            encoded_images.append(encoded.tobytes())
    except cv2.error as error:
        raise RuntimeError(f"Could not encode a re-registration photo: {error}") from error

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    registration_id = f"{time.time_ns():x}"
    for index, image_bytes in enumerate(encoded_images, start=1):
        image_path = KNOWN_FACES_DIR / (
            f"{clean_name}__re_{registration_id}_{index}.jpg"
        )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=KNOWN_FACES_DIR,
                prefix=f"{clean_name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(image_bytes)
            temporary_path.replace(image_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    load_known_face_embeddings.clear()
    return clean_name

def person_detection_box(person):
    return person.get("face_box") or person["box"]

def detection_box_iou(first_box, second_box) -> float:
    intersection_width = max(
        0,
        min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]),
    )
    intersection_height = max(
        0,
        min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
    )
    intersection = intersection_width * intersection_height
    first_area = max(first_box[2] - first_box[0], 0) * max(
        first_box[3] - first_box[1], 0
    )
    second_area = max(second_box[2] - second_box[0], 0) * max(
        second_box[3] - second_box[1], 0
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0

@st.cache_resource
def load_eye_detector():
    eye_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
    if not eye_path.is_file():
        return None
    detector = cv2.CascadeClassifier(str(eye_path))
    return None if detector.empty() else detector

def sleeping_posture_flags(image_bgr, people):
    face_detector = load_face_detector()
    eye_detector = load_eye_detector()
    face_data = []
    for person in people:
        x1, y1, x2, y2 = person["box"]
        person_crop = image_bgr[y1:y2, x1:x2]
        if person_crop.size == 0:
            face_data.append(None)
            continue
        face_box = person.get("face_box")
        face = person.get("face_landmarks")
        if face_box:
            face_x1, face_y1, face_x2, face_y2 = face_box
            face_crop = image_bgr[face_y1:face_y2, face_x1:face_x2]
            face_height = max(face_y2 - face_y1, 1)
            face_center_y = (
                (face_y1 + face_height / 2 - y1) / max(person_crop.shape[0], 1)
            )
        else:
            faces = detect_face_rows(face_detector, person_crop)
            if not faces:
                face_data.append(None)
                continue
            face = max(faces, key=lambda row: row[2] * row[3])
            face_height = max(float(face[3]), 1.0)
            face_center_y = (float(face[1]) + face_height / 2) / max(
                person_crop.shape[0], 1
            )
            face_crop = detected_face_crop(person_crop, face)
        eyes_closed = False
        tilted_down = False
        if face_crop.shape[1] >= 50 and face_crop.shape[0] >= 40:
            if eye_detector is not None:
                upper_face = cv2.cvtColor(
                    face_crop[: max(1, int(face_crop.shape[0] * 0.65))],
                    cv2.COLOR_BGR2GRAY,
                )
                with FACE_ENGINE_LOCK:
                    eyes = eye_detector.detectMultiScale(
                        upper_face,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(max(8, int(face_crop.shape[1] * 0.12)), 8),
                    )
                eyes_closed = len(eyes) < 2
            if face is not None and hasattr(face_detector, "setInputSize"):
                eye_midpoint_y = (float(face[5]) + float(face[7])) / 2
                nose_y = float(face[9])
                tilted_down = (nose_y - eye_midpoint_y) / face_height < 0.22
        face_data.append(
            {
                "center_y": face_center_y,
                "eyes_closed": eyes_closed,
                "tilted_down": tilted_down,
            }
        )

    visible_positions = [
        item["center_y"] for item in face_data if item is not None
    ]
    typical_head_position = (
        float(np.median(visible_positions)) if visible_positions else None
    )
    sleeping = []
    for face in face_data:
        if face is None:
            sleeping.append(False)
            continue
        head_is_lower = (
            face["center_y"] > 0.45
            if typical_head_position is None
            else face["center_y"] > max(0.40, typical_head_position + 0.12)
        )
        sleeping.append(
            face["eyes_closed"] and (head_is_lower or face["tilted_down"])
        )
    return sleeping

def confirm_sleeping_for_two_frames(people, candidates, tracks):
    if tracks is None:
        return [False] * len(people)

    previous_tracks = list(tracks)
    matched_track_indices = set()
    current_tracks = []
    confirmed = []
    for person, candidate in zip(people, candidates):
        best_track_index = None
        best_overlap = 0.3
        for track_index, track in enumerate(previous_tracks):
            if track_index in matched_track_indices:
                continue
            overlap = detection_box_iou(person["box"], track["box"])
            if overlap > best_overlap:
                best_track_index = track_index
                best_overlap = overlap

        consecutive_frames = 0
        if candidate:
            consecutive_frames = (
                previous_tracks[best_track_index]["frames"] + 1
                if best_track_index is not None
                else 1
            )
        if best_track_index is not None:
            matched_track_indices.add(best_track_index)
        current_tracks.append(
            {
                "box": list(person["box"]),
                "frames": consecutive_frames,
            }
        )
        confirmed.append(consecutive_frames >= 2)

    tracks[:] = current_tracks
    return confirmed

def split_people_by_faces(people):
    split_people = []
    for person in people:
        candidates = person.pop("face_candidates", [])
        if not candidates:
            person["match_scores"] = []
            split_people.append(person)
            continue
        for candidate in candidates:
            face_person = dict(person)
            face_person["face_box"] = candidate["face_box"]
            face_person["face_landmarks"] = candidate["face_row"]
            face_person["face_crop"] = candidate["face_crop"]
            face_person["source_crop"] = candidate["source_crop"]
            face_person["name"] = candidate["name"]
            face_person["match_confidence"] = candidate["match_confidence"]
            face_person["match_scores"] = []
            split_people.append(face_person)
    return split_people
