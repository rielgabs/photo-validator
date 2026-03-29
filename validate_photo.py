#!/usr/bin/env python3
"""
Photo validator — memory-optimized for low-RAM servers (512MB).
Models are loaded ONCE at startup and reused for every request.

Usage:  python validate_photo.py <image_path>
Output: JSON { valid: bool, errors: [str] }
Install: pip install opencv-python-headless mediapipe numpy flask gunicorn
"""

import cv2, sys, json, math, os, urllib.request, numpy as np

# ── Suppress MediaPipe warnings ───────────────────────────────────────────────
os.environ['GLOG_minloglevel']    = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MP = True
except ImportError:
    HAS_MP = False

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DETECTOR_PATH   = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    downloads = [
        ("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", DETECTOR_PATH),
        ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", LANDMARKER_PATH),
    ]
    for url, path in downloads:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded {os.path.basename(path)}", flush=True)

# ── Singleton model instances — loaded ONCE, reused forever ──────────────────
_detector   = None
_landmarker = None

def get_detector():
    """Return cached FaceDetector, creating it only once."""
    global _detector
    if _detector is None and HAS_MP and os.path.exists(DETECTOR_PATH):
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_PATH),
            min_detection_confidence=0.5,
        )
        _detector = mp_vision.FaceDetector.create_from_options(opts)
        print("FaceDetector loaded.", flush=True)
    return _detector

def get_landmarker():
    """Return cached FaceLandmarker, creating it only once."""
    global _landmarker
    if _landmarker is None and HAS_MP and os.path.exists(LANDMARKER_PATH):
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            output_face_blendshapes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        print("FaceLandmarker loaded.", flush=True)
    return _landmarker

def init_models():
    """Call once at app startup to pre-load models into memory."""
    if HAS_MP:
        ensure_models()
        get_detector()
        get_landmarker()
        print("All models ready.", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def to_mp_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

def resize_if_large(img, max_dim=640):
    """Downscale large images before processing to save memory."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ── Face detection ────────────────────────────────────────────────────────────

def get_face_box(img_bgr, img_h, img_w):
    detector = get_detector()
    if detector:
        result = detector.detect(to_mp_image(img_bgr))
        dets   = result.detections or []
        if len(dets) != 1:
            return None, len(dets)
        bb = dets[0].bounding_box
        return (max(0, bb.origin_x), max(0, bb.origin_y), bb.width, bb.height), 1
    else:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cas   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cas.detectMultiScale(gray, 1.1, 5)
        if len(faces) != 1:
            return None, len(faces)
        return tuple(int(v) for v in faces[0]), 1

# ── Face landmarks ────────────────────────────────────────────────────────────

def get_landmarks(img_bgr):
    landmarker = get_landmarker()
    if not landmarker:
        return None
    img_h, img_w = img_bgr.shape[:2]
    result = landmarker.detect(to_mp_image(img_bgr))
    if not result.face_landmarks:
        return None
    raw = result.face_landmarks[0]
    return [(raw[i].x * img_w, raw[i].y * img_h) for i in range(len(raw))]

# ── Rule checks ───────────────────────────────────────────────────────────────

def check_face_coverage(pts, face_box, img_h):
    """Face must cover at least 40% of photo height."""
    if pts:
        chin_y     = pts[152][1]
        forehead_y = pts[10][1]
        span       = chin_y - forehead_y
        head_top_y = max(0.0, forehead_y - span * 0.15)
        ratio      = (chin_y - head_top_y) / img_h
    else:
        _, y, _, fh = face_box
        ratio = (fh * 1.25) / img_h
    if ratio < 0.40:
        return f"Face is too small ({ratio:.0%} of photo height). Move closer so your face fills at least 40% of the photo."
    return None

def check_centered(face_box, img_h, img_w):
    x, y, fw, fh = face_box
    cx = (x + fw / 2) / img_w
    if cx < 0.25:
        return "Face is too far to the left. Center your face in the frame."
    if cx > 0.75:
        return "Face is too far to the right. Center your face in the frame."
    return None

def check_eyes_open(pts, face_box, img_bgr):
    if pts:
        def ear(o, i, t, b):
            return math.dist(pts[t], pts[b]) / (math.dist(pts[o], pts[i]) + 1e-6)
        if ear(33, 133, 159, 145) < 0.15 and ear(362, 263, 386, 374) < 0.15:
            return "Eyes appear closed. Please keep your eyes open."
    else:
        x, y, fw, fh = face_box
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cas  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if len(cas.detectMultiScale(gray[y:y+fh, x:x+fw], 1.1, 5)) < 2:
            return "Eyes are not clearly visible. Ensure both eyes are open."
    return None

def check_head_tilt(pts):
    if not pts:
        return None
    le = ((pts[33][0] + pts[133][0]) / 2, (pts[33][1] + pts[133][1]) / 2)
    re = ((pts[362][0] + pts[263][0]) / 2, (pts[362][1] + pts[263][1]) / 2)
    angle = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    if abs(angle) > 20:
        return f"Head is tilted too much to the {'right' if angle > 0 else 'left'}. Keep your head straight."
    return None

def check_looking_at_camera(pts, img_w):
    if not pts:
        return None
    yaw = (pts[1][0] - (pts[234][0] + pts[454][0]) / 2) / (img_w * 0.5)
    if abs(yaw) > 0.25:
        return f"Face is turned too far to the {'right' if yaw > 0 else 'left'}. Look straight at the camera."
    return None

def check_background(img_bgr, face_box, img_h, img_w):
    x, y, fw, fh = face_box
    pad  = int(max(fw, fh) * 0.5)
    mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
    mask[max(0,y-pad):min(img_h,y+fh+pad), max(0,x-pad):min(img_w,x+fw+pad)] = 0
    bg   = img_bgr[mask == 255]
    if bg.size == 0:
        return None
    lab    = cv2.cvtColor(bg.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float)
    L_mean = lab[:,0].mean()
    col_var = lab[:,1].std() + lab[:,2].std()
    if L_mean < 150:
        return "Background is too dark. Use a plain white or light background."
    if col_var > 25:
        return "Background is not plain. Remove objects or patterns behind you."
    hsv = cv2.cvtColor(bg.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    if hsv[:,1].mean() > 60:
        return "Background has a strong color. Use a plain white background."
    return None

def check_clothing(img_bgr, face_box, img_h, img_w):
    x, y, fw, fh = face_box
    cy1 = min(img_h-1, y+fh)
    cy2 = min(img_h, y+fh+int(fh*0.8))
    cx1 = max(0, x-int(fw*0.3))
    cx2 = min(img_w, x+fw+int(fw*0.3))
    if cy2 <= cy1:
        return None
    region = img_bgr[cy1:cy2, cx1:cx2]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].flatten().astype(float)
    val = hsv[:,:,2].flatten().astype(float)
    if np.sum((sat > 200) & (val > 150)) / len(sat) > 0.30:
        return "Clothing color is too bright or flashy. Wear plain formal attire."
    return None

# ── Main validate function ────────────────────────────────────────────────────

def validate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"valid": False, "errors": ["Image could not be loaded."]}

    # Resize large images to save memory during processing
    img = resize_if_large(img, max_dim=640)
    img_h, img_w = img.shape[:2]

    face_box, count = get_face_box(img, img_h, img_w)
    if face_box is None:
        msg = "No face detected. Ensure your face is clearly visible." if count == 0 \
              else "Multiple faces detected. Only one person should be in the photo."
        return {"valid": False, "errors": [msg]}

    pts = get_landmarks(img)

    errors = [e for e in [
        check_face_coverage(pts, face_box, img_h),
        check_centered(face_box, img_h, img_w),
        check_looking_at_camera(pts, img_w),
        check_eyes_open(pts, face_box, img),
        check_head_tilt(pts),
        check_background(img, face_box, img_h, img_w),
        check_clothing(img, face_box, img_h, img_w),
    ] if e is not None]

    return {"valid": len(errors) == 0, "errors": errors}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "errors": ["No image path provided."]}))
        sys.exit(1)
    init_models()
    print(json.dumps(validate(sys.argv[1]), indent=2))
