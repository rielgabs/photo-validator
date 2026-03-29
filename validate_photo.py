#!/usr/bin/env python3
"""
Photo validator — checks minimum requirements for ID / profile photos.
Usage:  python validate_photo.py <image_path>
Output: JSON { valid: bool, errors: [str] }

Install: pip install opencv-python mediapipe numpy
Models are auto-downloaded to ./models/ on first run.
"""

import cv2, sys, json, math, os, urllib.request, numpy as np

# ── MediaPipe setup ───────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MP = True
except ImportError:
    HAS_MP = False

MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
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
            urllib.request.urlretrieve(url, path)

def to_mp_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# ── Get face box ──────────────────────────────────────────────────────────────

def get_face_box(img_bgr, img_h, img_w):
    """Returns (x, y, w, h) of face or None. Uses MediaPipe or Haar fallback."""
    if HAS_MP and os.path.exists(DETECTOR_PATH):
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_PATH),
            min_detection_confidence=0.5,
        )
        with mp_vision.FaceDetector.create_from_options(opts) as det:
            result = det.detect(to_mp_image(img_bgr))
        dets = result.detections or []
        if len(dets) != 1:
            return None, len(dets)
        bb = dets[0].bounding_box
        return (max(0, bb.origin_x), max(0, bb.origin_y), bb.width, bb.height), 1
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cas  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cas.detectMultiScale(gray, 1.1, 5)
        if len(faces) != 1:
            return None, len(faces)
        return tuple(int(v) for v in faces[0]), 1

# ── Get face landmarks ────────────────────────────────────────────────────────

def get_landmarks(img_bgr):
    """Returns list of (x_px, y_px) for 478 landmarks, or None."""
    if not HAS_MP or not os.path.exists(LANDMARKER_PATH):
        return None
    img_h, img_w = img_bgr.shape[:2]
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        output_face_blendshapes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as lm:
        result = lm.detect(to_mp_image(img_bgr))
    if not result.face_landmarks:
        return None
    raw = result.face_landmarks[0]
    return [(raw[i].x * img_w, raw[i].y * img_h) for i in range(len(raw))]

# ── Individual checks — each returns a string error or None ───────────────────

def check_face_coverage(pts, face_box, img_h):
    """Face must cover at least 40% of photo height (chin to top of head)."""
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
        return f"Face is too small ({ratio:.0%} of photo height). Move closer so your face fills at least half the photo."
    return None

def check_centered(face_box, img_h, img_w):
    """Fail only if face is obviously far off to one side."""
    x, y, fw, fh = face_box
    cx = (x + fw / 2) / img_w
    if cx < 0.25:
        return "Face is too far to the left. Center your face in the frame."
    if cx > 0.75:
        return "Face is too far to the right. Center your face in the frame."
    return None

def check_eyes_open(pts, face_box, img_bgr):
    """Fail only if both eyes are clearly closed."""
    if pts:
        def ear(o, i, t, b):
            v = math.dist(pts[t], pts[b])
            h = math.dist(pts[o], pts[i]) + 1e-6
            return v / h
        left  = ear(33,  133, 159, 145)
        right = ear(362, 263, 386, 374)
        if left < 0.15 and right < 0.15:
            return "Eyes appear closed. Please keep your eyes open."
    else:
        x, y, fw, fh = face_box
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cas  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = cas.detectMultiScale(gray[y:y+fh, x:x+fw], 1.1, 5)
        if len(eyes) < 2:
            return "Eyes are not clearly visible. Ensure both eyes are open."
    return None

def check_head_tilt(pts):
    """Fail only if head is tilted more than 20°."""
    if not pts:
        return None
    le = ((pts[33][0] + pts[133][0]) / 2, (pts[33][1] + pts[133][1]) / 2)
    re = ((pts[362][0] + pts[263][0]) / 2, (pts[362][1] + pts[263][1]) / 2)
    angle = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    if abs(angle) > 20:
        side = "right" if angle > 0 else "left"
        return f"Head is tilted too much to the {side}. Keep your head straight."
    return None

def check_looking_at_camera(pts, img_w):
    """Fail only if face is obviously turned away from the camera."""
    if not pts:
        return None
    face_mid_x = (pts[234][0] + pts[454][0]) / 2
    yaw = (pts[1][0] - face_mid_x) / (img_w * 0.5)
    if abs(yaw) > 0.25:
        side = "right" if yaw > 0 else "left"
        return f"Face is turned too far to the {side}. Look straight at the camera."
    return None

def check_background(img_bgr, face_box, img_h, img_w):
    """Background must be plain and light/white."""
    x, y, fw, fh = face_box
    pad = int(max(fw, fh) * 0.5)
    mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
    mask[max(0,y-pad):min(img_h,y+fh+pad), max(0,x-pad):min(img_w,x+fw+pad)] = 0
    bg = img_bgr[mask == 255]
    if bg.size == 0:
        return None
    lab     = cv2.cvtColor(bg.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float)
    L_mean  = lab[:,0].mean()
    col_var = lab[:,1].std() + lab[:,2].std()
    if L_mean < 150:
        return "Background is too dark. Use a plain white or light background."
    if col_var > 25:
        return "Background is not plain. Remove objects or patterns behind you."
    if cv2.cvtColor(bg.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)[:,1].mean() > 60:
        return "Background has a strong color. Use a plain white background."
    return None

def check_clothing(img_bgr, face_box, img_h, img_w):
    """Fail only if clothing is clearly neon or extremely bright/flashy."""
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

# Removed: check_hair_covering_eyes — too many false positives
# Removed: check_accessories        — too many false positives
# Skipped: makeup check             — not possible with rule-based code

# ── Main ──────────────────────────────────────────────────────────────────────

def validate(image_path):
    errors = []

    img = cv2.imread(image_path)
    if img is None:
        return {"valid": False, "errors": ["Image could not be loaded."]}
    img_h, img_w = img.shape[:2]

    if HAS_MP:
        ensure_models()

    # Detect face
    face_box, count = get_face_box(img, img_h, img_w)
    if face_box is None:
        if count == 0:
            errors.append("No face detected. Ensure your face is clearly visible.")
        else:
            errors.append("Multiple faces detected. Only one person should be in the photo.")
        return {"valid": False, "errors": errors}

    # Get landmarks once (reused by multiple checks)
    pts = get_landmarks(img)

    # Run all checks
    checks = [
        check_face_coverage(pts, face_box, img_h),
        check_centered(face_box, img_h, img_w),
        check_looking_at_camera(pts, img_w),
        check_eyes_open(pts, face_box, img),
        check_head_tilt(pts),
        check_background(img, face_box, img_h, img_w),
        check_clothing(img, face_box, img_h, img_w),
    ]

    errors = [e for e in checks if e is not None]

    return {
        "valid":  len(errors) == 0,
        "errors": errors,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "errors": ["No image path provided."]}))
        sys.exit(1)
    print(json.dumps(validate(sys.argv[1]), indent=2))
