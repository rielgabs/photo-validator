# 📸 Photo Validator API

A rule-based photo validation API for school/university ID photos. Built with Python, OpenCV, and MediaPipe. Deployable to Railway in minutes.

---

## What It Checks

| Rule | Details |
|---|---|
| **Face detected** | Exactly one face must be present |
| **Face size** | Face must cover at least 50% of the photo height |
| **Face centered** | Face must not be far off to either side |
| **Looking at camera** | Face must not be turned away |
| **Eyes open** | Both eyes must be open |
| **Head tilt** | Head must not be tilted more than 20° |
| **Background** | Must be plain white or light-colored |
| **Clothing** | No neon or extremely bright/flashy colors |

> Makeup, accessories, and hair checks are intentionally excluded — they produce too many false positives with rule-based detection and would require a trained ML model to do reliably.

---

## API Endpoints

### `GET /`
Health check.

**Response**
```json
{
  "status": "ok",
  "message": "Photo validator API is running."
}
```

---

### `POST /validate`
Validate a photo. Send the image as `multipart/form-data` with the field name `photo`.

**Accepted formats:** JPG, JPEG, PNG  
**Max file size:** 10 MB

**Request**
```bash
curl -X POST https://your-app.up.railway.app/validate \
  -F "photo=@/path/to/photo.jpg"
```

**Response — passed**
```json
{
  "valid": true,
  "errors": []
}
```

**Response — failed** (HTTP 422)
```json
{
  "valid": false,
  "errors": [
    "Background is too dark. Use a plain white or light background.",
    "Head is tilted too much to the right. Keep your head straight."
  ]
}
```

**Response — bad request** (HTTP 400)
```json
{
  "valid": false,
  "errors": ["No file uploaded. Send a file with field name 'photo'."]
}
```

---

## Calling from Laravel

```php
use Illuminate\Support\Facades\Http;

$response = Http::attach(
    'photo',
    file_get_contents($request->file('photo')->path()),
    'photo.jpg'
)->post('https://your-app.up.railway.app/validate');

$result = $response->json();

if (! $result['valid']) {
    return back()->withErrors($result['errors']);
}
```

---

## Local Development

**Requirements**
- Python 3.10+
- pip

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the API locally**
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

**Test with curl**
```bash
curl -X POST http://localhost:5000/validate \
  -F "photo=@/path/to/photo.jpg"
```

**Run the validator directly (no server)**
```bash
python validate_photo.py /path/to/photo.jpg
```

> On first run, MediaPipe will automatically download two model files (~10 MB total) into a `models/` folder next to the script. This only happens once.

---

## Deploy to Railway

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/photo-validator.git
git push -u origin main
```

**2. Create a Railway project**
- Go to [railway.app](https://railway.app) and sign in
- Click **New Project → Deploy from GitHub repo**
- Select your repository
- Railway will auto-detect the config from `railway.json` and `nixpacks.toml`

**3. Get your URL**
- Once deployed, Railway assigns a public URL under **Settings → Networking → Generate Domain**
- Use that URL as your API endpoint

---

## Project Structure

```
photo-validator/
├── app.py              # Flask API
├── validate_photo.py   # Validation logic
├── requirements.txt    # Python dependencies
├── Procfile            # Process definition for Railway
├── railway.json        # Railway deployment config
└── nixpacks.toml       # System dependencies (libGL for OpenCV)
```

---

## Tech Stack

- **[Flask](https://flask.palletsprojects.com/)** — API framework
- **[OpenCV](https://opencv.org/)** — Image processing and Haar cascade fallback
- **[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)** — Face detection and landmark analysis
- **[Gunicorn](https://gunicorn.org/)** — Production WSGI server
- **[Railway](https://railway.app)** — Deployment platform

---

## Notes

- MediaPipe is used as a **pre-trained tool** for face/landmark detection — this project applies its own rule-based thresholds on top. No model training is involved.
- The `opencv-python-headless` package is used instead of `opencv-python` to avoid GUI dependencies on the server.
- The `nixpacks.toml` includes `libGL` and `libglib` system packages required by OpenCV on Linux.
