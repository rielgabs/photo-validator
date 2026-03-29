import os

# Suppress MediaPipe warnings before anything else imports
os.environ['GLOG_minloglevel']     = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

from validate_photo import init_models
from app import app

# Pre-load models once at startup — avoids reloading per request
print("Loading models...", flush=True)
init_models()
print("Models ready. Starting server...", flush=True)

port = int(os.environ.get("PORT", 8080))
print(f"Listening on port {port}", flush=True)
app.run(host="0.0.0.0", port=port, debug=False)
