import os
from app import app

# Railway always uses port 8080
port = int(os.environ.get("PORT", 8080))
print(f"Starting on port {port}", flush=True)
app.run(host="0.0.0.0", port=port, debug=False)
