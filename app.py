from flask import Flask, request, jsonify
import tempfile, os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max — reject big files early
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/validate", methods=["POST"])
def validate_photo():
    if "photo" not in request.files:
        return jsonify({"valid": False, "errors": ["No file uploaded. Use field name 'photo'."]}), 400

    file = request.files["photo"]
    if not file.filename or not allowed(file.filename):
        return jsonify({"valid": False, "errors": ["Invalid file. Only JPG and PNG accepted."]}), 400

    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        from validate_photo import validate
        result = validate(tmp_path)
    except Exception as e:
        return jsonify({"valid": False, "errors": [f"Validation error: {str(e)}"]}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return jsonify(result), (200 if result["valid"] else 422)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
