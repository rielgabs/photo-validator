from flask import Flask, request, jsonify
import tempfile, os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/validate", methods=["POST"])
def validate_photo():
    if "photo" not in request.files:
        return jsonify({"valid": False, "errors": ["No file uploaded. Use field name 'photo'."]}), 400

    file = request.files["photo"]
    allowed = {"jpg", "jpeg", "png"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"valid": False, "errors": ["Only JPG and PNG accepted."]}), 400

    suffix = f".{ext}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        from validate_photo import validate
        result = validate(tmp_path)
    except Exception as e:
        return jsonify({"valid": False, "errors": [f"Error: {str(e)}"]}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return jsonify(result), (200 if result["valid"] else 422)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
