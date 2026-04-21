"""
app.py — AudioMark Flask Backend
=================================
Imports embed.py and extract.py directly and calls their functions.
The web UI posts files + password here; this runs the real Python pipeline
and returns the result file.

Setup:
    pip install flask reedsolo pydub pywavelets numpy Pillow cryptography matplotlib
    (ffmpeg must be on PATH for MP3 output)

Run:
    python app.py

Then open:
    http://localhost:5000
"""

import os
import sys
import tempfile
import shutil
import importlib.util
import base64

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS   # pip install flask-cors

# ── Load embed and extract from the same folder ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_module(name, path):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

embed_path   = os.path.join(BASE_DIR, "embed.py")
extract_path = os.path.join(BASE_DIR, "extract.py")

if not os.path.exists(embed_path):
    print(f"  ✗  embed.py not found at {embed_path}")
    sys.exit(1)
if not os.path.exists(extract_path):
    print(f"  ✗  extract.py not found at {extract_path}")
    sys.exit(1)

print("  Loading embed.py …", end="", flush=True)
embed_mod   = _load_module("embed",   embed_path)
print(" done")
print("  Loading extract.py …", end="", flush=True)
extract_mod = _load_module("extract", extract_path)
print(" done")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)   # allow browser requests from same or different origin


@app.route("/")
def index():
    """Serve the frontend HTML."""
    return app.send_static_file("index.html")


# ── /embed ────────────────────────────────────────────────────────────────────
@app.route("/embed", methods=["POST"])
def embed():
    """
    Multipart POST fields:
        audio    — WAV file
        logo     — image file (PNG/JPG/BMP)
        password — encryption password
        logo_w   — logo width  (default 64)
        logo_h   — logo height (default 64)
        output   — desired output filename (default watermarked.mp3)

    Returns the watermarked audio file as a download.
    """
    try:
        # ── validate inputs ──────────────────────────────────────────────────
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400
        if "logo" not in request.files:
            return jsonify({"error": "No logo file uploaded."}), 400

        password = request.form.get("password", "").strip()
        if not password:
            return jsonify({"error": "Password is required."}), 400

        logo_w   = int(request.form.get("logo_w", 64))
        logo_h   = int(request.form.get("logo_h", 64))
        out_name = request.form.get("output", "watermarked.mp3").strip() or "watermarked.mp3"

        # ── save uploads to temp dir ─────────────────────────────────────────
        tmp_dir = tempfile.mkdtemp(prefix="audiomark_embed_")
        try:
            audio_path  = os.path.join(tmp_dir, "input.wav")
            logo_path   = os.path.join(tmp_dir, "logo" + _ext(request.files["logo"].filename, ".png"))
            output_path = os.path.join(tmp_dir, out_name)

            request.files["audio"].save(audio_path)
            request.files["logo"].save(logo_path)

            # ── call embed.py's embed() function directly ──────────────────
            embed_mod.embed(
                audio_path  = audio_path,
                logo_path   = logo_path,
                output_path = output_path,
                password    = password,
                logo_size   = (logo_w, logo_h),
            )

            # embed() may produce .wav or .mp3 depending on output_path ext
            if not os.path.exists(output_path):
                # try .wav fallback
                wav_path = os.path.splitext(output_path)[0] + ".wav"
                if os.path.exists(wav_path):
                    output_path = wav_path
                    out_name    = os.path.basename(wav_path)
                else:
                    return jsonify({"error": "Embedding failed — output file not created."}), 500

            # ── Read audio file as base64 ─────────────────────────────────────
            with open(output_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            # ── Check for analysis image ──────────────────────────────────────
            analysis_path = os.path.splitext(output_path)[0] + "_analysis.png"
            analysis_base64 = None
            if os.path.exists(analysis_path):
                with open(analysis_path, "rb") as f:
                    analysis_base64 = base64.b64encode(f.read()).decode("utf-8")

            mime = "audio/mpeg" if output_path.endswith(".mp3") else "audio/wav"
            return jsonify({
                "success": True,
                "audio_base64": audio_base64,
                "audio_mime": mime,
                "audio_filename": out_name,
                "analysis_base64": analysis_base64,
            })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except SystemExit:
        return jsonify({"error": "Embedding failed — see server console for details."}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /extract ──────────────────────────────────────────────────────────────────
@app.route("/extract", methods=["POST"])
def extract():
    """
    Multipart POST fields:
        audio        — watermarked MP3 or WAV file
        password     — decryption password
        output       — desired output logo filename (default recovered_logo.png)
        orig_output  — desired output filename for reconstructed audio
                       (default original_reconstructed.wav).
                       Use .mp3 extension to get MP3 output.

    Returns JSON:
        {
          "success": true,
          "logo_base64":  "<base64>",
          "logo_mime":    "image/png",
          "logo_filename": "recovered_logo.png",
          "audio_base64":  "<base64>",   // reconstructed original audio
          "audio_mime":    "audio/wav",
          "audio_filename": "original_reconstructed.wav"
        }
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400

        password = request.form.get("password", "").strip()
        if not password:
            return jsonify({"error": "Password is required."}), 400

        out_logo = request.form.get("output", "recovered_logo.png").strip() or "recovered_logo.png"
        if not out_logo.lower().endswith((".png", ".jpg", ".bmp")):
            out_logo += ".png"

        out_orig = request.form.get("orig_output", "original_reconstructed.wav").strip() \
                   or "original_reconstructed.wav"

        tmp_dir = tempfile.mkdtemp(prefix="audiomark_extract_")
        try:
            audio_file  = request.files["audio"]
            audio_ext   = _ext(audio_file.filename, ".wav")
            audio_path  = os.path.join(tmp_dir, "watermarked" + audio_ext)
            logo_path   = os.path.join(tmp_dir, out_logo)
            orig_path   = os.path.join(tmp_dir, out_orig)

            audio_file.save(audio_path)

            # ── call extract.py's extract() function directly ──────────────
            extract_mod.extract(
                audio_path            = audio_path,
                password              = password,
                output_logo_path      = logo_path,
                output_original_path  = orig_path,
            )

            if not os.path.exists(logo_path):
                return jsonify({"error": "Extraction failed — output logo not created."}), 500

            # ── Read logo as base64 ───────────────────────────────────────
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode("utf-8")
            logo_mime = "image/png" if out_logo.endswith(".png") else "image/jpeg"

            # ── Read reconstructed audio as base64 (if produced) ─────────
            audio_base64  = None
            audio_mime    = None
            audio_out_name = None
            if os.path.exists(orig_path):
                with open(orig_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
                audio_mime     = "audio/mpeg" if orig_path.endswith(".mp3") else "audio/wav"
                audio_out_name = out_orig

            return jsonify({
                "success":        True,
                "logo_base64":    logo_base64,
                "logo_mime":      logo_mime,
                "logo_filename":  out_logo,
                "audio_base64":   audio_base64,
                "audio_mime":     audio_mime,
                "audio_filename": audio_out_name,
            })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except SystemExit:
        return jsonify({"error": "Extraction failed — see server console for details."}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /health ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "embed": embed_path, "extract": extract_path})


# ── helpers ───────────────────────────────────────────────────────────────────
def _ext(filename, fallback=".bin"):
    """Return the file extension from a filename, lower-cased."""
    if filename:
        _, ext = os.path.splitext(filename)
        if ext:
            return ext.lower()
    return fallback


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  AudioMark Flask Backend                             ║")
    print("║  Calling embed.py + extract.py directly           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  embed.py   : {os.path.basename(embed_path):<38}║")
    print(f"║  extract.py : {os.path.basename(extract_path):<38}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Open → http://localhost:5000")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
