"""Flask web app for EDNIG low-light image enhancement."""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
PAPER_DIR = ROOT / "paper"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytorch_impl.inference import EDNIGEnhancer, classical_illumination_enhance  # noqa: E402

MAX_UPLOAD_MB = 25
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_WEIGHTS_CANDIDATES = [
    ROOT / "weights" / "ednig_generator_latest.pt",
    ROOT / "weights" / "ednig_generator_best.pt",
    ROOT / "weights" / "generator.pt",
]
INFER_SIZE = int(os.environ.get("EDNIG_INFER_SIZE", "512"))

# PDFs in the paper/ folder that we expose
PAPER_FILES = {
    "original": "2507.13360v1_LowLightEnhancementViaEncoderDecoderNetworkWithIlluminationGuidance.pdf",
    "vietnamese": "NLPLab_LowLightEnhancementViaEncoderDecoderNetwordWithIlluminationGuidance_v2.pdf",
}

app = Flask(
    __name__,
    template_folder=str(THIS_DIR / "templates"),
    static_folder=str(THIS_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ENHANCER = EDNIGEnhancer(img_size=INFER_SIZE)
for cand in DEFAULT_WEIGHTS_CANDIDATES:
    if cand.exists():
        try:
            ENHANCER = EDNIGEnhancer(weights_path=str(cand), img_size=INFER_SIZE)
            print(f"[EDNIG] Loaded weights from {cand}")
            break
        except Exception as e:
            print(f"[EDNIG] Failed to load {cand}: {e}")

MODE = "model" if ENHANCER.has_weights() else "classical"
print(f"[EDNIG] Active enhancement mode: {MODE}")

_SESSION_CACHE = {}


def _allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


def _read_bgr(file_storage):
    raw = file_storage.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image (unsupported format or corrupt file).")
    return bgr


def _encode_png(bgr):
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return bytes(buf)


def _encode_jpeg(bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return bytes(buf)


def _b64(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


@app.route("/")
def index():
    paper_links = {
        "original_pdf": "/paper/original",
        "vietnamese_pdf": "/paper/vietnamese",
        "arxiv": "https://arxiv.org/abs/2507.13360",
        "ieee": "https://ieeexplore.ieee.org/abstract/document/11474151",
        "github": "https://github.com/tranleanh/ednig",
    }
    return render_template(
        "index.html",
        mode=MODE,
        max_mb=MAX_UPLOAD_MB,
        paper=paper_links,
    )


@app.route("/api/status")
def api_status():
    return jsonify({
        "mode": MODE,
        "device": str(ENHANCER.device) if ENHANCER.device is not None else "n/a",
        "infer_size": INFER_SIZE,
        "max_upload_mb": MAX_UPLOAD_MB,
    })


@app.route("/paper/<key>")
def paper_file(key):
    """Serve the original / Vietnamese PDF inline in the browser."""
    if key not in PAPER_FILES:
        abort(404)
    fname = PAPER_FILES[key]
    fpath = PAPER_DIR / fname
    if not fpath.exists():
        abort(404)
    return send_from_directory(
        str(PAPER_DIR),
        fname,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=fname,
    )


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image field in request."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400
    if not _allowed(f.filename):
        return jsonify({"ok": False, "error": f"Unsupported file type: {f.filename}"}), 400

    try:
        bgr = _read_bgr(f)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    t0 = time.time()
    try:
        if ENHANCER.has_weights():
            out_bgr = ENHANCER.enhance(bgr)
            used = "model"
        else:
            out_bgr = classical_illumination_enhance(bgr)
            used = "classical"
    except Exception as e:
        return jsonify({"ok": False, "error": f"Inference failed: {e}"}), 500
    ms = int((time.time() - t0) * 1000)

    h, w = out_bgr.shape[:2]
    sid = uuid.uuid4().hex
    _SESSION_CACHE[sid] = {
        "enhanced": _encode_png(out_bgr),
        "original": _encode_png(bgr),
    }
    if len(_SESSION_CACHE) > 32:
        _SESSION_CACHE.pop(next(iter(_SESSION_CACHE)), None)

    original_jpg = _encode_jpeg(bgr, quality=85)
    enhanced_jpg = _encode_jpeg(out_bgr, quality=92)

    return jsonify({
        "ok": True, "session": sid, "mode": used,
        "width": w, "height": h, "ms": ms,
        "original_b64": _b64(original_jpg),
        "enhanced_b64": _b64(enhanced_jpg),
    })


@app.route("/api/download")
def api_download():
    sid = request.args.get("session", "")
    kind = request.args.get("kind", "enhanced")
    fmt = request.args.get("format", "png").lower()
    if sid not in _SESSION_CACHE:
        return jsonify({"ok": False, "error": "Session expired or not found."}), 404
    if kind not in ("enhanced", "original"):
        return jsonify({"ok": False, "error": "Bad kind."}), 400

    png_bytes = _SESSION_CACHE[sid][kind]
    if fmt in ("jpg", "jpeg"):
        arr = np.frombuffer(png_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        out = _encode_jpeg(bgr, quality=95)
        mime, ext = "image/jpeg", "jpg"
    else:
        out, mime, ext = png_bytes, "image/png", "png"

    return send_file(io.BytesIO(out), mimetype=mime, as_attachment=True,
                     download_name=f"ednig_{kind}.{ext}")


if __name__ == "__main__":
    host = os.environ.get("EDNIG_HOST", "127.0.0.1")
    port = int(os.environ.get("EDNIG_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
