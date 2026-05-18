"""Flask web app for EDNIG low-light image enhancement.

Persists every enhancement run to disk under ``webapp/history/`` so the user
can review past inputs/outputs during a demo session. Each run gets its own
folder named ``<YYYYMMDD_HHMMSS>_<sid>`` containing:
    - input.png   : the original uploaded image (lossless)
    - output.png  : the enhanced image (lossless, full resolution)
    - meta.json   : run metadata (original filename, mode, dimensions, ms, etc.)
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
PAPER_DIR = ROOT / "paper"
HISTORY_DIR = THIS_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytorch_impl.inference import EDNIGEnhancer, classical_illumination_enhance  # noqa: E402

MAX_UPLOAD_MB = 25
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_WEIGHTS_CANDIDATES = [
    ROOT / "weights" / "ednig_generator_best.pt",    # preferred
    ROOT / "weights" / "ednig_generator_latest.pt",  # legacy fallback
    ROOT / "weights" / "generator.pt",
]
INFER_SIZE = int(os.environ.get("EDNIG_INFER_SIZE", "512"))
MAX_HISTORY = int(os.environ.get("EDNIG_MAX_HISTORY", "200"))

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
print(f"[EDNIG] History folder: {HISTORY_DIR}")


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


def _new_run_id():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = uuid.uuid4().hex[:10]
    return f"{ts}_{sid}"


def _read_live_model_info():
    """Always read ednig_model_info.json fresh from disk.

    Training rewrites this file every time a new best epoch is found, so the
    web app must re-read on each page load — caching at startup would show
    stale info.
    """
    info_path = ROOT / "weights" / "ednig_model_info.json"
    if not info_path.exists():
        return ENHANCER.info  # fall back to whatever was loaded at startup
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return ENHANCER.info


def _safe_run_path(run_id: str) -> Path:
    """Resolve a run folder safely (defends against path traversal)."""
    target = (HISTORY_DIR / run_id).resolve()
    if not str(target).startswith(str(HISTORY_DIR.resolve())):
        abort(400)
    return target


def _save_run(run_id: str, *, input_bgr, output_bgr, meta: dict):
    folder = HISTORY_DIR / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "input.png", "wb") as f:
        f.write(_encode_png(input_bgr))
    with open(folder / "output.png", "wb") as f:
        f.write(_encode_png(output_bgr))
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _prune_history():
    """Keep only the most recent MAX_HISTORY runs on disk."""
    if MAX_HISTORY <= 0:
        return
    entries = sorted(
        [p for p in HISTORY_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for old in entries[MAX_HISTORY:]:
        try:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        except OSError:
            pass


def _list_history(limit: int = 50):
    entries = sorted(
        [p for p in HISTORY_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    items = []
    for folder in entries[:limit]:
        meta_path = folder / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        items.append({
            "id": folder.name,
            "meta": meta,
            "thumb_input": f"/history-check/{folder.name}/input.png",
            "thumb_output": f"/history-check/{folder.name}/output.png",
        })
    return items


@app.route("/")
def index():
    paper_links = {
        "original_pdf": "/paper/original",
        "vietnamese_pdf": "/paper/vietnamese",
        "arxiv": "https://arxiv.org/abs/2507.13360",
        "ieee": "https://ieeexplore.ieee.org/abstract/document/11474151",
        "github": "https://github.com/tranleanh/ednig",
    }
    history_count = sum(1 for p in HISTORY_DIR.iterdir() if p.is_dir())
    return render_template(
        "index.html",
        mode=MODE,
        max_mb=MAX_UPLOAD_MB,
        paper=paper_links,
        history_count=history_count,
        model_info=_read_live_model_info(),  # always fresh from disk
    )


@app.route("/history-check")
def history_page():
    items = _list_history(limit=200)
    return render_template("history.html", items=items, mode=MODE)


@app.route("/history-check/api")
def api_history():
    return jsonify({"ok": True, "items": _list_history(limit=200)})


@app.route("/history-check/<run_id>/<kind>.png")
def serve_history_image(run_id, kind):
    if kind not in ("input", "output"):
        abort(404)
    folder = _safe_run_path(run_id)
    if not folder.is_dir():
        abort(404)
    return send_from_directory(str(folder), f"{kind}.png", mimetype="image/png", as_attachment=False)


@app.route("/api/status")
def api_status():
    return jsonify({
        "mode": MODE,
        "device": str(ENHANCER.device) if ENHANCER.device is not None else "n/a",
        "infer_size": INFER_SIZE,
        "max_upload_mb": MAX_UPLOAD_MB,
        "history_dir": str(HISTORY_DIR),
        "max_history": MAX_HISTORY,
        "model_info": _read_live_model_info(),
    })


@app.route("/paper/<key>")
def paper_file(key):
    if key not in PAPER_FILES:
        abort(404)
    fname = PAPER_FILES[key]
    fpath = PAPER_DIR / fname
    if not fpath.exists():
        abort(404)
    return send_from_directory(str(PAPER_DIR), fname, mimetype="application/pdf",
                                as_attachment=False, download_name=fname)


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image field in request."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400
    if not _allowed(f.filename):
        return jsonify({"ok": False, "error": f"Unsupported file type: {f.filename}"}), 400

    original_name = Path(f.filename).name
    try:
        bgr = _read_bgr(f)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # Optional mode override from form: "auto" | "single" | "tiled"
    enhance_mode = request.form.get("mode", "auto").lower()
    if enhance_mode not in ("auto", "single", "tiled"):
        enhance_mode = "auto"

    t0 = time.time()
    try:
        if ENHANCER.has_weights():
            out_bgr = ENHANCER.enhance(bgr, mode=enhance_mode)
            used = f"model:{enhance_mode}"
        else:
            out_bgr = classical_illumination_enhance(bgr)
            used = "classical"
    except Exception as e:
        return jsonify({"ok": False, "error": f"Inference failed: {e}"}), 500
    ms = int((time.time() - t0) * 1000)

    h, w = out_bgr.shape[:2]
    run_id = _new_run_id()
    meta = {
        "id": run_id,
        "original_filename": original_name,
        "mode": used,
        "width": w,
        "height": h,
        "input_mean_brightness": float(bgr.mean()),
        "output_mean_brightness": float(out_bgr.mean()),
        "ms": ms,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        _save_run(run_id, input_bgr=bgr, output_bgr=out_bgr, meta=meta)
        _prune_history()
    except Exception as e:
        print(f"[EDNIG] Failed to save history for {run_id}: {e}")

    original_jpg = _encode_jpeg(bgr, quality=85)
    enhanced_jpg = _encode_jpeg(out_bgr, quality=92)

    return jsonify({
        "ok": True,
        "session": run_id,
        "mode": used,
        "width": w,
        "height": h,
        "ms": ms,
        "original_b64": _b64(original_jpg),
        "enhanced_b64": _b64(enhanced_jpg),
    })


@app.route("/api/download")
def api_download():
    sid = request.args.get("session", "")
    kind = request.args.get("kind", "enhanced")
    fmt = request.args.get("format", "png").lower()
    if kind not in ("enhanced", "original"):
        return jsonify({"ok": False, "error": "Bad kind."}), 400

    # Map enhanced/original -> output/input
    file_kind = "output" if kind == "enhanced" else "input"
    folder = _safe_run_path(sid)
    src = folder / f"{file_kind}.png"
    if not src.exists():
        return jsonify({"ok": False, "error": "Session not found."}), 404

    png_bytes = src.read_bytes()
    if fmt in ("jpg", "jpeg"):
        arr = np.frombuffer(png_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        out = _encode_jpeg(bgr, quality=95)
        mime, ext = "image/jpeg", "jpg"
    else:
        out, mime, ext = png_bytes, "image/png", "png"

    return send_file(io.BytesIO(out), mimetype=mime, as_attachment=True,
                     download_name=f"ednig_{kind}.{ext}")


@app.route("/history-check/<run_id>", methods=["DELETE"])
def api_delete_run(run_id):
    folder = _safe_run_path(run_id)
    if not folder.is_dir():
        return jsonify({"ok": False, "error": "Not found"}), 404
    try:
        for f in folder.iterdir():
            f.unlink()
        folder.rmdir()
        return jsonify({"ok": True})
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("EDNIG_HOST", "127.0.0.1")
    port = int(os.environ.get("EDNIG_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
