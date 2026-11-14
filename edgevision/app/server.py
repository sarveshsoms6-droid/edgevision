"""Offline Flask web application for traffic video upload and analysis.
Runs offline without CDN or external dependencies.
Enforces single concurrent job execution, 100 MB upload limit, and 60-second video limit.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from edgevision.config import EdgeVisionConfig, load_config
from edgevision.pipeline import EdgeVisionPipeline, PipelineResult
from edgevision.stream_utils import patch_stdout_stderr, safe_print
from edgevision.video_io import get_video_metadata

patch_stdout_stderr()

app = Flask(
    __name__,
    template_folder=str(_project_root / "app" / "templates"),
    static_folder=str(_project_root / "app" / "static"),
)

CONFIG_PATH = _project_root / "config.yaml"
BASE_CONFIG: EdgeVisionConfig = load_config(CONFIG_PATH)

# Setup storage folders - redirect to /tmp in serverless/read-only environments like Vercel
_is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_data_dir = Path("/tmp/edgevision") if _is_serverless else _project_root

app.config["MAX_CONTENT_LENGTH"] = BASE_CONFIG.upload.max_mb * 1024 * 1024
UPLOAD_FOLDER = _data_dir / "uploads"
OUTPUT_FOLDER = _data_dir / "outputs"
TESTING_FOLDER = _data_dir / "testing"
TRAINING_FOLDER = _data_dir / "training"
try:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    TESTING_FOLDER.mkdir(parents=True, exist_ok=True)
    TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)
except Exception as _mk_err:
    safe_print(f"[Warning] Could not initialize directory: {_mk_err}")

# Single concurrent job lock for edge memory safety
job_lock = threading.Lock()
active_job_id: Optional[str] = None
jobs: Dict[str, Dict[str, Any]] = {}


def _run_job_worker(
    job_id: str,
    video_path: Path,
    out_dir: Path,
    runtime: str,
    stride: str | int,
    conf_thresh: float,
    save_video: bool = False,
) -> None:
    global active_job_id
    job = jobs[job_id]
    job["state"] = "processing"
    job["start_time"] = time.time()

    try:
        # Clone base configuration with user-specified overrides
        cfg = load_config(CONFIG_PATH)
        cfg.model.runtime = runtime  # type: ignore
        if str(stride).lower() == "auto":
            cfg.frame_stride = "auto"
        else:
            cfg.frame_stride = int(stride)
        cfg.conf_threshold = float(conf_thresh)
        cfg.output.save_video = bool(save_video)
        cfg.validate()

        pipeline = EdgeVisionPipeline(cfg)

        def progress_cb(processed: int, total: int, cur_fps: float) -> None:
            job["processed_frames"] = processed
            job["total_frames"] = total
            job["current_fps"] = cur_fps

        res: PipelineResult = pipeline.process_video(
            video_path=video_path,
            output_dir=out_dir,
            runtime_override=runtime,
            progress_callback=progress_cb,
        )

        job["state"] = "completed"
        job["result"] = res
        job["processed_frames"] = res.processed_frames
        job["total_frames"] = res.total_frames
        job["effective_fps"] = res.effective_fps
        job["end_to_end_fps"] = res.end_to_end_fps
        job["total_time"] = res.processing_time_seconds
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        safe_print(f"\n[JOB FAILURE] Job {job_id} failed:\n{tb}")
        job["state"] = "failed"
        job["error"] = f"{type(e).__name__}: {str(e)}"
        job["traceback"] = tb
    finally:
        with job_lock:
            if active_job_id == job_id:
                active_job_id = None


@app.route("/")
def index():
    """Render upload form with configuration defaults."""
    cfg = load_config(CONFIG_PATH)
    return render_template(
        "index.html",
        config=cfg,
        is_busy=(active_job_id is not None),
        active_job_id=active_job_id,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Handle video file upload and kick off processing thread."""
    global active_job_id

    # Check concurrent job lock
    with job_lock:
        if active_job_id is not None:
            active_state = jobs.get(active_job_id, {}).get("state", "processing")
            if active_state in ("pending", "processing"):
                return (
                    jsonify({
                        "error": "The system is currently busy processing another video. Please wait for it to complete.",
                        "busy": True,
                        "active_job_id": active_job_id,
                    }),
                    429,
                )

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400

    orig_name = secure_filename(file.filename)
    ext = orig_name.rsplit(".", 1)[-1].lower() if "." in orig_name else ""

    cfg = load_config(CONFIG_PATH)
    if ext not in cfg.upload.allowed:
        return (
            jsonify({
                "error": f"Disallowed file format .{ext}. Allowed formats: {', '.join(cfg.upload.allowed)}"
            }),
            400,
        )

    # Generate job ID and save file
    job_id = uuid.uuid4().hex[:8]
    saved_filename = f"{job_id}.{ext}"
    saved_path = UPLOAD_FOLDER / saved_filename
    file.save(str(saved_path))

    # Store user input into the testing folder
    import shutil
    try:
        test_filename = orig_name if not (TESTING_FOLDER / orig_name).exists() else f"{job_id}_{orig_name}"
        test_dest = TESTING_FOLDER / test_filename
        shutil.copy2(str(saved_path), str(test_dest))
        safe_print(f"[EdgeVision Server] Stored user input into testing folder: {test_dest}")
    except Exception as copy_err:
        safe_print(f"[Warning] Failed to copy user input to testing folder: {copy_err}")

    # Verify duration using OpenCV
    try:
        meta = get_video_metadata(saved_path)
        if meta.duration_seconds > cfg.upload.max_seconds:
            if saved_path.is_file():
                saved_path.unlink()
            return (
                jsonify({
                    "error": (
                        f"Video duration is {meta.duration_seconds:.1f} s, which exceeds "
                        f"the maximum allowed limit of {cfg.upload.max_seconds} s."
                    )
                }),
                400,
            )
    except Exception as e:
        if saved_path.is_file():
            saved_path.unlink()
        return jsonify({"error": f"Failed to read video file metadata: {e}"}), 400

    # User options overrides
    runtime = request.form.get("runtime", cfg.model.runtime)
    stride = request.form.get("stride", str(cfg.frame_stride))
    conf_thresh = float(request.form.get("conf_threshold", str(cfg.conf_threshold)))
    save_video_raw = request.form.get("save_video", "false").lower()
    save_video = save_video_raw in ("true", "1", "on", "yes")

    out_dir = OUTPUT_FOLDER / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Register job and launch thread
    with job_lock:
        active_job_id = job_id
        jobs[job_id] = {
            "job_id": job_id,
            "filename": orig_name,
            "filepath": str(saved_path),
            "output_dir": str(out_dir),
            "state": "pending",
            "processed_frames": 0,
            "total_frames": meta.total_frames,
            "current_fps": 0.0,
            "error": None,
            "result": None,
        }

    thread = threading.Thread(
        target=_run_job_worker,
        args=(job_id, saved_path, out_dir, runtime, stride, conf_thresh, save_video),
        daemon=True,
    )
    thread.start()

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify({
            "job_id": job_id,
            "status_url": f"/status/{job_id}",
            "job_url": f"/job/{job_id}",
        })
    return redirect(url_for("status_page", job_id=job_id))


@app.route("/cancel/<job_id>", methods=["POST", "GET"])
def cancel_job(job_id: str):
    """Cancel or reset active job lock."""
    global active_job_id
    with job_lock:
        if active_job_id == job_id:
            active_job_id = None
        if job_id in jobs and jobs[job_id]["state"] in ("pending", "processing"):
            jobs[job_id]["state"] = "failed"
            jobs[job_id]["error"] = "Job was cancelled by user."
    return redirect(url_for("index"))


@app.route("/status/<job_id>")
def status(job_id: str):
    """JSON status API for client polling."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job["job_id"],
        "state": job["state"],
        "processed_frames": job["processed_frames"],
        "total_frames": job["total_frames"],
        "current_fps": round(job.get("current_fps", 0.0), 1),
        "error": job.get("error"),
        "traceback": job.get("traceback"),
    })


@app.route("/job/<job_id>")
def status_page(job_id: str):
    """HTML progress page for a job."""
    job = jobs.get(job_id)
    if not job:
        abort(404)
    return render_template("status.html", job=job)


@app.route("/result/<job_id>")
def result(job_id: str):
    """Result dashboard showing metrics, sample images, and counts table."""
    job = jobs.get(job_id)
    if not job:
        abort(404)

    res: Optional[PipelineResult] = job.get("result")
    out_dir = Path(job.get("output_dir", ""))
    csv_file = out_dir / "counts.csv"

    if not res and not csv_file.is_file():
        return redirect(url_for("status_page", job_id=job_id))

    records = []
    if res and res.per_frame_counts:
        records = res.per_frame_counts
    elif csv_file.is_file():
        import csv
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        except Exception:
            records = []

    display_records = records[:50]
    total_records_count = len(records)

    # Relative paths for web serving
    samples = []
    if res and res.saved_sample_paths:
        for p in res.saved_sample_paths:
            p_obj = Path(p)
            samples.append(f"/outputs/{job_id}/samples/{p_obj.name}")
    else:
        samples_dir = out_dir / "samples"
        if samples_dir.is_dir():
            for p_obj in sorted(samples_dir.glob("*.jpg")):
                samples.append(f"/outputs/{job_id}/samples/{p_obj.name}")

    summary_text = ""
    summary_file = Path(job["output_dir"]) / "summary.md"
    if summary_file.is_file():
        summary_text = summary_file.read_text(encoding="utf-8")

    chart_url = None
    chart_file = Path(job["output_dir"]) / "counts_chart.png"
    if chart_file.is_file():
        chart_url = f"/outputs/{job_id}/counts_chart.png"

    video_url = None
    annotated_video = Path(job["output_dir"]) / "annotated_video.mp4"
    if annotated_video.is_file():
        video_url = f"/outputs/{job_id}/annotated_video.mp4"

    return render_template(
        "result.html",
        job=job,
        result=res,
        records=display_records,
        total_records_count=total_records_count,
        samples=samples,
        summary_text=summary_text,
        chart_url=chart_url,
        video_url=video_url,
    )


@app.route("/outputs/<job_id>/<path:subpath>")
def serve_output(job_id: str, subpath: str):
    """Serve generated sample images, charts, and artifacts."""
    target = OUTPUT_FOLDER / f"job_{job_id}" / subpath
    if not target.is_file():
        abort(404)
    return send_file(str(target))


@app.route("/download/<job_id>/<file_type>")
def download_artifact(job_id: str, file_type: str):
    """Download CSV, benchmark JSON, summary markdown, or annotated video artifacts."""
    job = jobs.get(job_id)
    if not job:
        abort(404)
    out_dir = Path(job["output_dir"])

    if file_type == "csv":
        target = out_dir / "counts.csv"
        download_name = f"edgevision_counts_{job_id}.csv"
    elif file_type == "benchmark":
        target = out_dir / "benchmark.json"
        download_name = f"edgevision_benchmark_{job_id}.json"
    elif file_type == "summary":
        target = out_dir / "summary.md"
        download_name = f"edgevision_summary_{job_id}.md"
    elif file_type == "video":
        target = out_dir / "annotated_video.mp4"
        download_name = f"edgevision_annotated_{job_id}.mp4"
    else:
        abort(404)

    if not target.is_file():
        abort(404)
    return send_file(str(target), as_attachment=True, download_name=download_name)


def main() -> None:
    """Run offline Flask server bound to LAN (0.0.0.0:5000) and open browser."""
    import webbrowser

    port = 5000
    local_url = f"http://127.0.0.1:{port}"

    print("=" * 60)
    print(f"[EdgeVision Server] Starting offline Flask server")
    print(f"[EdgeVision Server] Web Interface : {local_url}")
    print(f"[EdgeVision Server] Network Access : http://0.0.0.0:{port}")
    print("=" * 60)

    # Launch browser automatically after a short delay
    def _open_browser() -> None:
        time.sleep(1.2)
        try:
            print(f"[EdgeVision Server] Launching default web browser to: {local_url}")
            webbrowser.open(local_url)
        except Exception as e:
            safe_print(f"[EdgeVision Server] Could not launch browser automatically: {e}")

    threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
