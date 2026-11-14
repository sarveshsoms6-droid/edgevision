"""Unit and integration tests for EdgeVision Flask web application."""

from __future__ import annotations

import io
from pathlib import Path
import time
import pytest

from app.server import app, jobs, active_job_id


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify home page loads successfully with upload form."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "EdgeVision" in html
    assert "Upload Traffic Video" in html
    assert "60 seconds" in html


def test_upload_no_file(client):
    """Verify rejection when no file is provided."""
    resp = client.post("/upload", data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_upload_disallowed_extension(client):
    """Verify rejection of files with invalid extensions."""
    data = {
        "video": (io.BytesIO(b"fake data"), "script.exe"),
    }
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_data = resp.get_json()
    assert "Disallowed file format" in res_data["error"]


def test_upload_valid_video_and_poll(client):
    """Verify uploading a valid video clip and polling status."""
    sample_video = Path(__file__).resolve().parent.parent / "sample_data" / "traffic_sample.mp4"
    if not sample_video.is_file():
        pytest.skip("sample_data/traffic_sample.mp4 not found")

    with open(sample_video, "rb") as f:
        video_bytes = f.read()

    data = {
        "video": (io.BytesIO(video_bytes), "test_clip.mp4"),
        "runtime": "pytorch-reference",
        "stride": "4",
        "conf_threshold": "0.40",
    }
    resp = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 200
    res_json = resp.get_json()
    assert "job_id" in res_json
    job_id = res_json["job_id"]

    # Poll status
    status_resp = client.get(f"/status/{job_id}")
    assert status_resp.status_code == 200
    status_json = status_resp.get_json()
    assert status_json["job_id"] == job_id
    assert status_json["state"] in ("pending", "processing", "completed")

    # Wait for completion
    max_wait = 30
    start = time.time()
    while time.time() - start < max_wait:
        poll = client.get(f"/status/{job_id}").get_json()
        if poll["state"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert poll["state"] == "completed"

    # Verify result page renders
    result_page = client.get(f"/result/{job_id}")
    assert result_page.status_code == 200
    res_html = result_page.data.decode("utf-8")
    assert "Analysis Results" in res_html
    assert "Total Vehicles" in res_html

    # Test download endpoints
    csv_resp = client.get(f"/download/{job_id}/csv")
    assert csv_resp.status_code == 200
    assert b"total_vehicles" in csv_resp.data

    bench_resp = client.get(f"/download/{job_id}/benchmark")
    assert bench_resp.status_code == 200
    assert b"runtime_used" in bench_resp.data

    summary_resp = client.get(f"/download/{job_id}/summary")
    assert summary_resp.status_code == 200
    assert b"EdgeVision" in summary_resp.data

    # Reset active job lock for subsequent tests
    import app.server
    with app.server.job_lock:
        app.server.active_job_id = None


def test_upload_valid_video_yolov11(client):
    """Verify uploading a valid video clip with YOLOv11 runtime."""
    import app.server
    with app.server.job_lock:
        app.server.active_job_id = None
    sample_video = Path(__file__).resolve().parent.parent / "sample_data" / "traffic_sample.mp4"
    if not sample_video.is_file():
        pytest.skip("sample_data/traffic_sample.mp4 not found")

    with open(sample_video, "rb") as f:
        video_bytes = f.read()

    data = {
        "video": (io.BytesIO(video_bytes), "test_clip_yolo.mp4"),
        "runtime": "yolov11",
        "stride": "4",
        "conf_threshold": "0.35",
    }
    resp = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 200
    res_json = resp.get_json()
    assert "job_id" in res_json
    job_id = res_json["job_id"]

    max_wait = 20
    start = time.time()
    while time.time() - start < max_wait:
        poll = client.get(f"/status/{job_id}").get_json()
        if poll["state"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert poll["state"] == "completed"
    result_page = client.get(f"/result/{job_id}")
    assert result_page.status_code == 200
    assert "yolov11" in result_page.data.decode("utf-8")

    with app.server.job_lock:
        app.server.active_job_id = None


def test_cancel_job(client):
    """Verify job cancellation and lock release."""
    jobs["test_cancel"] = {
        "job_id": "test_cancel",
        "state": "processing",
    }
    resp = client.get("/cancel/test_cancel", follow_redirects=True)
    assert resp.status_code == 200
    assert jobs["test_cancel"]["state"] == "failed"


def test_safe_stream_wrapper():
    """Verify SafeStreamWrapper swallows Errno 22 (EINVAL) without raising."""
    from edgevision.stream_utils import SafeStreamWrapper, safe_print

    class BrokenStream:
        def write(self, data):
            err = OSError("[Errno 22] Invalid argument")
            err.errno = 22
            raise err

        def flush(self):
            err = OSError("[Errno 22] Invalid argument")
            err.errno = 22
            raise err

    wrapper = SafeStreamWrapper(BrokenStream())
    # Should not raise exception
    assert wrapper.write("test data") == 9
    wrapper.flush()

    # safe_print should not raise
    safe_print("test print")
