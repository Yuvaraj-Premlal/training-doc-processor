import os
import json
import logging
import datetime
import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas, ContentSettings

import video_indexer    as vi
import openai_client    as oai
import document_builder as db

logger = logging.getLogger("training_doc_processor")
app    = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# ══════════════════════════════════════════════════════════════════════
# HTTP: STATUS DASHBOARD
# ══════════════════════════════════════════════════════════════════════
@app.route(route="status", methods=["GET"])
def status(req: func.HttpRequest) -> func.HttpResponse:
    try:
        conn_str    = os.environ.get("STORAGE_CONNECTION_STRING", "")
        blob_client = BlobServiceClient.from_connection_string(conn_str)
        job_name    = req.params.get("job")

        if job_name:
            log = _load_json(blob_client, f"{job_name}/", "pipeline_log.json")
            if not log:
                return _json({"error": f"No job found: {job_name}"}, 404)
            return _json(_enrich_log(log))

        jobs = []
        try:
            container = blob_client.get_container_client("intermediate")
            seen = set()
            for blob in container.list_blobs():
                parts = blob.name.split("/")
                if len(parts) >= 2:
                    job = parts[0]
                    if job not in seen:
                        seen.add(job)
                        log = _load_json(blob_client, f"{job}/", "pipeline_log.json")
                        if log:
                            jobs.append(_enrich_log(log))
        except Exception as e:
            logger.warning(f"Could not list jobs: {e}")

        completed = []
        try:
            for blob in blob_client.get_container_client("outputs").list_blobs():
                if blob.name.endswith(".docx"):
                    completed.append({
                        "file":    blob.name,
                        "size_mb": round(blob.size / 1024 / 1024, 2),
                        "created": blob.last_modified.isoformat() if blob.last_modified else "",
                    })
        except Exception as e:
            logger.warning(f"Could not list outputs: {e}")

        return _json({
            "function_app":   "training-doc-processor",
            "status":         "running",
            "active_jobs":    jobs,
            "completed_docs": completed,
            "timestamp":      datetime.datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Status error: {e}", exc_info=True)
        return _json({"error": str(e)}, 500)


def _enrich_log(log: dict) -> dict:
    stage_weights = {
        "submit":    5,
        "indexing":  35,
        "captions":  20,
        "structure": 10,
        "content":   15,
        "quiz":       5,
        "build_doc":  5,
        "upload":     3,
        "cleanup":    2,
    }
    total_weight = sum(stage_weights.values())
    earned = 0
    stages = log.get("stages", {})
    for stage, weight in stage_weights.items():
        val = stages.get(stage, "pending")
        if isinstance(val, str):
            if val.startswith("done"):
                earned += weight
            elif val.startswith("running"):
                earned += weight * 0.5

    log["progress_pct"] = round((earned / total_weight) * 100, 1)

    stage_pct = {}
    for stage in stage_weights:
        val = stages.get(stage, "pending")
        if isinstance(val, str):
            if val.startswith("done"):
                stage_pct[stage] = "100%"
            elif val.startswith("running"):
                if stage == "indexing" and log.get("indexing_progress"):
                    stage_pct[stage] = log["indexing_progress"]
                elif stage == "captions" and log.get("captions_progress"):
                    stage_pct[stage] = log["captions_progress"]
                elif stage == "content" and log.get("content_progress"):
                    stage_pct[stage] = log["content_progress"]
                else:
                    stage_pct[stage] = "in progress"
            else:
                stage_pct[stage] = "pending"
    log["stage_progress"] = stage_pct
    return log


# ══════════════════════════════════════════════════════════════════════
# BLOB TRIGGER — Submit video only (fast, < 2 min)
# ══════════════════════════════════════════════════════════════════════
@app.blob_trigger(
    arg_name="myblob",
    path="videos/{name}",
    connection="STORAGE_CONNECTION_STRING",
)
def process_video_blob(myblob: func.InputStream):
    blob_name  = myblob.name.split("/")[-1]
    video_name = blob_name.rsplit(".", 1)[0]
    logger.info(f"Blob trigger: {blob_name}")

    if not blob_name.lower().endswith(".mp4"):
        return

    conn_str     = os.environ.get("STORAGE_CONNECTION_STRING", "")
    account_name = os.environ.get("STORAGE_ACCOUNT_NAME", "")
    inter_prefix = f"{video_name}/"
    blob_client  = BlobServiceClient.from_connection_string(conn_str)

    _save_log(blob_client, inter_prefix, {
        "job":               video_name,
        "blob_name":         blob_name,
        "status":            "running",
        "current_stage":     "submit",
        "started_at":        _now(),
        "updated_at":        _now(),
        "progress_pct":      0,
        "indexing_progress": "0%",
        "captions_progress": "0%",
        "content_progress":  "0%",
        "stages": {
            "submit":    "running",
            "indexing":  "pending",
            "captions":  "pending",
            "structure": "pending",
            "content":   "pending",
            "quiz":      "pending",
            "build_doc": "pending",
            "upload":    "pending",
            "cleanup":   "pending",
        },
        "error": None,
    })

    try:
        account_key = _get_account_key(conn_str)
        sas_token   = generate_blob_sas(
            account_name=account_name,
            container_name="videos",
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=6),
        )
        blob_url = f"https://{account_name}.blob.core.windows.net/videos/{blob_name}?{sas_token}"

        logger.info(f"Submitting {video_name} to Video Indexer...")
        video_id = vi.submit_video_from_blob(blob_url, video_name)

        _set_stage(blob_client, inter_prefix, "submit",   "done")
        _set_stage(blob_client, inter_prefix, "indexing", "running — 0% indexed")
        _update_log(blob_client, inter_prefix, {
            "video_indexer_id":  video_id,
            "current_stage":     "indexing",
            "indexing_progress": "0%",
        })
        logger.info(f"Submitted. video_id={video_id}. Timer will poll.")

    except Exception as e:
        logger.error(f"Submit failed: {e}", exc_info=True)
        _update_log(blob_client, inter_prefix, {
            "status":     "failed",
            "error":      str(e),
            "updated_at": _now(),
        })
        raise


# ══════════════════════════════════════════════════════════════════════
# TIMER TRIGGER — Poll + continue pipeline every 5 minutes
# ══════════════════════════════════════════════════════════════════════
@app.timer_trigger(
    arg_name="mytimer",
    schedule="0 */5 * * * *",
    run_on_startup=False,
)
def watchdog_timer(mytimer: func.TimerRequest):
    if mytimer.past_due:
        logger.warning("Timer past due.")

    conn_str = os.environ.get("STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        return

    account_name = os.environ.get("STORAGE_ACCOUNT_NAME", "")
    blob_client  = BlobServiceClient.from_connection_string(conn_str)

    running_jobs = []
    try:
        container = blob_client.get_container_client("intermediate")
        seen = set()
        for blob in container.list_blobs():
            parts = blob.name.split("/")
            if len(parts) >= 2:
                job = parts[0]
                if job not in seen:
                    seen.add(job)
                    log = _load_json(blob_client, f"{job}/", "pipeline_log.json")
                    if log and log.get("status") == "running":
                        running_jobs.append(log)
    except Exception as e:
        logger.warning(f"Could not scan jobs: {e}")
        return

    logger.info(f"Watchdog: {len(running_jobs)} running job(s)")
    for job_log in running_jobs:
        try:
            _process_job(blob_client, job_log, account_name)
        except Exception as e:
            video_name   = job_log.get("job", "unknown")
            inter_prefix = f"{video_name}/"
            logger.error(f"Job {video_name} failed: {e}", exc_info=True)
            _update_log(blob_client, inter_prefix, {
                "status":     "failed",
                "error":      str(e),
                "updated_at": _now(),
            })


def _process_job(blob_client, job_log: dict, account_name: str):
    video_name   = job_log["job"]
    inter_prefix = f"{video_name}/"
    stage        = job_log.get("current_stage", "")
    video_id     = job_log.get("video_indexer_id", "")
    logger.info(f"Processing job={video_name} stage={stage}")

    # ── Poll indexing ─────────────────────────────────────────────────
    if stage == "indexing":
        state, progress, index_data = vi.check_indexing_status(video_id)
        logger.info(f"VI state={state} progress={progress}")
        _update_log(blob_client, inter_prefix, {
            "indexing_progress": progress,
            "updated_at":        _now(),
        })
        _set_stage(blob_client, inter_prefix, "indexing", f"running — {progress}")

        if state == "Failed":
            raise RuntimeError(f"Video Indexer failed: video_id={video_id}")
        if state != "Processed":
            logger.info(f"Still indexing: {progress}")
            return

        transcript = vi.extract_transcript(index_data)
        keyframes  = vi.extract_keyframes(index_data, video_id)
        topics     = vi.extract_topics(index_data)

        # ── Download thumbnails NOW before deleting video ──────────────
        # VI thumbnails return 404 once the video is deleted
        logger.info(f"Downloading {len(keyframes)} thumbnails before deleting video...")
        for i, kf in enumerate(keyframes):
            img_bytes = oai.download_frame(kf["url"])
            if img_bytes:
                frame_blob_name = f"{inter_prefix}frames/frame_{i:03d}.jpg"
                try:
                    fb = blob_client.get_blob_client(container="intermediate", blob=frame_blob_name)
                    fb.upload_blob(img_bytes, overwrite=True)
                    keyframes[i]["url"]       = frame_blob_name
                    keyframes[i]["blob_path"] = frame_blob_name
                    logger.info(f"Saved thumbnail {i+1}/{len(keyframes)} to blob")
                except Exception as e:
                    logger.warning(f"Could not save thumbnail {i}: {e}")
            else:
                logger.warning(f"Could not download thumbnail {i}")

        # ── Now safe to delete video from VI ──────────────────────────
        vi.delete_video(video_id)

        _save_json(blob_client, inter_prefix, "transcript.json", transcript)
        _save_json(blob_client, inter_prefix, "keyframes.json",  keyframes)
        _save_json(blob_client, inter_prefix, "topics.json",     topics)
        _set_stage(blob_client, inter_prefix, "indexing", "done",
                   f"{len(transcript)} segments, {len(keyframes)} keyframes")
        _update_log(blob_client, inter_prefix, {
            "current_stage":     "captions",
            "indexing_progress": "100%",
        })
        stage = "captions"

    # ── Captions ──────────────────────────────────────────────────────
    if stage == "captions":
        _set_stage(blob_client, inter_prefix, "captions", "running")
        keyframes = _load_json(blob_client, inter_prefix, "keyframes.json") or []
        total     = len(keyframes)
        logger.info(f"Captions: {total} keyframes to process for job={video_name}")
        _update_log(blob_client, inter_prefix, {
            "keyframe_count": total,
            "updated_at": _now(),
        })
        captioned = []
        for i, kf in enumerate(keyframes):
            logger.info(f"Captioning frame {i+1}/{total} at t={kf.get('timestamp')} source={kf.get('source','?')}")

            # Thumbnails already downloaded to blob during indexing stage
            blob_path = kf.get("blob_path", "")
            if not blob_path:
                logger.warning(f"Skipping frame {i+1} — no blob_path in keyframe")
                continue

            try:
                fb        = blob_client.get_blob_client(container="intermediate", blob=blob_path)
                img_bytes = fb.download_blob().readall()
                logger.info(f"Loaded frame {i+1} from blob: {len(img_bytes)} bytes")
            except Exception as e:
                logger.warning(f"Skipping frame {i+1} — could not load from blob: {e}")
                continue

            result = oai.caption_keyframe(img_bytes, kf["timestamp"])
            logger.info(f"Frame {i+1} result: score={result.get('score')} is_useful={result.get('is_useful')} caption={result.get('caption','')[:60]}")
            result["blob_path"] = blob_path
            result["url"]       = blob_path
            captioned.append(result)
            pct = round(((i + 1) / max(total, 1)) * 100)
            _update_log(blob_client, inter_prefix, {
                "captions_progress": f"{pct}% ({i+1}/{total} frames)",
                "updated_at":        _now(),
            })
        useful = [f for f in captioned if f.get("is_useful", False)]
        # Save ALL captioned frames so doc builder has maximum choice
        _save_json(blob_client, inter_prefix, "captions.json", captioned)
        _set_stage(blob_client, inter_prefix, "captions", "done",
                   f"{len(useful)}/{len(captioned)} useful frames")
        _update_log(blob_client, inter_prefix, {
            "current_stage":    "structure",
            "captions_progress": "100%",
        })
        stage = "structure"

    # ── Structure ─────────────────────────────────────────────────────
    if stage == "structure":
        _set_stage(blob_client, inter_prefix, "structure", "running")
        transcript = _load_json(blob_client, inter_prefix, "transcript.json") or []
        captions   = _load_json(blob_client, inter_prefix, "captions.json")   or []
        topics     = _load_json(blob_client, inter_prefix, "topics.json")     or []
        structure  = oai.build_document_structure(transcript, captions, topics, video_name)
        _save_json(blob_client, inter_prefix, "structure.json", structure)
        _set_stage(blob_client, inter_prefix, "structure", "done",
                   f"{len(structure.get('sections', []))} sections")
        _update_log(blob_client, inter_prefix, {"current_stage": "content"})
        stage = "content"

    # ── Content ───────────────────────────────────────────────────────
    if stage == "content":
        _set_stage(blob_client, inter_prefix, "content", "running")
        structure = _load_json(blob_client, inter_prefix, "structure.json") or {}
        sections  = structure.get("sections", [])
        total     = len(sections)
        for i, section in enumerate(sections):
            sections[i] = oai.write_section_content(section, i + 1)
            pct = round(((i + 1) / max(total, 1)) * 100)
            _update_log(blob_client, inter_prefix, {
                "content_progress": f"{pct}% ({i+1}/{total} sections)",
                "updated_at":       _now(),
            })
        structure["sections"] = sections
        _save_json(blob_client, inter_prefix, "content.json", structure)
        _set_stage(blob_client, inter_prefix, "content", "done")
        _update_log(blob_client, inter_prefix, {
            "current_stage":   "quiz",
            "content_progress": "100%",
        })
        stage = "quiz"

    # ── Quiz ──────────────────────────────────────────────────────────
    if stage == "quiz":
        _set_stage(blob_client, inter_prefix, "quiz", "running")
        structure = _load_json(blob_client, inter_prefix, "content.json") or {}
        quiz      = oai.generate_quiz(structure)
        _save_json(blob_client, inter_prefix, "quiz.json", quiz)
        _set_stage(blob_client, inter_prefix, "quiz", "done", f"{len(quiz)} questions")
        _update_log(blob_client, inter_prefix, {"current_stage": "build_doc"})
        stage = "build_doc"

    # ── Build + Upload + Cleanup ──────────────────────────────────────
    if stage == "build_doc":
        _set_stage(blob_client, inter_prefix, "build_doc", "running")
        structure  = _load_json(blob_client, inter_prefix, "content.json")  or {}
        captions   = _load_json(blob_client, inter_prefix, "captions.json") or []
        quiz       = _load_json(blob_client, inter_prefix, "quiz.json")     or []
        doc_buffer = db.build_document(structure, captions, quiz, blob_client=blob_client)
        _set_stage(blob_client, inter_prefix, "build_doc", "done")

        _set_stage(blob_client, inter_prefix, "upload", "running")
        output_name = f"{video_name}_training_manual.docx"
        out_client  = blob_client.get_blob_client(container="outputs", blob=output_name)
        out_client.upload_blob(
            doc_buffer, overwrite=True,
            content_settings=ContentSettings(
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )
        _set_stage(blob_client, inter_prefix, "upload", "done", f"outputs/{output_name}")

        _set_stage(blob_client, inter_prefix, "cleanup", "running")
        blob_name = job_log.get("blob_name", "")
        if blob_name:
            try:
                blob_client.get_blob_client(container="videos", blob=blob_name).delete_blob()
            except Exception as e:
                logger.warning(f"Could not delete video: {e}")
        _set_stage(blob_client, inter_prefix, "cleanup", "done")

        _update_log(blob_client, inter_prefix, {
            "status":        "completed",
            "current_stage": "done",
            "output_file":   output_name,
            "completed_at":  _now(),
            "progress_pct":  100,
        })
        logger.info(f"Pipeline complete: {output_name}")


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def _now():
    return datetime.datetime.utcnow().isoformat()

def _json(data, status=200):
    return func.HttpResponse(
        json.dumps(data, indent=2),
        mimetype="application/json",
        status_code=status,
    )

def _save_json(client, prefix, filename, data):
    try:
        blob = client.get_blob_client(container="intermediate", blob=f"{prefix}{filename}")
        blob.upload_blob(json.dumps(data, ensure_ascii=False), overwrite=True)
    except Exception as e:
        logger.warning(f"Could not save {filename}: {e}")

def _load_json(client, prefix, filename):
    try:
        blob = client.get_blob_client(container="intermediate", blob=f"{prefix}{filename}")
        return json.loads(blob.download_blob().readall())
    except Exception:
        return None

def _save_log(client, prefix, data):
    _save_json(client, prefix, "pipeline_log.json", data)

def _update_log(client, prefix, updates: dict):
    try:
        log = _load_json(client, prefix, "pipeline_log.json") or {}
        log.update(updates)
        log["updated_at"] = _now()
        _save_log(client, prefix, log)
    except Exception as e:
        logger.warning(f"Could not update log: {e}")

def _set_stage(client, prefix, stage: str, status: str, detail: str = ""):
    try:
        log = _load_json(client, prefix, "pipeline_log.json") or {}
        if "stages" not in log:
            log["stages"] = {}
        log["stages"][stage] = status + (f" — {detail}" if detail else "")
        log["current_stage"] = stage
        log["updated_at"]    = _now()
        _save_log(client, prefix, log)
    except Exception as e:
        logger.warning(f"Could not set stage {stage}: {e}")

def _get_account_key(conn_str):
    parts = dict(p.split("=", 1) for p in conn_str.split(";") if "=" in p)
    return parts.get("AccountKey", "")
