import os
import io
import json
import logging
import datetime

import azure.functions as func
from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
    ContentSettings,
)

import video_indexer    as vi
import openai_client    as oai
import document_builder as db
import notifier

logger = logging.getLogger("training_doc_processor")
app    = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# ══════════════════════════════════════════════════════════════════════════════
# HTTP TRIGGER — Health check + pipeline status dashboard
# GET  /api/status        → shows all jobs + their pipeline stage
# GET  /api/status?job=X  → shows detailed log for one job
# ══════════════════════════════════════════════════════════════════════════════

@app.route(route="status", methods=["GET"])
def status(req: func.HttpRequest) -> func.HttpResponse:
    try:
        conn_str = os.environ.get("STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            return func.HttpResponse(
                json.dumps({"error": "STORAGE_CONNECTION_STRING not set"}),
                mimetype="application/json", status_code=500
            )

        blob_client = BlobServiceClient.from_connection_string(conn_str)
        job_name    = req.params.get("job")

        # ── Single job detail ──────────────────────────────────────────────
        if job_name:
            log_data = _load_json(blob_client, f"{job_name}/", "pipeline_log.json")
            if not log_data:
                return func.HttpResponse(
                    json.dumps({"error": f"No job found: {job_name}"}),
                    mimetype="application/json", status_code=404
                )
            return func.HttpResponse(
                json.dumps(log_data, indent=2),
                mimetype="application/json"
            )

        # ── All jobs summary ───────────────────────────────────────────────
        jobs = []
        try:
            container = blob_client.get_container_client("intermediate")
            seen      = set()
            for blob in container.list_blobs():
                parts = blob.name.split("/")
                if len(parts) >= 2:
                    job = parts[0]
                    if job not in seen:
                        seen.add(job)
                        log_data = _load_json(blob_client, f"{job}/", "pipeline_log.json")
                        if log_data:
                            jobs.append({
                                "job":     job,
                                "status":  log_data.get("status", "unknown"),
                                "stage":   log_data.get("current_stage", ""),
                                "started": log_data.get("started_at", ""),
                                "updated": log_data.get("updated_at", ""),
                                "error":   log_data.get("error", None),
                            })
        except Exception as e:
            logger.warning(f"Could not list jobs: {e}")

        # Also check outputs for completed jobs
        completed = []
        try:
            out_container = blob_client.get_container_client("outputs")
            for blob in out_container.list_blobs():
                if blob.name.endswith(".docx"):
                    completed.append({
                        "file":     blob.name,
                        "size_mb":  round(blob.size / 1024 / 1024, 2),
                        "created":  blob.last_modified.isoformat() if blob.last_modified else "",
                    })
        except Exception as e:
            logger.warning(f"Could not list outputs: {e}")

        result = {
            "function_app": "training-doc-processor",
            "status":        "running",
            "active_jobs":   jobs,
            "completed_docs": completed,
            "timestamp":     datetime.datetime.utcnow().isoformat(),
        }
        return func.HttpResponse(
            json.dumps(result, indent=2),
            mimetype="application/json"
        )

    except Exception as e:
        logger.error(f"Status endpoint error: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json", status_code=500
        )


# ══════════════════════════════════════════════════════════════════════════════
# BLOB TRIGGER — fires when MP4 lands in /videos container
# ══════════════════════════════════════════════════════════════════════════════

@app.blob_trigger(
    arg_name="myblob",
    path="videos/{name}",
    connection="STORAGE_CONNECTION_STRING",
)
def process_video_blob(myblob: func.InputStream):
    blob_name  = myblob.name.split("/")[-1]
    video_name = blob_name.rsplit(".", 1)[0]
    logger.info(f"Blob trigger fired: {blob_name} ({myblob.length} bytes)")

    if not blob_name.lower().endswith(".mp4"):
        logger.info(f"Skipping non-MP4: {blob_name}")
        return

    conn_str     = os.environ.get("STORAGE_CONNECTION_STRING", "")
    account_name = os.environ.get("STORAGE_ACCOUNT_NAME", "")
    inter_prefix = f"{video_name}/"
    blob_client  = BlobServiceClient.from_connection_string(conn_str)

    # Initialise pipeline log
    _update_log(blob_client, inter_prefix, {
        "job":           video_name,
        "status":        "running",
        "current_stage": "started",
        "started_at":    datetime.datetime.utcnow().isoformat(),
        "updated_at":    datetime.datetime.utcnow().isoformat(),
        "stages": {
            "sas_url":       "pending",
            "video_indexer": "pending",
            "captions":      "pending",
            "structure":     "pending",
            "content":       "pending",
            "quiz":          "pending",
            "build_doc":     "pending",
            "upload":        "pending",
            "cleanup":       "pending",
        },
        "error": None,
    })

    try:
        # ── Step 1: SAS URL ───────────────────────────────────────────────
        _set_stage(blob_client, inter_prefix, "sas_url", "running")
        logger.info("Step 1: Generating SAS URL")
        account_key = _get_account_key(conn_str)
        sas_token   = generate_blob_sas(
            account_name=account_name,
            container_name="videos",
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=6),
        )
        blob_url = (
            f"https://{account_name}.blob.core.windows.net"
            f"/videos/{blob_name}?{sas_token}"
        )
        _set_stage(blob_client, inter_prefix, "sas_url", "done")

        # ── Step 2+3: Video Indexer ───────────────────────────────────────
        transcript_cache = _load_json(blob_client, inter_prefix, "transcript.json")
        keyframes_cache  = _load_json(blob_client, inter_prefix, "keyframes.json")
        topics_cache     = _load_json(blob_client, inter_prefix, "topics.json")

        if transcript_cache and keyframes_cache:
            logger.info("Using cached transcript + keyframes")
            transcript_segments = transcript_cache
            keyframes           = keyframes_cache
            topics              = topics_cache or []
            _set_stage(blob_client, inter_prefix, "video_indexer", "done (cached)")
        else:
            _set_stage(blob_client, inter_prefix, "video_indexer", "running")
            logger.info("Step 2: Submitting to Video Indexer")
            video_id = vi.submit_video_from_blob(blob_url, video_name)
            _update_log(blob_client, inter_prefix, {"video_indexer_id": video_id})
            logger.info(f"Step 3: Waiting for indexing (video_id={video_id})")
            index_data          = vi.wait_for_indexing(video_id)
            transcript_segments = vi.extract_transcript(index_data)
            keyframes           = vi.extract_keyframes(index_data, video_id)
            topics              = vi.extract_topics(index_data)
            _save_json(blob_client, inter_prefix, "transcript.json", transcript_segments)
            _save_json(blob_client, inter_prefix, "keyframes.json",  keyframes)
            _save_json(blob_client, inter_prefix, "topics.json",     topics)
            vi.delete_video(video_id)
            _set_stage(blob_client, inter_prefix, "video_indexer", "done",
                       detail=f"{len(transcript_segments)} transcript segments, {len(keyframes)} keyframes")

        # ── Step 4: Caption keyframes ─────────────────────────────────────
        captions_cache = _load_json(blob_client, inter_prefix, "captions.json")
        if captions_cache:
            captioned_frames = captions_cache
            _set_stage(blob_client, inter_prefix, "captions", "done (cached)")
        else:
            _set_stage(blob_client, inter_prefix, "captions", "running")
            logger.info("Step 4: Captioning keyframes")
            captioned_frames = oai.caption_all_keyframes(keyframes)
            _save_json(blob_client, inter_prefix, "captions.json", captioned_frames)
            _set_stage(blob_client, inter_prefix, "captions", "done",
                       detail=f"{len(captioned_frames)} useful frames")

        # ── Step 5: Document structure ────────────────────────────────────
        structure_cache = _load_json(blob_client, inter_prefix, "structure.json")
        if structure_cache:
            structure = structure_cache
            _set_stage(blob_client, inter_prefix, "structure", "done (cached)")
        else:
            _set_stage(blob_client, inter_prefix, "structure", "running")
            logger.info("Step 5: Building document structure")
            structure = oai.build_document_structure(
                transcript_segments, captioned_frames, topics, video_name
            )
            _save_json(blob_client, inter_prefix, "structure.json", structure)
            _set_stage(blob_client, inter_prefix, "structure", "done",
                       detail=f"{len(structure.get('sections', []))} sections")

        # ── Step 6: Write content ─────────────────────────────────────────
        content_cache = _load_json(blob_client, inter_prefix, "content.json")
        if content_cache:
            structure = content_cache
            _set_stage(blob_client, inter_prefix, "content", "done (cached)")
        else:
            _set_stage(blob_client, inter_prefix, "content", "running")
            logger.info("Step 6: Writing section content")
            structure = oai.write_all_sections(structure)
            _save_json(blob_client, inter_prefix, "content.json", structure)
            _set_stage(blob_client, inter_prefix, "content", "done")

        # ── Step 7: Quiz ──────────────────────────────────────────────────
        _set_stage(blob_client, inter_prefix, "quiz", "running")
        logger.info("Step 7: Generating quiz")
        quiz = oai.generate_quiz(structure)
        _set_stage(blob_client, inter_prefix, "quiz", "done",
                   detail=f"{len(quiz)} questions")

        # ── Step 8: Build docx ────────────────────────────────────────────
        _set_stage(blob_client, inter_prefix, "build_doc", "running")
        logger.info("Step 8: Building document")
        doc_buffer = db.build_document(structure, captioned_frames, quiz)
        _set_stage(blob_client, inter_prefix, "build_doc", "done")

        # ── Step 9: Upload ────────────────────────────────────────────────
        _set_stage(blob_client, inter_prefix, "upload", "running")
        logger.info("Step 9: Uploading to outputs")
        output_name = f"{video_name}_training_manual.docx"
        out_client  = blob_client.get_blob_client(container="outputs", blob=output_name)
        out_client.upload_blob(
            doc_buffer, overwrite=True,
            content_settings=ContentSettings(
                content_type=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                )
            ),
        )
        _set_stage(blob_client, inter_prefix, "upload", "done",
                   detail=f"outputs/{output_name}")

        # ── Step 10: Cleanup ──────────────────────────────────────────────
        _set_stage(blob_client, inter_prefix, "cleanup", "running")
        logger.info("Step 10: Deleting original video")
        blob_client.get_blob_client(container="videos", blob=blob_name).delete_blob()
        _set_stage(blob_client, inter_prefix, "cleanup", "done")

        # ── Mark complete ─────────────────────────────────────────────────
        _update_log(blob_client, inter_prefix, {
            "status":        "completed",
            "current_stage": "done",
            "output_file":   output_name,
            "completed_at":  datetime.datetime.utcnow().isoformat(),
            "updated_at":    datetime.datetime.utcnow().isoformat(),
        })
        logger.info(f"Pipeline complete. Document: {output_name}")

        notifier.send_completion_email(
            video_name=video_name,
            document_name=output_name,
            section_count=len(structure.get("sections", [])),
            blob_url=f"https://{account_name}.blob.core.windows.net/outputs/{output_name}",
        )

    except Exception as e:
        logger.error(f"Pipeline failed for {blob_name}: {e}", exc_info=True)
        _update_log(blob_client, inter_prefix, {
            "status":     "failed",
            "error":      str(e),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        })
        notifier.send_failure_email(video_name=video_name, error_message=str(e))
        raise


# ══════════════════════════════════════════════════════════════════════════════
# TIMER TRIGGER — watchdog every 30 minutes
# ══════════════════════════════════════════════════════════════════════════════

@app.timer_trigger(
    arg_name="mytimer",
    schedule="0 */30 * * * *",
    run_on_startup=False,
)
def watchdog_timer(mytimer: func.TimerRequest):
    if mytimer.past_due:
        logger.warning("Watchdog timer is past due.")
    conn_str = os.environ.get("STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        return
    try:
        blob_client = BlobServiceClient.from_connection_string(conn_str)
        container   = blob_client.get_container_client("videos")
        now         = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        stuck       = []
        for blob in container.list_blobs():
            if not blob.name.lower().endswith(".mp4"):
                continue
            age = now - blob.last_modified
            if age.total_seconds() > 7200:
                stuck.append(f"{blob.name} (age: {age})")
        if stuck:
            logger.warning(f"Stuck videos: {stuck}")
            notifier.send_failure_email(
                video_name="Watchdog",
                error_message="Videos stuck over 2 hours:\n" + "\n".join(stuck),
            )
        else:
            logger.info("Watchdog: No stuck videos.")
    except Exception as e:
        logger.error(f"Watchdog error: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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


def _update_log(client, prefix, updates: dict):
    try:
        existing = _load_json(client, prefix, "pipeline_log.json") or {}
        existing.update(updates)
        existing["updated_at"] = datetime.datetime.utcnow().isoformat()
        _save_json(client, prefix, "pipeline_log.json", existing)
    except Exception as e:
        logger.warning(f"Could not update log: {e}")


def _set_stage(client, prefix, stage: str, status: str, detail: str = ""):
    try:
        log = _load_json(client, prefix, "pipeline_log.json") or {}
        if "stages" not in log:
            log["stages"] = {}
        log["stages"][stage] = status + (f" — {detail}" if detail else "")
        log["current_stage"] = stage
        log["updated_at"]    = datetime.datetime.utcnow().isoformat()
        _save_json(client, prefix, "pipeline_log.json", log)
    except Exception as e:
        logger.warning(f"Could not set stage {stage}: {e}")


def _get_account_key(conn_str):
    parts = dict(p.split("=", 1) for p in conn_str.split(";") if "=" in p)
    return parts.get("AccountKey", "")
