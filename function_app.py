"""
function_app.py
Main Azure Function App.

Functions:
  - process_video_blob : BlobTrigger — fires when MP4 lands in /videos container
  - watchdog_timer     : TimerTrigger — every 30 min, checks for stuck jobs

Pipeline:
  1. Generate SAS URL for the blob
  2. Submit to Video Indexer
  3. Poll until indexed
  4. Extract transcript, keyframes, topics
  5. Caption keyframes with GPT-4o Vision
  6. Build document structure with GPT-4o
  7. Write section content with GPT-4o
  8. Generate quiz with GPT-4o
  9. Build .docx with python-docx
  10. Upload .docx to /outputs container
  11. Delete original video from /videos
  12. Send email notification
"""

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
)

import video_indexer   as vi
import openai_client   as oai
import document_builder as db
import notifier

logger = logging.getLogger(__name__)
app    = func.FunctionApp()

# ── Config ────────────────────────────────────────────────────────────────────
CONN_STR      = os.environ["STORAGE_CONNECTION_STRING"]
ACCOUNT_NAME  = os.environ["STORAGE_ACCOUNT_NAME"]
VIDEOS_CONT   = "videos"
OUTPUTS_CONT  = "outputs"
INTER_CONT    = "intermediate"
KEYFRAMES_CONT= "keyframes"


# ══════════════════════════════════════════════════════════════════════════════
# BLOB TRIGGER — fires when an MP4 lands in the /videos container
# ══════════════════════════════════════════════════════════════════════════════

@app.blob_trigger(
    arg_name="myblob",
    path=f"{VIDEOS_CONT}/{{name}}",
    connection="STORAGE_CONNECTION_STRING",
)
def process_video_blob(myblob: func.InputStream):
    blob_name = myblob.name.split("/")[-1]
    logger.info(f"▶ Blob trigger fired: {blob_name} ({myblob.length} bytes)")

    # Only process MP4 files
    if not blob_name.lower().endswith(".mp4"):
        logger.info(f"  Skipping non-MP4 file: {blob_name}")
        return

    video_name    = blob_name.rsplit(".", 1)[0]
    inter_prefix  = f"{video_name}/"
    blob_client   = BlobServiceClient.from_connection_string(CONN_STR)

    try:
        # ── Step 1: Generate SAS URL ──────────────────────────────────────────
        logger.info("Step 1: Generating SAS URL...")
        sas_token = generate_blob_sas(
            account_name=ACCOUNT_NAME,
            container_name=VIDEOS_CONT,
            blob_name=blob_name,
            account_key=_get_account_key(blob_client),
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=6),
        )
        blob_url = (
            f"https://{ACCOUNT_NAME}.blob.core.windows.net"
            f"/{VIDEOS_CONT}/{blob_name}?{sas_token}"
        )
        logger.info(f"  SAS URL generated (expires in 6h).")

        # ── Step 2: Submit to Video Indexer ───────────────────────────────────
        logger.info("Step 2: Submitting to Video Indexer...")
        transcript_cache = _load_intermediate(blob_client, inter_prefix, "transcript.json")
        keyframes_cache  = _load_intermediate(blob_client, inter_prefix, "keyframes.json")

        if transcript_cache and keyframes_cache:
            logger.info("  Transcript + keyframes found in cache — skipping Video Indexer.")
            transcript_segments = transcript_cache
            keyframes            = keyframes_cache
        else:
            video_id    = vi.submit_video_from_blob(blob_url, video_name)
            logger.info(f"  Submitted. video_id={video_id}")

            # ── Step 3: Wait for indexing ─────────────────────────────────────
            logger.info("Step 3: Waiting for Video Indexer to finish...")
            index_data = vi.wait_for_indexing(video_id)

            # ── Step 4: Extract data ──────────────────────────────────────────
            logger.info("Step 4: Extracting transcript, keyframes, topics...")
            transcript_segments = vi.extract_transcript(index_data)
            keyframes            = vi.extract_keyframes(index_data, video_id)
            topics               = vi.extract_topics(index_data)

            _save_intermediate(blob_client, inter_prefix, "transcript.json", transcript_segments)
            _save_intermediate(blob_client, inter_prefix, "keyframes.json",  keyframes)
            _save_intermediate(blob_client, inter_prefix, "topics.json",     topics)
            logger.info(f"  Transcript: {len(transcript_segments)} segments")
            logger.info(f"  Keyframes:  {len(keyframes)} frames")
            logger.info(f"  Topics:     {len(topics)} topics")

            # Clean up from Video Indexer (not from blob)
            vi.delete_video(video_id)
        
        topics = _load_intermediate(blob_client, inter_prefix, "topics.json") or []

        # ── Step 5: Caption keyframes ─────────────────────────────────────────
        logger.info("Step 5: Captioning keyframes with GPT-4o Vision...")
        captions_cache = _load_intermediate(blob_client, inter_prefix, "captions.json")
        if captions_cache:
            logger.info("  Captions found in cache.")
            captioned_frames = captions_cache
        else:
            captioned_frames = oai.caption_all_keyframes(keyframes)
            _save_intermediate(blob_client, inter_prefix, "captions.json", captioned_frames)
        logger.info(f"  {len(captioned_frames)} useful frames.")

        # ── Step 6: Build document structure ──────────────────────────────────
        logger.info("Step 6: Building document structure with GPT-4o...")
        structure_cache = _load_intermediate(blob_client, inter_prefix, "structure.json")
        if structure_cache:
            logger.info("  Structure found in cache.")
            structure = structure_cache
        else:
            structure = oai.build_document_structure(
                transcript_segments, captioned_frames, topics, video_name
            )
            _save_intermediate(blob_client, inter_prefix, "structure.json", structure)
        logger.info(f"  {len(structure.get('sections', []))} sections planned.")

        # ── Step 7: Write section content ─────────────────────────────────────
        logger.info("Step 7: Writing training content per section...")
        content_cache = _load_intermediate(blob_client, inter_prefix, "content.json")
        if content_cache:
            logger.info("  Content found in cache.")
            structure = content_cache
        else:
            structure = oai.write_all_sections(structure)
            _save_intermediate(blob_client, inter_prefix, "content.json", structure)

        # ── Step 8: Generate quiz ──────────────────────────────────────────────
        logger.info("Step 8: Generating quiz...")
        quiz = oai.generate_quiz(structure)
        logger.info(f"  {len(quiz)} quiz questions generated.")

        # ── Step 9: Build .docx ────────────────────────────────────────────────
        logger.info("Step 9: Building .docx document...")
        doc_buffer = db.build_document(structure, captioned_frames, quiz)

        # ── Step 10: Upload .docx to /outputs ─────────────────────────────────
        logger.info("Step 10: Uploading .docx to outputs container...")
        output_name = f"{video_name}_training_manual.docx"
        out_client  = blob_client.get_blob_client(
            container=OUTPUTS_CONT, blob=output_name
        )
        out_client.upload_blob(
            doc_buffer,
            overwrite=True,
            content_settings=_docx_content_settings(),
        )
        logger.info(f"  Uploaded: {output_name}")

        # Build download URL
        out_sas = generate_blob_sas(
            account_name=ACCOUNT_NAME,
            container_name=OUTPUTS_CONT,
            blob_name=output_name,
            account_key=_get_account_key(blob_client),
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(days=7),
        )
        download_url = (
            f"https://{ACCOUNT_NAME}.blob.core.windows.net"
            f"/{OUTPUTS_CONT}/{output_name}?{out_sas}"
        )

        # ── Step 11: Delete original video ────────────────────────────────────
        logger.info("Step 11: Deleting original video from blob storage...")
        vid_client = blob_client.get_blob_client(
            container=VIDEOS_CONT, blob=blob_name
        )
        vid_client.delete_blob()
        logger.info(f"  Deleted {blob_name} from /{VIDEOS_CONT}.")

        # ── Step 12: Send email notification ──────────────────────────────────
        logger.info("Step 12: Sending email notification...")
        notifier.send_completion_email(
            video_name=video_name,
            document_name=output_name,
            section_count=len(structure.get("sections", [])),
            blob_url=download_url,
        )

        logger.info(f"✅ Pipeline complete for '{video_name}'. Document: {output_name}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed for '{blob_name}': {e}", exc_info=True)
        notifier.send_failure_email(video_name=video_name, error_message=str(e))
        raise


# ══════════════════════════════════════════════════════════════════════════════
# TIMER TRIGGER — watchdog, runs every 30 minutes
# ══════════════════════════════════════════════════════════════════════════════

@app.timer_trigger(
    arg_name="mytimer",
    schedule="0 */30 * * * *",
    run_on_startup=False,
)
def watchdog_timer(mytimer: func.TimerRequest):
    """
    Checks for MP4 files stuck in /videos container older than 2 hours.
    Sends an alert email if any are found.
    """
    if mytimer.past_due:
        logger.warning("Watchdog timer is running late.")

    try:
        blob_client = BlobServiceClient.from_connection_string(CONN_STR)
        container   = blob_client.get_container_client(VIDEOS_CONT)
        now         = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        stuck       = []

        for blob in container.list_blobs():
            if not blob.name.lower().endswith(".mp4"):
                continue
            age = now - blob.last_modified
            if age.total_seconds() > 7200:  # 2 hours
                stuck.append(f"{blob.name} (age: {age})")

        if stuck:
            logger.warning(f"Watchdog found {len(stuck)} stuck video(s): {stuck}")
            notifier.send_failure_email(
                video_name="Watchdog Alert",
                error_message=(
                    f"The following videos have been in /videos for over 2 hours "
                    f"without producing a document:\n" + "\n".join(stuck)
                ),
            )
        else:
            logger.info("Watchdog: No stuck videos found.")

    except Exception as e:
        logger.error(f"Watchdog error: {e}", exc_info=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_intermediate(
    client: BlobServiceClient, prefix: str, filename: str, data: object
):
    """Save a JSON checkpoint to the /intermediate container."""
    try:
        blob = client.get_blob_client(
            container=INTER_CONT, blob=f"{prefix}{filename}"
        )
        blob.upload_blob(json.dumps(data, ensure_ascii=False), overwrite=True)
    except Exception as e:
        logger.warning(f"Could not save intermediate {filename}: {e}")


def _load_intermediate(
    client: BlobServiceClient, prefix: str, filename: str
) -> object | None:
    """Load a JSON checkpoint from the /intermediate container, or None if missing."""
    try:
        blob = client.get_blob_client(
            container=INTER_CONT, blob=f"{prefix}{filename}"
        )
        data = blob.download_blob().readall()
        return json.loads(data)
    except Exception:
        return None


def _get_account_key(client: BlobServiceClient) -> str:
    """Extract storage account key from the connection string."""
    parts = dict(
        part.split("=", 1)
        for part in CONN_STR.split(";")
        if "=" in part
    )
    return parts.get("AccountKey", "")


def _docx_content_settings():
    """Return content settings for .docx blob upload."""
    from azure.storage.blob import ContentSettings
    return ContentSettings(
        content_type=(
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
    )
