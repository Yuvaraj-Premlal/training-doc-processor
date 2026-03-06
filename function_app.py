import os
import json
import logging
import datetime
import io

import azure.functions as func
from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
    ContentSettings,
)

import video_indexer as vi
import openai_client as oai
import document_builder as db
import notifier


logger = logging.getLogger("training_doc_processor")

app = func.FunctionApp()


# ----------------------------------------------------------
# BLOB TRIGGER
# ----------------------------------------------------------

@app.blob_trigger(
    arg_name="myblob",
    path="videos/{name}",
    connection="AzureWebJobsStorage",
)
def process_video_blob(myblob: func.InputStream):

    blob_name = myblob.name.split("/")[-1]
    video_name = blob_name.rsplit(".", 1)[0]

    logger.info(f"Blob trigger fired: {blob_name} ({myblob.length} bytes)")

    if not blob_name.lower().endswith(".mp4"):
        logger.info(f"Skipping non-MP4: {blob_name}")
        return

    conn_str = os.environ.get("AzureWebJobsStorage")
    account_name = os.environ.get("STORAGE_ACCOUNT_NAME")

    if not conn_str or not account_name:
        logger.error("Missing required storage environment variables.")
        return

    blob_client = BlobServiceClient.from_connection_string(conn_str)
    inter_prefix = f"{video_name}/"

    try:

        # ----------------------------------------------------------
        # Step 1 - Generate SAS URL
        # ----------------------------------------------------------

        logger.info("Step 1: Generating SAS URL")

        account_key = _get_account_key(conn_str)

        sas_token = generate_blob_sas(
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

        # ----------------------------------------------------------
        # Step 2 - Video Indexer
        # ----------------------------------------------------------

        transcript_cache = _load_json(blob_client, inter_prefix, "transcript.json")
        keyframes_cache = _load_json(blob_client, inter_prefix, "keyframes.json")
        topics_cache = _load_json(blob_client, inter_prefix, "topics.json")

        if transcript_cache and keyframes_cache:
            logger.info("Using cached transcript + keyframes")
            transcript_segments = transcript_cache
            keyframes = keyframes_cache
            topics = topics_cache or []

        else:
            logger.info("Submitting video to Video Indexer")

            video_id = vi.submit_video_from_blob(blob_url, video_name)

            logger.info("Waiting for indexing to complete")

            index_data = vi.wait_for_indexing(video_id)

            transcript_segments = vi.extract_transcript(index_data)
            keyframes = vi.extract_keyframes(index_data, video_id)
            topics = vi.extract_topics(index_data)

            _save_json(blob_client, inter_prefix, "transcript.json", transcript_segments)
            _save_json(blob_client, inter_prefix, "keyframes.json", keyframes)
            _save_json(blob_client, inter_prefix, "topics.json", topics)

            vi.delete_video(video_id)

        # ----------------------------------------------------------
        # Step 3 - Caption frames
        # ----------------------------------------------------------

        captions_cache = _load_json(blob_client, inter_prefix, "captions.json")

        if captions_cache:
            captioned_frames = captions_cache
        else:
            logger.info("Captioning keyframes")
            captioned_frames = oai.caption_all_keyframes(keyframes)
            _save_json(blob_client, inter_prefix, "captions.json", captioned_frames)

        # ----------------------------------------------------------
        # Step 4 - Build structure
        # ----------------------------------------------------------

        structure_cache = _load_json(blob_client, inter_prefix, "structure.json")

        if structure_cache:
            structure = structure_cache
        else:
            logger.info("Building document structure")

            structure = oai.build_document_structure(
                transcript_segments,
                captioned_frames,
                topics,
                video_name,
            )

            _save_json(blob_client, inter_prefix, "structure.json", structure)

        # ----------------------------------------------------------
        # Step 5 - Write sections
        # ----------------------------------------------------------

        content_cache = _load_json(blob_client, inter_prefix, "content.json")

        if content_cache:
            structure = content_cache
        else:
            logger.info("Writing section content")

            structure = oai.write_all_sections(structure)

            _save_json(blob_client, inter_prefix, "content.json", structure)

        # ----------------------------------------------------------
        # Step 6 - Quiz
        # ----------------------------------------------------------

        logger.info("Generating quiz")
        quiz = oai.generate_quiz(structure)

        # ----------------------------------------------------------
        # Step 7 - Build DOCX
        # ----------------------------------------------------------

        logger.info("Building document")

        doc_buffer = db.build_document(structure, captioned_frames, quiz)

        if isinstance(doc_buffer, bytes):
            doc_buffer = io.BytesIO(doc_buffer)

        # ----------------------------------------------------------
        # Step 8 - Upload DOCX
        # ----------------------------------------------------------

        logger.info("Uploading document")

        output_name = f"{video_name}_training_manual.docx"

        out_client = blob_client.get_blob_client(
            container="outputs",
            blob=output_name,
        )

        out_client.upload_blob(
            doc_buffer,
            overwrite=True,
            content_settings=ContentSettings(
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )

        # ----------------------------------------------------------
        # Step 9 - Delete source video
        # ----------------------------------------------------------

        logger.info("Deleting original video")

        blob_client.get_blob_client(
            container="videos",
            blob=blob_name,
        ).delete_blob()

        # ----------------------------------------------------------
        # Step 10 - Notify
        # ----------------------------------------------------------

        notifier.send_completion_email(
            video_name=video_name,
            document_name=output_name,
            section_count=len(structure.get("sections", [])),
            blob_url=f"https://{account_name}.blob.core.windows.net/outputs/{output_name}",
        )

        logger.info(f"Pipeline complete for {video_name}")

    except Exception as e:

        logger.error(f"Pipeline failed for {blob_name}: {e}", exc_info=True)

        notifier.send_failure_email(
            video_name=video_name,
            error_message=str(e),
        )

        raise


# ----------------------------------------------------------
# TIMER TRIGGER
# ----------------------------------------------------------

@app.timer_trigger(
    arg_name="mytimer",
    schedule="0 */30 * * * *",
    run_on_startup=False,
)
def watchdog_timer(mytimer: func.TimerRequest):

    if mytimer.past_due:
        logger.warning("Watchdog timer running late")

    conn_str = os.environ.get("AzureWebJobsStorage")

    if not conn_str:
        logger.warning("AzureWebJobsStorage not configured")
        return

    try:

        blob_client = BlobServiceClient.from_connection_string(conn_str)

        container = blob_client.get_container_client("videos")

        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

        stuck = []

        for blob in container.list_blobs():

            if not blob.name.lower().endswith(".mp4"):
                continue

            age = now - blob.last_modified

            if age.total_seconds() > 7200:
                stuck.append(f"{blob.name} (age: {age})")

        if stuck:

            logger.warning(f"Stuck videos detected: {stuck}")

            notifier.send_failure_email(
                video_name="Watchdog",
                error_message="Videos stuck over 2 hours:\n" + "\n".join(stuck),
            )

        else:
            logger.info("Watchdog check passed")

    except Exception as e:
        logger.error(f"Watchdog error: {e}", exc_info=True)


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def _save_json(client, prefix, filename, data):

    try:

        blob = client.get_blob_client(
            container="intermediate",
            blob=f"{prefix}{filename}",
        )

        blob.upload_blob(json.dumps(data, ensure_ascii=False), overwrite=True)

    except Exception as e:

        logger.warning(f"Could not save {filename}: {e}")


def _load_json(client, prefix, filename):

    try:

        blob = client.get_blob_client(
            container="intermediate",
            blob=f"{prefix}{filename}",
        )

        return json.loads(blob.download_blob().readall())

    except Exception:
        return None


def _get_account_key(conn_str):

    parts = dict(
        p.split("=", 1)
        for p in conn_str.split(";")
        if "=" in p
    )

    return parts.get("AccountKey", "")
