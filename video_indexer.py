"""
video_indexer.py
Handles all interactions with Azure Video Indexer ARM-based API.
Uploads video, polls for completion, fetches transcript and keyframes.
"""

import os
import time
import logging
import requests
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

ARM_BASE = "https://management.azure.com"
VI_BASE  = "https://api.videoindexer.ai"


def _load_config():
    """Safely load environment configuration."""
    config = {
        "ACCOUNT_ID": os.getenv("VIDEO_INDEXER_ACCOUNT_ID"),
        "LOCATION": os.getenv("VIDEO_INDEXER_LOCATION"),
        "SUBSCRIPTION_ID": os.getenv("SUBSCRIPTION_ID"),
        "RESOURCE_GROUP": os.getenv("RESOURCE_GROUP", "training-doc-generator"),
        "VI_ACCOUNT_NAME": os.getenv("VI_ACCOUNT_NAME", "training-doc-ai"),
    }

    missing = [k for k, v in config.items() if not v and k not in ("RESOURCE_GROUP", "VI_ACCOUNT_NAME")]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    return config


def _get_arm_token() -> str:
    """Get ARM access token using managed identity."""
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default")
    return token.token


def _get_vi_access_token(arm_token: str) -> str:
    """Exchange ARM token for a Video Indexer account access token."""
    cfg = _load_config()

    url = (
        f"{ARM_BASE}/subscriptions/{cfg['SUBSCRIPTION_ID']}"
        f"/resourceGroups/{cfg['RESOURCE_GROUP']}"
        f"/providers/Microsoft.VideoIndexer/accounts/{cfg['VI_ACCOUNT_NAME']}"
        f"/generateAccessToken?api-version=2024-01-01"
    )

    body = {"permissionType": "Contributor", "scope": "Account"}

    headers = {
        "Authorization": f"Bearer {arm_token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["accessToken"]


def _get_tokens():
    arm_token = _get_arm_token()
    vi_token  = _get_vi_access_token(arm_token)
    return vi_token


def submit_video_from_blob(blob_url: str, video_name: str) -> str:
    """Submit video from blob SAS URL to Video Indexer."""
    cfg = _load_config()
    vi_token = _get_tokens()

    url = f"{VI_BASE}/{cfg['LOCATION']}/Accounts/{cfg['ACCOUNT_ID']}/Videos"

    params = {
        "accessToken": vi_token,
        "name": video_name,
        "videoUrl": blob_url,
        "language": "auto",
        "indexingPreset": "Default",
        "streamingPreset": "NoStreaming",
    }

    resp = requests.post(url, params=params, timeout=60)
    resp.raise_for_status()

    video_id = resp.json()["id"]
    logger.info(f"Video submitted to Video Indexer. video_id={video_id}")
    return video_id


def wait_for_indexing(video_id: str, poll_interval: int = 30, max_wait: int = 3600) -> dict:
    """Poll Video Indexer until processing completes."""
    cfg = _load_config()
    vi_token = _get_tokens()

    url = f"{VI_BASE}/{cfg['LOCATION']}/Accounts/{cfg['ACCOUNT_ID']}/Videos/{video_id}/Index"
    elapsed = 0

    while elapsed < max_wait:
        params = {"accessToken": vi_token, "language": "English"}

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()

        state = data.get("state", "")
        progress = data.get("videos", [{}])[0].get("processingProgress", "0%")

        logger.info(f"Video Indexer state={state} progress={progress}")

        if state == "Processed":
            logger.info("Video Indexer processing complete.")
            return data

        if state == "Failed":
            raise RuntimeError(f"Video Indexer failed for video_id={video_id}")

        if elapsed % 600 == 0 and elapsed > 0:
            vi_token = _get_tokens()

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Video Indexer did not finish within {max_wait}s")


def extract_transcript(index_data: dict) -> list:
    """Extract transcript segments."""
    segments = []

    try:
        transcript_items = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("transcript", [])
        )

        for item in transcript_items:
            text = item.get("text", "").strip()
            if not text:
                continue

            inst = item.get("instances", [{}])[0]

            segments.append({
                "text": text,
                "start": _parse_time(inst.get("start", "0:0:0")),
                "end": _parse_time(inst.get("end", "0:0:0")),
            })

    except Exception as e:
        logger.warning(f"Error extracting transcript: {e}")

    return segments


def extract_keyframes(index_data: dict, video_id: str) -> list:
    """Extract keyframe metadata."""
    cfg = _load_config()
    vi_token = _get_tokens()

    keyframes = []

    try:
        shots = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("shots", [])
        )

        for shot in shots:
            for kf in shot.get("keyFrames", []):

                inst = kf.get("instances", [{}])[0]
                thumbnail_id = inst.get("thumbnailId")

                if not thumbnail_id:
                    continue

                timestamp = _parse_time(inst.get("adjustedStart", "0:0:0"))

                thumb_url = (
                    f"{VI_BASE}/{cfg['LOCATION']}/Accounts/{cfg['ACCOUNT_ID']}"
                    f"/Videos/{video_id}/Thumbnails/{thumbnail_id}"
                    f"?accessToken={vi_token}&format=Jpeg"
                )

                keyframes.append({
                    "timestamp": timestamp,
                    "thumbnail_id": thumbnail_id,
                    "url": thumb_url,
                })

    except Exception as e:
        logger.warning(f"Error extracting keyframes: {e}")

    logger.info(f"Extracted {len(keyframes)} keyframes.")
    return keyframes


def extract_topics(index_data: dict) -> list:
    """Extract high-level topics."""
    topics = []

    try:
        raw_topics = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("topics", [])
        )

        for t in raw_topics:
            inst = t.get("instances", [{}])[0]

            topics.append({
                "name": t.get("name", ""),
                "start": _parse_time(inst.get("adjustedStart", "0:0:0")),
                "end": _parse_time(inst.get("adjustedEnd", "0:0:0")),
            })

    except Exception as e:
        logger.warning(f"Error extracting topics: {e}")

    return topics


def delete_video(video_id: str):
    """Delete video from Video Indexer."""
    cfg = _load_config()
    vi_token = _get_tokens()

    url = f"{VI_BASE}/{cfg['LOCATION']}/Accounts/{cfg['ACCOUNT_ID']}/Videos/{video_id}"

    try:
        resp = requests.delete(url, params={"accessToken": vi_token}, timeout=30)
        resp.raise_for_status()
        logger.info(f"Deleted video {video_id} from Video Indexer.")

    except Exception as e:
        logger.warning(f"Could not delete video {video_id}: {e}")


def _parse_time(time_str: str) -> float:
    """Convert 'H:MM:SS.mmm' to seconds."""
    try:
        parts = time_str.split(":")

        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)

        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)

        return float(time_str)

    except Exception:
        return 0.0
