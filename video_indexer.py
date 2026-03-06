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

ACCOUNT_ID      = os.environ["VIDEO_INDEXER_ACCOUNT_ID"]
LOCATION        = os.environ["VIDEO_INDEXER_LOCATION"]
SUBSCRIPTION_ID = os.environ["SUBSCRIPTION_ID"]
RESOURCE_GROUP  = os.environ.get("RESOURCE_GROUP", "training-doc-generator")
VI_ACCOUNT_NAME = os.environ.get("VI_ACCOUNT_NAME", "training-doc-ai")

ARM_BASE  = "https://management.azure.com"
VI_BASE   = f"https://api.videoindexer.ai"


def _get_arm_token() -> str:
    """Get ARM access token using managed identity / DefaultAzureCredential."""
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default")
    return token.token


def _get_vi_access_token(arm_token: str) -> str:
    """Exchange ARM token for a Video Indexer account-level access token."""
    url = (
        f"{ARM_BASE}/subscriptions/{SUBSCRIPTION_ID}"
        f"/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.VideoIndexer/accounts/{VI_ACCOUNT_NAME}"
        f"/generateAccessToken?api-version=2024-01-01"
    )
    body = {"permissionType": "Contributor", "scope": "Account"}
    headers = {"Authorization": f"Bearer {arm_token}", "Content-Type": "application/json"}
    resp = requests.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["accessToken"]


def _get_tokens():
    arm_token = _get_arm_token()
    vi_token  = _get_vi_access_token(arm_token)
    return vi_token


def submit_video_from_blob(blob_url: str, video_name: str) -> str:
    """
    Submit a video to Video Indexer from a blob SAS URL.
    Returns the Video Indexer video_id.
    """
    vi_token = _get_tokens()
    url = f"{VI_BASE}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos"
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
    """
    Poll Video Indexer until the video finishes processing.
    Returns the full index JSON when done.
    Raises TimeoutError if max_wait seconds exceeded.
    """
    vi_token = _get_tokens()
    url = f"{VI_BASE}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{video_id}/Index"
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
        elif state == "Failed":
            raise RuntimeError(f"Video Indexer failed for video_id={video_id}")

        # Refresh token every ~10 minutes to avoid expiry
        if elapsed % 600 == 0 and elapsed > 0:
            vi_token = _get_tokens()

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Video Indexer did not finish within {max_wait}s for video_id={video_id}")


def extract_transcript(index_data: dict) -> list[dict]:
    """
    Extract transcript as a list of segments:
    [{ "text": "...", "start": 12.5, "end": 18.2 }, ...]
    """
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
            instances = item.get("instances", [{}])
            start = _parse_time(instances[0].get("start", "0:0:0"))
            end   = _parse_time(instances[0].get("end",   "0:0:0"))
            segments.append({"text": text, "start": start, "end": end})
    except Exception as e:
        logger.warning(f"Error extracting transcript: {e}")
    return segments


def extract_keyframes(index_data: dict, video_id: str) -> list[dict]:
    """
    Extract keyframe metadata and thumbnail URLs.
    Returns: [{ "timestamp": 12.5, "thumbnail_id": "...", "url": "..." }, ...]
    """
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
                instances = kf.get("instances", [{}])
                timestamp = _parse_time(instances[0].get("adjustedStart", "0:0:0"))
                thumbnail_id = kf.get("instances", [{}])[0].get("thumbnailId", "")
                if not thumbnail_id:
                    continue
                thumb_url = (
                    f"{VI_BASE}/{LOCATION}/Accounts/{ACCOUNT_ID}"
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


def extract_topics(index_data: dict) -> list[dict]:
    """
    Extract high-level topics with their timestamps.
    Returns: [{ "name": "...", "start": 0.0, "end": 120.0 }, ...]
    """
    topics = []
    try:
        raw_topics = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("topics", [])
        )
        for t in raw_topics:
            instances = t.get("instances", [{}])
            start = _parse_time(instances[0].get("adjustedStart", "0:0:0"))
            end   = _parse_time(instances[0].get("adjustedEnd",   "0:0:0"))
            topics.append({"name": t.get("name", ""), "start": start, "end": end})
    except Exception as e:
        logger.warning(f"Error extracting topics: {e}")
    return topics


def delete_video(video_id: str):
    """Delete the video from Video Indexer after processing."""
    vi_token = _get_tokens()
    url = f"{VI_BASE}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{video_id}"
    params = {"accessToken": vi_token}
    try:
        resp = requests.delete(url, params=params, timeout=30)
        resp.raise_for_status()
        logger.info(f"Deleted video {video_id} from Video Indexer.")
    except Exception as e:
        logger.warning(f"Could not delete video {video_id} from VI: {e}")


def _parse_time(time_str: str) -> float:
    """Convert 'H:MM:SS.mmm' or 'H:MM:SS' string to float seconds."""
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(time_str)
    except Exception:
        return 0.0
