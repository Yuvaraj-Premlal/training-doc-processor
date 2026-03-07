"""
video_indexer.py - Azure Video Indexer ARM-based API client
"""
import os
import time
import logging
import requests
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

ARM_BASE = "https://management.azure.com"
VI_BASE  = "https://api.videoindexer.ai"


def _get_config():
    return {
        "account_id":      os.environ.get("VIDEO_INDEXER_ACCOUNT_ID", ""),
        "location":        os.environ.get("VIDEO_INDEXER_LOCATION", "eastus"),
        "subscription_id": os.environ.get("SUBSCRIPTION_ID", ""),
        "resource_group":  os.environ.get("RESOURCE_GROUP", "training-doc-generator"),
        "account_name":    os.environ.get("VI_ACCOUNT_NAME", "training-doc-ai"),
    }


def _get_arm_token() -> str:
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default")
    return token.token


def _get_vi_access_token(arm_token: str, cfg: dict) -> str:
    url = (
        f"{ARM_BASE}/subscriptions/{cfg['subscription_id']}"
        f"/resourceGroups/{cfg['resource_group']}"
        f"/providers/Microsoft.VideoIndexer/accounts/{cfg['account_name']}"
        f"/generateAccessToken?api-version=2024-01-01"
    )
    body    = {"permissionType": "Contributor", "scope": "Account"}
    headers = {"Authorization": f"Bearer {arm_token}", "Content-Type": "application/json"}
    resp    = requests.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["accessToken"]


def _get_tokens():
    cfg       = _get_config()
    arm_token = _get_arm_token()
    vi_token  = _get_vi_access_token(arm_token, cfg)
    return vi_token, cfg


def submit_video_from_blob(blob_url: str, video_name: str) -> str:
    vi_token, cfg = _get_tokens()
    url    = f"{VI_BASE}/{cfg['location']}/Accounts/{cfg['account_id']}/Videos"
    params = {
        "accessToken":    vi_token,
        "name":           video_name,
        "videoUrl":       blob_url,
        "language":       "auto",
        "indexingPreset": "Advanced",
        "streamingPreset": "NoStreaming",
    }
    resp = requests.post(url, params=params, timeout=60)
    resp.raise_for_status()
    video_id = resp.json()["id"]
    logger.info(f"Video submitted. video_id={video_id}")
    return video_id


def check_indexing_status(video_id: str) -> tuple:
    """
    Returns (state, progress_str, index_data_or_None)
    state: "Processed" | "Failed" | "Processing" | "Uploaded"
    """
    vi_token, cfg = _get_tokens()
    url    = f"{VI_BASE}/{cfg['location']}/Accounts/{cfg['account_id']}/Videos/{video_id}/Index"
    params = {"accessToken": vi_token, "language": "English"}
    resp   = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data     = resp.json()
    state    = data.get("state", "")
    progress = data.get("videos", [{}])[0].get("processingProgress", "0%")
    if state == "Processed":
        return state, "100%", data
    return state, progress, None


def extract_transcript(index_data: dict) -> list:
    segments = []
    try:
        items = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("transcript", [])
        )
        for item in items:
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


def extract_keyframes(index_data: dict, video_id: str) -> list:
    """
    Extract keyframes from Video Indexer shots.
    Also adds evenly-spaced interval frames to ensure good coverage
    especially for presentation/slide-style videos.
    """
    vi_token, cfg = _get_tokens()
    keyframes = []
    seen_timestamps = set()

    try:
        shots = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("shots", [])
        )

        # Get VI keyframes from shots
        for shot in shots:
            for kf in shot.get("keyFrames", []):
                instances    = kf.get("instances", [{}])
                timestamp    = _parse_time(instances[0].get("adjustedStart", "0:0:0"))
                thumbnail_id = instances[0].get("thumbnailId", "")
                if not thumbnail_id:
                    continue
                t_rounded = round(timestamp)
                if t_rounded in seen_timestamps:
                    continue
                seen_timestamps.add(t_rounded)
                thumb_url = (
                    f"{VI_BASE}/{cfg['location']}/Accounts/{cfg['account_id']}"
                    f"/Videos/{video_id}/Thumbnails/{thumbnail_id}"
                    f"?accessToken={vi_token}&format=Jpeg"
                )
                keyframes.append({
                    "timestamp":    timestamp,
                    "thumbnail_id": thumbnail_id,
                    "url":          thumb_url,
                    "source":       "vi_keyframe",
                })

        # Get video duration
        duration = _get_video_duration(index_data)
        logger.info(f"Video duration: {duration}s, VI keyframes: {len(keyframes)}")

        # Add interval-based thumbnails every 60 seconds for better coverage
        # This is critical for slide/presentation style videos
        if duration > 0:
            interval = 60  # every 60 seconds
            t = interval
            while t < duration:
                t_rounded = round(t)
                if t_rounded not in seen_timestamps:
                    seen_timestamps.add(t_rounded)
                    # Use Video Indexer thumbnail at specific time
                    thumb_url = (
                        f"{VI_BASE}/{cfg['location']}/Accounts/{cfg['account_id']}"
                        f"/Videos/{video_id}/Thumbnails"
                        f"?accessToken={vi_token}&format=Jpeg&time={t_rounded}"
                    )
                    keyframes.append({
                        "timestamp": float(t),
                        "thumbnail_id": f"interval_{t_rounded}",
                        "url":          thumb_url,
                        "source":       "interval",
                    })
                t += interval

        # Sort by timestamp
        keyframes.sort(key=lambda x: x["timestamp"])

    except Exception as e:
        logger.warning(f"Error extracting keyframes: {e}")

    logger.info(f"Total keyframes to caption: {len(keyframes)}")
    return keyframes


def _get_video_duration(index_data: dict) -> float:
    try:
        duration_str = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("duration", "0:0:0")
        )
        return _parse_time(duration_str)
    except Exception:
        return 0.0


def extract_topics(index_data: dict) -> list:
    topics = []
    try:
        raw = (
            index_data.get("videos", [{}])[0]
            .get("insights", {})
            .get("topics", [])
        )
        for t in raw:
            instances = t.get("instances", [{}])
            start = _parse_time(instances[0].get("adjustedStart", "0:0:0"))
            end   = _parse_time(instances[0].get("adjustedEnd",   "0:0:0"))
            topics.append({"name": t.get("name", ""), "start": start, "end": end})
    except Exception as e:
        logger.warning(f"Error extracting topics: {e}")
    return topics


def delete_video(video_id: str):
    try:
        vi_token, cfg = _get_tokens()
        url    = f"{VI_BASE}/{cfg['location']}/Accounts/{cfg['account_id']}/Videos/{video_id}"
        params = {"accessToken": vi_token}
        resp   = requests.delete(url, params=params, timeout=30)
        resp.raise_for_status()
        logger.info(f"Deleted video {video_id} from Video Indexer.")
    except Exception as e:
        logger.warning(f"Could not delete video {video_id}: {e}")


def _parse_time(time_str: str) -> float:
    try:
        parts = str(time_str).split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(time_str)
    except Exception:
        return 0.0
