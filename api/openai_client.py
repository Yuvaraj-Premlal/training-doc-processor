"""
openai_client.py - Azure OpenAI GPT-4o calls
"""
import os
import json
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
API_VER = "2024-02-15-preview"


def _get_config():
    return {
        "endpoint":   os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        "api_key":    os.environ.get("AZURE_OPENAI_KEY", ""),
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    }


def _chat(messages: list, max_tokens: int = 1500, json_mode: bool = False) -> str:
    cfg  = _get_config()
    url  = f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}/chat/completions?api-version={API_VER}"
    hdrs = {"api-key": cfg["api_key"], "Content-Type": "application/json"}
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=hdrs, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def download_frame(image_url: str) -> bytes | None:
    """Download image bytes from a URL. Returns None on failure."""
    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        logger.info(f"Downloaded frame: {len(resp.content)} bytes from {image_url[:80]}")
        return resp.content
    except Exception as e:
        logger.warning(f"Could not download frame: {e}")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def caption_keyframe(image_bytes: bytes, timestamp: float) -> dict:
    """Caption a keyframe given its raw image bytes."""
    import base64 as _b64
    minutes    = int(timestamp // 60)
    seconds    = int(timestamp % 60)
    time_label = f"{minutes:02d}:{seconds:02d}"

    image_b64  = _b64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing screenshots from a training video. "
                "The video may contain slides, diagrams, UI screens, or whiteboard content. "
                "For each frame describe what is shown and rate its usefulness as a training screenshot. "
                "Be generous with scores — even a slide with text is useful (score 3+). "
                "Respond only in JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Frame at timestamp {time_label}.\n"
                        "Respond with JSON:\n"
                        "{\n"
                        '  "caption": "One sentence describing what is shown",\n'
                        '  "ui_element": "What screen, slide, or content is visible",\n'
                        '  "user_action": "What is being demonstrated or explained, or null",\n'
                        '  "score": <1-5 where 5=very useful, 3=useful slide or diagram, 1=blank/transitional>,\n'
                        '  "is_useful": <true if score >= 2>\n'
                        "}"
                    ),
                },
            ],
        },
    ]
    try:
        raw    = _chat(messages, max_tokens=300, json_mode=True)
        result = json.loads(raw)
        result["timestamp"] = timestamp
        return result
    except Exception as e:
        logger.warning(f"Vision caption failed at {time_label}: {e}")
        return {
            "caption":    f"Screenshot at {time_label}",
            "ui_element": "",
            "user_action": None,
            "score":      2,
            "is_useful":  True,
            "timestamp":  timestamp,
        }


def caption_all_keyframes(keyframes: list) -> list:
    logger.info(f"Captioning {len(keyframes)} keyframes...")
    captioned = [caption_keyframe(kf["url"], kf["timestamp"]) for kf in keyframes]
    useful    = [f for f in captioned if f.get("is_useful", False)]
    logger.info(f"{len(useful)} useful frames out of {len(captioned)}.")
    return useful


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def build_document_structure(
    transcript_segments: list,
    captioned_frames: list,
    topics: list,
    video_name: str,
) -> dict:
    transcript_text = "\n".join(
        [f"[{int(s['start']//60):02d}:{int(s['start']%60):02d}] {s['text']}"
         for s in transcript_segments[:300]]
    )
    frames_summary = "\n".join([
        f"[{int(f['timestamp']//60):02d}:{int(f['timestamp']%60):02d}] "
        f"Score:{f.get('score',2)} | {f.get('caption','')} | UI: {f.get('ui_element','')}"
        for f in captioned_frames
    ])
    topics_text = "\n".join([
        f"[{int(t['start']//60):02d}:{int(t['start']%60):02d}] {t['name']}"
        for t in topics
    ])
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical writer creating step-by-step training manuals "
                "from video recordings. Match each section to the most relevant screenshot timestamp."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Video: '{video_name}'\n\n"
                f"=== TRANSCRIPT ===\n{transcript_text}\n\n"
                f"=== SCREENSHOTS AVAILABLE ===\n{frames_summary}\n\n"
                f"=== TOPICS ===\n{topics_text}\n\n"
                "Create a structured training document with 4-8 logical sections.\n"
                "For each section, pick the BEST matching screenshot_timestamp from the screenshots list above.\n"
                "Respond ONLY with JSON:\n"
                "{\n"
                '  "title": "Training document title",\n'
                '  "overview": "2-3 sentence overview",\n'
                '  "prerequisites": ["prereq 1"],\n'
                '  "sections": [{\n'
                '    "title": "Section title",\n'
                '    "objective": "What the user will learn",\n'
                '    "start_time": 0.0,\n'
                '    "end_time": 120.0,\n'
                '    "screenshot_timestamp": 45.0,\n'
                '    "transcript_chunk": "relevant transcript text",\n'
                '    "key_actions": ["action 1"]\n'
                '  }]\n'
                "}"
            ),
        },
    ]
    raw       = _chat(messages, max_tokens=3000, json_mode=True)
    structure = json.loads(raw)
    logger.info(f"Document structure: {len(structure.get('sections', []))} sections.")
    return structure


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def write_section_content(section: dict, section_number: int) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical writer creating step-by-step training manuals. "
                "Write in clear, direct language with numbered steps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write training content for Section {section_number}: '{section['title']}'\n\n"
                f"Objective: {section.get('objective', '')}\n"
                f"Transcript: {section.get('transcript_chunk', '')}\n"
                f"Key actions: {', '.join(section.get('key_actions', []))}\n\n"
                "Respond ONLY with JSON:\n"
                "{\n"
                '  "introduction": "1-2 sentence intro",\n'
                '  "steps": [{\n'
                '    "step_number": 1,\n'
                '    "instruction": "Clear action",\n'
                '    "detail": "Extra context or null",\n'
                '    "tip": "Helpful tip or null",\n'
                '    "warning": "Warning or null"\n'
                '  }],\n'
                '  "summary": "1-2 sentence summary",\n'
                '  "key_takeaways": ["takeaway 1"]\n'
                "}"
            ),
        },
    ]
    try:
        raw     = _chat(messages, max_tokens=2000, json_mode=True)
        content = json.loads(raw)
        section["content"] = content
        return section
    except Exception as e:
        logger.warning(f"Content generation failed for section {section_number}: {e}")
        section["content"] = {
            "introduction": section.get("objective", ""),
            "steps": [{"step_number": 1, "instruction": a, "detail": None, "tip": None, "warning": None}
                      for a in section.get("key_actions", [])],
            "summary": "", "key_takeaways": [],
        }
        return section


def write_all_sections(structure: dict) -> dict:
    sections = structure.get("sections", [])
    logger.info(f"Writing content for {len(sections)} sections...")
    for i, section in enumerate(sections):
        sections[i] = write_section_content(section, i + 1)
    structure["sections"] = sections
    return structure


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
def generate_quiz(structure: dict) -> list:
    section_titles = [s["title"] for s in structure.get("sections", [])]
    takeaways = []
    for s in structure.get("sections", []):
        takeaways.extend(s.get("content", {}).get("key_takeaways", []))
    messages = [
        {"role": "system", "content": "You create concise knowledge-check quizzes for training."},
        {
            "role": "user",
            "content": (
                f"Training: '{structure.get('title', '')}'\n"
                f"Sections: {', '.join(section_titles)}\n"
                f"Takeaways: {'; '.join(takeaways[:10])}\n\n"
                "Generate 5 multiple-choice questions. Respond ONLY with JSON:\n"
                '{"questions": [{"question": "?", "options": ["A.", "B.", "C.", "D."], "correct_answer": "A"}]}'
            ),
        },
    ]
    try:
        raw = _chat(messages, max_tokens=1500, json_mode=True)
        return json.loads(raw).get("questions", [])
    except Exception as e:
        logger.warning(f"Quiz generation failed: {e}")
        return []
