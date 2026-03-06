"""
openai_client.py
Handles all Azure OpenAI calls:
  - Call 1: GPT-4o Vision  → caption + score each keyframe
  - Call 2: GPT-4o Text    → build document structure (JSON)
  - Call 3: GPT-4o Text    → write training content per section
"""

import os
import json
import base64
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VER    = "2024-02-15-preview"

HEADERS = {
    "api-key": API_KEY,
    "Content-Type": "application/json",
}


def _chat(messages: list, max_tokens: int = 1500, json_mode: bool = False) -> str:
    """Base chat completion call to Azure OpenAI."""
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VER}"
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=HEADERS, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── CALL 1: Vision — Caption + Score each keyframe ───────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def caption_keyframe(image_url: str, timestamp: float) -> dict:
    """
    Send a keyframe image to GPT-4o Vision.
    Returns: { "caption": "...", "score": 4, "is_useful": True }
    """
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    time_label = f"{minutes:02d}:{seconds:02d}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing screenshots from a software UI training video. "
                "For each frame, describe what is shown and rate its usefulness "
                "as a training illustration. Respond only in JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                },
                {
                    "type": "text",
                    "text": (
                        f"This frame is at timestamp {time_label} in the video.\n"
                        "Respond with JSON in this exact format:\n"
                        "{\n"
                        '  "caption": "One sentence describing what is shown on screen",\n'
                        '  "ui_element": "What UI screen, dialog, or feature is visible",\n'
                        '  "user_action": "What the user appears to be doing (or null)",\n'
                        '  "score": <integer 1-5 where 5=very useful training screenshot>,\n'
                        '  "is_useful": <true if score >= 3, false otherwise>\n'
                        "}"
                    ),
                },
            ],
        },
    ]

    try:
        raw = _chat(messages, max_tokens=300, json_mode=True)
        result = json.loads(raw)
        result["timestamp"] = timestamp
        result["url"] = image_url
        return result
    except Exception as e:
        logger.warning(f"Vision caption failed for timestamp {time_label}: {e}")
        return {
            "caption": f"Screenshot at {time_label}",
            "ui_element": "",
            "user_action": None,
            "score": 2,
            "is_useful": False,
            "timestamp": timestamp,
            "url": image_url,
        }


def caption_all_keyframes(keyframes: list[dict]) -> list[dict]:
    """
    Caption all keyframes, filter to useful ones only.
    Returns list of captioned frames with score >= 3.
    """
    logger.info(f"Captioning {len(keyframes)} keyframes...")
    captioned = []
    for i, kf in enumerate(keyframes):
        logger.info(f"  Captioning frame {i+1}/{len(keyframes)} @ {kf['timestamp']:.1f}s")
        result = caption_keyframe(kf["url"], kf["timestamp"])
        captioned.append(result)

    useful = [f for f in captioned if f.get("is_useful", False)]
    logger.info(f"  {len(useful)} useful frames out of {len(captioned)} total.")
    return useful


# ── CALL 2: Architect — Build document structure ──────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def build_document_structure(
    transcript_segments: list[dict],
    captioned_frames: list[dict],
    topics: list[dict],
    video_name: str,
) -> dict:
    """
    GPT-4o analyzes transcript + frames and returns a structured document outline.
    Returns JSON: { "title": "...", "overview": "...", "sections": [...] }
    """
    # Build compact transcript text with timestamps
    transcript_text = "\n".join(
        [f"[{int(s['start']//60):02d}:{int(s['start']%60):02d}] {s['text']}" 
         for s in transcript_segments[:300]]  # cap to avoid token limits
    )

    # Build frame summary
    frames_summary = "\n".join([
        f"[{int(f['timestamp']//60):02d}:{int(f['timestamp']%60):02d}] "
        f"Score:{f['score']} | {f['caption']} | UI: {f.get('ui_element','')}"
        for f in captioned_frames
    ])

    # Build topics summary
    topics_text = "\n".join([
        f"[{int(t['start']//60):02d}:{int(t['start']%60):02d}] {t['name']}"
        for t in topics
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical writer creating step-by-step training manuals "
                "from software UI walkthrough videos. You create clear, structured, "
                "practical documentation that helps users learn software effectively."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Video: '{video_name}'\n\n"
                f"=== TRANSCRIPT (with timestamps) ===\n{transcript_text}\n\n"
                f"=== SCREENSHOT ANALYSIS (with timestamps) ===\n{frames_summary}\n\n"
                f"=== TOPICS DETECTED ===\n{topics_text}\n\n"
                "Create a structured training document outline. "
                "Group the content into 4-8 logical sections based on the workflow shown. "
                "For each section, assign the best matching screenshot timestamp.\n\n"
                "Respond ONLY with JSON in this exact format:\n"
                "{\n"
                '  "title": "Training document title",\n'
                '  "overview": "2-3 sentence overview of what this training covers",\n'
                '  "prerequisites": ["prereq 1", "prereq 2"],\n'
                '  "sections": [\n'
                '    {\n'
                '      "title": "Section title",\n'
                '      "objective": "What the user will learn in this section",\n'
                '      "start_time": 0.0,\n'
                '      "end_time": 120.0,\n'
                '      "screenshot_timestamp": 45.0,\n'
                '      "transcript_chunk": "relevant transcript text for this section",\n'
                '      "key_actions": ["action 1", "action 2"]\n'
                '    }\n'
                '  ]\n'
                "}"
            ),
        },
    ]

    raw = _chat(messages, max_tokens=3000, json_mode=True)
    structure = json.loads(raw)
    logger.info(f"Document structure built: {len(structure.get('sections', []))} sections.")
    return structure


# ── CALL 3: Writer — Generate training content per section ────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def write_section_content(section: dict, section_number: int) -> dict:
    """
    GPT-4o writes full training content for a single section.
    Returns the section dict enriched with written content.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical writer creating step-by-step software training manuals. "
                "Write in clear, direct language. Use numbered steps for procedures. "
                "Include tips and warnings where relevant. Be specific and actionable."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write the training content for Section {section_number}: '{section['title']}'\n\n"
                f"Learning objective: {section['objective']}\n\n"
                f"Transcript from this section:\n{section.get('transcript_chunk', '')}\n\n"
                f"Key actions observed: {', '.join(section.get('key_actions', []))}\n\n"
                "Write the following in JSON format:\n"
                "{\n"
                '  "introduction": "1-2 sentence intro to this section",\n'
                '  "steps": [\n'
                '    {\n'
                '      "step_number": 1,\n'
                '      "instruction": "Clear action the user takes",\n'
                '      "detail": "Additional explanation or context (can be null)",\n'
                '      "tip": "Helpful tip (can be null)",\n'
                '      "warning": "Important warning (can be null)"\n'
                '    }\n'
                '  ],\n'
                '  "summary": "1-2 sentence summary of what was accomplished",\n'
                '  "key_takeaways": ["takeaway 1", "takeaway 2"]\n'
                "}"
            ),
        },
    ]

    try:
        raw = _chat(messages, max_tokens=2000, json_mode=True)
        content = json.loads(raw)
        section["content"] = content
        return section
    except Exception as e:
        logger.warning(f"Content generation failed for section {section_number}: {e}")
        section["content"] = {
            "introduction": section.get("objective", ""),
            "steps": [{"step_number": 1, "instruction": t, "detail": None, "tip": None, "warning": None}
                      for t in section.get("key_actions", [])],
            "summary": "",
            "key_takeaways": [],
        }
        return section


def write_all_sections(structure: dict) -> dict:
    """Write content for all sections in the document structure."""
    sections = structure.get("sections", [])
    logger.info(f"Writing content for {len(sections)} sections...")
    for i, section in enumerate(sections):
        logger.info(f"  Writing section {i+1}/{len(sections)}: {section['title']}")
        sections[i] = write_section_content(section, i + 1)
    structure["sections"] = sections
    return structure


# ── CALL 4: Quiz Generator ────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
def generate_quiz(structure: dict) -> list[dict]:
    """
    Generate a short knowledge-check quiz based on the full document.
    Returns: [{ "question": "...", "options": [...], "answer": "..." }]
    """
    section_titles = [s["title"] for s in structure.get("sections", [])]
    takeaways = []
    for s in structure.get("sections", []):
        takeaways.extend(s.get("content", {}).get("key_takeaways", []))

    messages = [
        {
            "role": "system",
            "content": "You create concise knowledge-check quizzes for software training documents.",
        },
        {
            "role": "user",
            "content": (
                f"Training document: '{structure.get('title', '')}'\n"
                f"Sections covered: {', '.join(section_titles)}\n"
                f"Key takeaways: {'; '.join(takeaways[:10])}\n\n"
                "Generate 5 multiple-choice quiz questions to check understanding.\n"
                "Respond ONLY with JSON:\n"
                "{\n"
                '  "questions": [\n'
                '    {\n'
                '      "question": "Question text?",\n'
                '      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],\n'
                '      "correct_answer": "A"\n'
                '    }\n'
                '  ]\n'
                "}"
            ),
        },
    ]

    try:
        raw = _chat(messages, max_tokens=1500, json_mode=True)
        return json.loads(raw).get("questions", [])
    except Exception as e:
        logger.warning(f"Quiz generation failed: {e}")
        return []
