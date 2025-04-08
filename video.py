# -*- coding: utf-8 -*-
import os
import subprocess
import math
import json
import logging
# import argparse # No longer used
import shutil
import time
from pydub import AudioSegment
from openai import OpenAI, APIError
from moviepy.editor import (VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips)
from moviepy.config import change_settings
from dotenv import load_dotenv

# --- Configuration ---
# change_settings({"FFMPEG_BINARY": "/path/to/your/ffmpeg"})

# --- Constants ---
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp_files"
AUDIO_CHUNKS_FOLDER = os.path.join(TEMP_FOLDER, "audio_chunks")
URL_INPUT_FILE = "urls.txt"
DELAY_BETWEEN_VIDEOS_SECONDS = 5

CHUNK_SIZE_MB = 24
CHUNK_LENGTH_MS = 10 * 60 * 1000

# Subtitle appearance
SUBTITLE_FONT = './NanumGothicBold.ttf'
SUBTITLE_FONTSIZE = 60
SUBTITLE_COLOR = 'white'
SUBTITLE_BG_COLOR = 'black'
SUBTITLE_POS = ('center', 0.8)

# Translation settings
TRANSLATION_RETRY_DELAY_SECONDS = 5

# --- Logging Setup ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])

# --- Helper Functions ---
def run_command(command):
    logging.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        stdout_log = result.stdout
        if len(stdout_log) > 1000: stdout_log = result.stdout[:500] + "\n...\n" + result.stdout[-500:]
        if stdout_log.strip(): logging.info(f"Command stdout (snippet):\n{stdout_log}")
        if result.stderr:
            stderr_log = result.stderr
            if len(stderr_log) > 1000: stderr_log = result.stderr[:500] + "\n...\n" + result.stderr[-500:]
            if stderr_log.strip(): logging.warning(f"Command stderr (snippet):\n{stderr_log}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {' '.join(command)}"); logging.error(f"Return code: {e.returncode}")
        stderr_str = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
        stdout_str = e.stdout.decode('utf-8', errors='ignore') if isinstance(e.stdout, bytes) else str(e.stdout)
        logging.error(f"Stderr:\n{stderr_str}"); logging.error(f"Stdout:\n{stdout_str}")
        return False
    except FileNotFoundError: logging.error(f"Command not found: {command[0]}."); return False
    except Exception as e: logging.error(f"Unexpected error running command: {e}"); logging.exception("Traceback:"); return False

def setup_directories():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_CHUNKS_FOLDER, exist_ok=True)
    logging.info(f"Ensured directories: {OUTPUT_FOLDER}, {TEMP_FOLDER}, {AUDIO_CHUNKS_FOLDER}")

def cleanup(paths_to_delete):
    logging.info("Starting cleanup...")
    for path in paths_to_delete:
        target_path = os.path.abspath(path)
        try:
            if os.path.isfile(target_path): os.remove(target_path); logging.info(f"Removed file: {target_path}")
            elif os.path.isdir(target_path): shutil.rmtree(target_path); logging.info(f"Removed directory: {target_path}")
            else: logging.warning(f"Cleanup path not found: {target_path}")
        except OSError as e: logging.error(f"Error removing {target_path}: {e}")
    logging.info("Cleanup finished.")

# --- Translation Function ---
def translate_text_to_korean( text_to_translate: str, client: OpenAI, max_retries=2, prev_context: str | None = None, next_context: str | None = None) -> str | None:
    # (Function remains unchanged)
    if not text_to_translate or not text_to_translate.strip(): return text_to_translate
    prev_ctx = prev_context.strip() if prev_context else ""; next_ctx = next_context.strip() if next_context else ""
    if prev_ctx or next_ctx:
        prompt = (f"Expert EN->KO subtitle translator.\nNatural flow with context.\n\nCONTEXT:\n"
                  f"{'Prev: \"' + prev_ctx + '\"\\n' if prev_ctx else ''}{'Next: \"' + next_ctx + '\"\\n' if next_ctx else ''}\nTASK:\n"
                  f"Translate ONLY 'Target Text Segment' to Korean. Output ONLY translation.\n\nTarget:\n\"\"\"\n{text_to_translate}\n\"\"\"")
        system_message = "Expert EN->KO subtitle translator. Natural flow. Output only translation."; log_prefix = "Contextual translation"
    else:
        prompt = (f"Translate English to Korean. Output only translation.\n\nEnglish:\n\"\"\"\n{text_to_translate}\n\"\"\""); system_message = "Proficient EN->KO translator. Output only translation."; log_prefix = "Simple translation"
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.5)
            korean_text = response.choices[0].message.content.strip()
            if korean_text == text_to_translate and text_to_translate: logging.warning(f"{log_prefix} result identical: '{text_to_translate[:60]}...'.")
            return korean_text
        except APIError as e:
            logging.warning(f"{log_prefix} API error (Att {attempt+1}/{max_retries+1}): {e.status_code}");
            if attempt < max_retries:
                if e.status_code == 429 or e.status_code >= 500: time.sleep(TRANSLATION_RETRY_DELAY_SECONDS); continue
                else: logging.error("Non-retryable API error."); break
            else: logging.error("Max retries API error.")
        except Exception as e:
            logging.error(f"Unexpected {log_prefix} error (Att {attempt+1}/{max_retries+1}): {e}"); logging.exception("Traceback:")
            if attempt < max_retries: time.sleep(TRANSLATION_RETRY_DELAY_SECONDS)
            else: logging.error("Max retries unexpected error.")
    logging.error(f"{log_prefix} failed: '{text_to_translate[:60]}...'. Returning original."); return text_to_translate

# --- Phase Implementations ---
def download_full_audio(youtube_url, audio_path):
    # (Function remains unchanged)
    logging.info(f"Starting full audio download for: {youtube_url}")
    audio_command = ['yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'mp3', '--audio-quality', '2', '-o', audio_path.replace('.mp3', '.%(ext)s'), youtube_url]
    if not run_command(audio_command): logging.error("Full audio download failed."); return False
    final_audio_path = audio_path
    if not os.path.exists(final_audio_path):
         potential_path = audio_path + ".mp3"
         if os.path.exists(potential_path):
             logging.warning(f"Downloaded audio found at {potential_path}, renaming to {final_audio_path}")
             try: shutil.move(potential_path, final_audio_path)
             except Exception as move_err: logging.error(f"Failed to rename audio: {move_err}"); return False
         else: logging.error(f"Extracted audio not found: {final_audio_path}"); return False
    logging.info(f"Full audio saved: {final_audio_path}"); return True

# MODIFIED: download_video_section (Removed --force-keyframes-at-cuts)
def download_video_section(youtube_url, start_time, end_time, video_section_path):
    """Downloads only the specified video section (<=1080p) using yt-dlp --download-sections."""
    logging.info(f"Starting video section download for: {youtube_url}")
    logging.info(f"Section: {start_time:.2f}s to {end_time:.2f}s")
    section_format = f"*{start_time:.3f}-{end_time:.3f}"

    video_command = [
        'yt-dlp',
        '-S', 'res:1080,vcodec:h264,vcodec:vp9', # Sort by Res (<=1080), then prefer H.264, then VP9
        '-f', 'bestvideo[vcodec!=av01]+bestaudio/best[vcodec!=av01]',
        '--download-sections', section_format,
        # '--force-keyframes-at-cuts', # REMOVED this flag
        '--merge-output-format', 'mp4',
        '-o', video_section_path,
        youtube_url
    ]
    if not run_command(video_command):
        logging.error("Video section download failed.")
        return False
    logging.info(f"Video section download command finished.")
    if not os.path.exists(video_section_path):
         logging.error(f"Downloaded video section not found: {video_section_path}")
         return False
    logging.info(f"Video section saved: {video_section_path}")
    return True

# chunk_audio remains the same...
def chunk_audio(original_audio_path, chunk_folder):
    # (Function remains unchanged)
    logging.info(f"Checking audio file size: {original_audio_path}")
    try: file_size = os.path.getsize(original_audio_path)
    except FileNotFoundError: logging.error(f"Audio file not found: {original_audio_path}"); return None
    except Exception as e: logging.error(f"Error getting audio file size: {e}"); return None
    limit_bytes = CHUNK_SIZE_MB * 1024 * 1024
    if file_size <= limit_bytes: logging.info("Audio size OK. No chunking."); return [original_audio_path]
    logging.info(f"Audio size ({file_size/(1024*1024):.2f}MB) > limit ({CHUNK_SIZE_MB}MB). Chunking..."); chunk_paths = []
    try:
        audio = AudioSegment.from_file(original_audio_path); total_length_ms = len(audio)
        num_chunks = math.ceil(total_length_ms / CHUNK_LENGTH_MS); logging.info(f"Splitting into {num_chunks} chunks.")
        for i in range(num_chunks):
            start_ms = i * CHUNK_LENGTH_MS; end_ms = min(start_ms + CHUNK_LENGTH_MS, total_length_ms)
            if start_ms >= end_ms: continue
            chunk = audio[start_ms:end_ms]; chunk_file_path = os.path.join(chunk_folder, f"chunk_{i+1}.mp3")
            logging.info(f"Exporting chunk {i+1}/{num_chunks}: {chunk_file_path} ({start_ms/1000:.2f}s to {end_ms/1000:.2f}s)")
            chunk.export(chunk_file_path, format="mp3")
            chunk_size_bytes = os.path.getsize(chunk_file_path)
            if chunk_size_bytes > limit_bytes: logging.warning(f"Chunk {chunk_file_path} ({chunk_size_bytes/(1024*1024):.2f}MB) exceeds target size!")
            chunk_paths.append(chunk_file_path)
        logging.info(f"Finished chunking: {len(chunk_paths)} chunks."); return chunk_paths
    except Exception as e: logging.error(f"Error during chunking: {e}"); logging.exception("Traceback:"); return None

# transcribe_chunks remains the same...
def transcribe_chunks(audio_chunk_paths, client):
    # (Function remains unchanged)
    all_segments = []; full_transcript_text = ""; previous_transcript_text_for_prompt = ""; current_offset_time_seconds = 0.0
    logging.info(f"Starting transcription for {len(audio_chunk_paths)} chunks...")
    for i, chunk_path in enumerate(audio_chunk_paths):
        logging.info(f"Transcribing chunk {i+1}/{len(audio_chunk_paths)}: {chunk_path}")
        try:
            with open(chunk_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="verbose_json",
                                                              timestamp_granularities=["segment"], prompt=previous_transcript_text_for_prompt)
            if hasattr(response, 'language'): logging.info(f"Whisper detected lang {i+1}: {response.language}")
            if not all(hasattr(response, attr) for attr in ['segments', 'text', 'duration']): logging.error(f"Whisper response missing data chunk {i+1}."); return None, None
            segments = response.segments; current_chunk_text = response.text; chunk_duration = response.duration
            logging.info(f"Chunk {i+1}: Duration={chunk_duration:.2f}s, Segments={len(segments)}")
            full_transcript_text += current_chunk_text + " "; previous_transcript_text_for_prompt = current_chunk_text[-225:]
            for segment in segments:
                adjusted_start = segment.start + current_offset_time_seconds; adjusted_end = segment.end + current_offset_time_seconds
                all_segments.append({'id': segment.id, 'seek': segment.seek, 'start': adjusted_start, 'end': adjusted_end,
                                     'text': segment.text if segment.text is not None else '', 'tokens': segment.tokens, 'temperature': segment.temperature,
                                     'avg_logprob': segment.avg_logprob, 'compression_ratio': segment.compression_ratio, 'no_speech_prob': segment.no_speech_prob})
            current_offset_time_seconds += chunk_duration
        except APIError as e: logging.error(f"API error chunk {i+1}: {e.status_code} - {e.message}"); return None, None
        except FileNotFoundError: logging.error(f"Audio chunk not found: {chunk_path}"); return None, None
        except Exception as e: logging.error(f"Error processing chunk {i+1}: {e}"); logging.exception("Traceback:"); return None, None
    logging.info(f"Finished transcription. Segments: {len(all_segments)}. Length: {len(full_transcript_text)} chars."); return all_segments, full_transcript_text.strip()

# find_key_segment remains the same...
def find_key_segment(english_transcript, client):
    # (Function remains unchanged, uses revised prompt)
    if not english_transcript or not english_transcript.strip(): logging.error("Cannot find key segment: Input English empty."); return None, None, None
    logging.info("Identifying key segment using GPT-4o (English)...")
    prompt_content = (
        "You are an expert video analyst tasked with identifying the **single most impactful, engaging, and informative continuous segment** from the following **English** transcript. Your primary goal is selecting the **absolute best content** suitable for a viral social media highlight clip – prioritize substance and hook over strict duration adherence.\n\n"
        "CRITERIA:\n"
        "1.  **Peak Content:** Find the core message, a compelling argument, a surprising revelation, or the most exciting/useful part.\n"
        "2.  **Duration:** Aim for a segment **roughly between 60 and 90 seconds**. However, finding the *perfect* content segment is **more important** than rigidly hitting this duration. A slightly shorter or longer segment (e.g., 50-100 seconds) is acceptable if it captures the peak moment best.\n"
        "3.  **No Filler:** Avoid intros, outros, excessive pleasantries, or parts that don't make sense without much wider context.\n"
        "4.  **Crucial Start/End:** Selecting the precise start and end times is critical to capture the essence effectively.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object containing:\n"
        "1. 'start_time': The start time of the *best* segment in seconds (float or int), relative to the transcript start.\n"
        "2. 'end_time': The end time of the *best* segment in seconds (float or int), relative to the transcript start.\n"
        "3. 'reasoning': Your brief justification for *why this specific segment is the most impactful/viral*, **written in Korean (한국어)**.\n\n"
        "TRANSCRIPT (English):\n"
        f"{english_transcript}"
    )
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[
            {"role": "system", "content": "Analyst returning JSON {start_time: float, end_time: float, reasoning: str_in_korean} from English text."},
            {"role": "user", "content": prompt_content}], response_format={"type": "json_object"}, temperature=0.3)
        response_content = response.choices[0].message.content
        if not response_content: logging.error("LLM empty response key segment."); return None, None, None
        logging.info(f"LLM response key segment: {response_content}"); result_json = json.loads(response_content)
        start_time=result_json.get("start_time"); end_time=result_json.get("end_time"); reasoning=result_json.get("reasoning")
        if start_time is None or end_time is None: logging.error(f"LLM missing time: {result_json}"); return None, None, None
        if reasoning is None: logging.warning(f"LLM missing reasoning: {result_json}"); reasoning = ""
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)): logging.error(f"Invalid time types: {result_json}"); return None, None, None
        if not isinstance(reasoning, str): logging.error(f"Invalid reasoning type: {result_json}"); reasoning = str(reasoning)
        start_time=float(start_time); end_time=float(end_time); duration = end_time - start_time
        if duration <= 0: logging.error(f"Invalid duration: {duration:.2f}s"); return None, None, None
        target_min=50; target_max=100
        if not (target_min <= duration <= target_max): logging.warning(f"Segment duration ({duration:.2f}s) outside ({target_min}-{target_max}s).")
        else: logging.info(f"Identified segment: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        logging.info(f"Reasoning (Korean): {reasoning}"); return start_time, end_time, reasoning
    except APIError as e: logging.error(f"API error key segment: {e.status_code}"); return None, None, None
    except json.JSONDecodeError as e: logging.error(f"Failed parse key segment JSON: {e}\nResponse: {response_content}"); return None, None, None
    except Exception as e: logging.error(f"Error key segment ID: {e}"); logging.exception("Traceback:"); return None, None, None

# create_subtitled_video remains the same...
def create_subtitled_video(video_section_path, korean_segments_for_clip, key_start_time, final_output_path):
    # (Function remains unchanged)
    logging.info(f"Starting subtitling for video section: {video_section_path}")
    logging.info(f"Adding {len(korean_segments_for_clip)} subtitle segments.")
    if not os.path.exists(video_section_path): logging.error(f"Video section file not found: {video_section_path}"); return False
    video_clip = None; final_clip = None; subtitle_clips = []
    try:
        logging.info(f"Loading video section: {video_section_path}"); video_clip = VideoFileClip(video_section_path)
        clip_w, clip_h, clip_duration = video_clip.w, video_clip.h, video_clip.duration
        logging.info(f"Video section loaded. Dimensions: {clip_w}x{clip_h}, Duration: {clip_duration:.2f}s")
        added_count = 0
        for i, segment in enumerate(korean_segments_for_clip):
            text = segment.get('text', '').strip();
            if not text: continue
            rel_start = max(0, segment['start'] - key_start_time); rel_end = min(clip_duration, segment['end'] - key_start_time)
            duration = rel_end - rel_start
            if duration > 0.05:
                try:
                    txt_clip = TextClip(txt=text, fontsize=SUBTITLE_FONTSIZE, font=SUBTITLE_FONT, color=SUBTITLE_COLOR, bg_color=SUBTITLE_BG_COLOR,
                                        method='caption', size=(clip_w * 0.9, None), align='center'
                                       ).set_position(SUBTITLE_POS, relative=True).set_start(rel_start).set_duration(duration)
                    subtitle_clips.append(txt_clip); added_count += 1
                except Exception as tc_err: logging.error(f"Error TextClip {i+1}: {tc_err}"); logging.exception("Traceback:")
        logging.info(f"Successfully created {len(subtitle_clips)} TextClips.")
        if not subtitle_clips:
            logging.warning("No subtitles created. Copying video section.");
            try: shutil.copyfile(video_section_path, final_output_path); logging.info(f"Saved non-subtitled clip: {final_output_path}"); return True
            except Exception as copy_err: logging.error(f"Failed to copy non-subtitled section: {copy_err}"); return False
        logging.info(f"Compositing video + {added_count} subtitles..."); final_clip = CompositeVideoClip([video_clip] + subtitle_clips, size=video_clip.size)
        logging.info(f"Writing final video: {final_output_path}"); final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", preset='medium', threads=4, logger='bar')
        logging.info(f"Subtitled video saved: {final_output_path}"); return True
    except Exception as e: logging.error(f"Error subtitling section: {e}"); logging.exception("Traceback:"); return False
    finally:
        logging.debug("Closing subtitle video objects...");
        for clip_obj in subtitle_clips:
             try: del clip_obj
             except Exception: pass
        subtitle_clips.clear()
        if final_clip:
            try: del final_clip; logging.debug("Final clip deleted.")
            except Exception as e: logging.warning(f"Error cleaning final_clip: {e}")
        if video_clip:
            try: video_clip.close(); logging.debug("Video section clip closed.")
            except Exception as e: logging.warning(f"Error closing video section: {e}")
        logging.debug("Finished closing subtitle objects.")

# generate_social_post remains the same...
def generate_social_post(korean_segments_for_clip, reasoning, client):
    # (Function remains unchanged, uses full prompts)
    logging.info("Generating Korean social media post using GPT-4o...")
    clip_text_for_prompt = " ".join([seg['text'].strip() for seg in korean_segments_for_clip if seg.get('text') and seg['text'].strip()])
    if not clip_text_for_prompt: logging.warning("No clip text for post."); clip_text_for_prompt = "[내용 요약 없음]"
    if not reasoning: logging.warning("No reasoning for post."); reasoning = "선정 이유 없음."
    example_post_1 = """
와 벌써..?
Runway에서 "Gen-4 Turbo"를 공개했습니다.
기존 모델 중 가장 강력하면서도 빠른 속도를 자랑하는데요
10초짜리 영상을 생성하는 데 단 30초밖에 걸리지 않아, 빠르게 반복 실험하거나 아이디어 테스트 용도로 적합합니다.
현재 모든 요금제에서 사용 가능합니다.
#Runway #Gen4Turbo #AI영상 #기술트렌드 #생성형AI
"""
    example_post_2 = """
미래는 정해져 있습니다.
오픈AI가 조니 아이브와 협업 중인 하드웨어 스타트업 "io Products"를 인수하려는 논의가 최근 있었던 것으로 알려졌습니다.
이 프로젝트는 샘 알트먼과 조니 아이브가 함께 개발 중인 인공지능 기반 개인 기기입니다.
논의된 인수 금액은 5억 달러 이상이며, 해당 기기는 화면이 없고 음성 제어 방식으로 작동할 예정입니다.
영화 Her에 등장하는 AI 기기에서 영감을 받은 형태입니다.
현재 인수 외에도 파트너십 체결 가능성도 거론되고 있으며, 일상 속에서 AI를 더 자연스럽게 통합하는 것이 목표입니다.
이 프로젝트는 오픈AI가 기존 협력 관계에 있는 애플과 직접 경쟁하게 만들 수 있습니다.
SF보다 더 SF 같은 현실이며 시나리오대로 흘러가고 있습니다.
기술 보다 미래를 보는 것이 현실적입니다.
#오픈AI #조니아이브 #AI하드웨어 #미래기술 #샘알트먼
"""
    prompt_content = (
        "🟩 목표 (Goal):\n"
        "제공된 비디오 클립의 한국어 트랜스크립트 발췌 내용과 이 클립이 선택된 이유(Reasoning, 한국어)를 바탕으로, 사람들이 이 비디오 클립을 보고 싶어지도록 **호기심을 자극하고 참여를 유도하는** 매력적인 소셜 미디어 게시물을 **한국어**로 작성하십시오. 게시물은 Twitter/X, Instagram Reels 설명, Facebook, LinkedIn 등에 적합해야 합니다.\n\n"
        "🟦 반환 형식 (Return Format):\n"
        "간결하고 자연스러운 **한국어 게시물 텍스트** 한 문단을 반환합니다. 다른 부가 설명, 제목, 머리말/꼬리말 없이 순수한 게시물 내용만 포함해야 합니다. 게시물 끝에는 관련성 높고 인기 있는 **한국어 해시태그** 3~5개를 '#해시태그' 형식으로 포함하십시오.\n\n"
        "🟧 주의사항 (Warnings):\n"
        "- 게시물은 시선을 사로잡고, 간결하며, 임팩트 있어야 합니다.\n"
        "- 단순히 내용을 요약하는 것을 넘어, 클립의 핵심이나 흥미로운 점을 부각하여 **마케팅적인 매력**을 더하십시오.\n"
        "- 한국 소셜 미디어 사용자 및 트렌드에 맞는 **자연스럽고 관련성 높은 한국어 해시태그**를 신중하게 선택하십시오.\n"
        "- 제공된 '스타일 참고용 예시 게시물'의 톤(호기심 유발, 정보성, 때로는 대담함)과 구조(본문 + 해시태그)를 참고하되, 내용은 주어진 컨텍스트에 맞게 창의적으로 작성하십시오.\n\n"
        "⬛ 컨텍스트 (Context):\n\n"
        "1. 클립 선택 이유 (Reasoning for clip selection - Provided in Korean):\n"
        f"   \"{reasoning}\"\n\n"
        "2. 클립 트랜스크립트 발췌 (Transcript Excerpt from Clip - Provided in Korean):\n"
        f"   \"\"\"{clip_text_for_prompt}\"\"\"\n\n"
        "3. 스타일 참고용 예시 게시물 (Example Posts for Style Reference):\n"
        "   --- Example 1 ---\n"
        f"   {example_post_1}\n"
        "   --- Example 2 ---\n"
        f"   {example_post_2}\n"
        "   --- End Examples ---\n\n"
        "▶️ 생성할 게시물 (Your turn - Generate the post based on the context above):"
    )
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[
                {"role": "system", "content": "뛰어난 한국어 소셜 마케터로서, 제공된 컨텍스트로 흥미로운 게시물과 관련 해시태그를 생성합니다. 게시물+해시태그만 출력."},
                {"role": "user", "content": prompt_content}], temperature=0.7)
        post_text = response.choices[0].message.content.strip()
        if not post_text: logging.warning("LLM empty post."); return "-- Empty post --"
        if '#' not in post_text[-100:]: logging.warning("Post missing hashtags?.")
        logging.info(f"Generated social post:\n---\n{post_text}\n---"); return post_text
    except APIError as e: logging.error(f"API error post: {e}"); return "-- Error (API) --"
    except Exception as e: logging.error(f"Error post: {e}"); logging.exception("Traceback:"); return "-- Error --"

# --- Main Orchestration Loop ---
def main_processing_loop(urls_to_process):
    # (Function remains unchanged)
    total_urls = len(urls_to_process); logging.info(f"Starting processing for {total_urls} URLs from {URL_INPUT_FILE}.")
    client = None
    try:
        api_key = os.getenv("OPENAI_API_KEY");
        if not api_key: raise ValueError("Missing OpenAI API Key.")
        logging.info("Initializing OpenAI client..."); client = OpenAI(api_key=api_key)
        logging.info("Testing OpenAI connection..."); client.models.list(); logging.info("OpenAI client OK.")
    except Exception as e: logging.critical(f"CRITICAL: OpenAI Client Error: {e}. Cannot process."); return

    for i, youtube_url in enumerate(urls_to_process):
        url_start_time = time.time(); logging.info(f"--- Processing URL {i+1}/{total_urls}: {youtube_url} ---")
        vid_id = youtube_url.split('v=')[-1].split('&')[0]; base_filename = f"video_{vid_id}_{int(url_start_time)}"
        original_audio_path = os.path.abspath(os.path.join(TEMP_FOLDER, f"{base_filename}_full_audio.mp3"))
        video_section_path = os.path.abspath(os.path.join(TEMP_FOLDER, f"{base_filename}_section.mp4"))
        final_clip_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, f"{base_filename}_final_subtitled.mp4"))
        social_post_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, f"{base_filename}_social_post.txt"))
        try:
            setup_directories()
            # Phase 1: Download Full Audio
            phase_start_time = time.time(); logging.info("--- Phase 1: Download Full Audio ---")
            if not download_full_audio(youtube_url, original_audio_path): raise Exception("Phase 1 Fail: Audio DL")
            logging.info(f"--- Phase 1 Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 2: Chunk Audio
            phase_start_time = time.time(); logging.info("--- Phase 2: Chunk Audio ---")
            audio_chunk_paths = chunk_audio(original_audio_path, AUDIO_CHUNKS_FOLDER)
            if audio_chunk_paths is None: raise Exception("Phase 2 Fail: Chunking")
            logging.info(f"--- Phase 2 Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 3: Transcribe Audio
            phase_start_time = time.time(); logging.info("--- Phase 3: Transcribe Audio ---")
            all_segments_english, full_transcript_english = transcribe_chunks(audio_chunk_paths, client)
            if all_segments_english is None: raise Exception("Phase 3 Fail: Transcription")
            if not all_segments_english: logging.warning("No speech segments. Skipping URL."); continue
            logging.info(f"--- Phase 3 Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 4: Identify Key Segment
            phase_start_time = time.time(); logging.info(f"--- Phase 4: Identify Key Segment ---")
            key_start_time, key_end_time, reasoning = find_key_segment(full_transcript_english, client)
            if key_start_time is None or key_end_time is None: raise Exception("Phase 4 Fail: Key Segment ID")
            logging.info(f"--- Phase 4 Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 5: Download Video Section
            phase_start_time = time.time(); logging.info(f"--- Phase 5: Download Video Section ---")
            if not download_video_section(youtube_url, key_start_time, key_end_time, video_section_path): raise Exception("Phase 5 Fail: Video Section DL")
            logging.info(f"--- Phase 5 Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 6a: Filter & Translate Clip Segments
            phase_start_time = time.time(); logging.info("--- Phase 6a: Filter & Translate Clip Segments ---")
            english_segments = [seg for seg in all_segments_english if seg['start'] < key_end_time and seg['end'] > key_start_time]
            korean_segments_for_clip = []; failures = 0
            logging.info(f"Found {len(english_segments)} overlapping segments. Translating...");
            for idx, segment in enumerate(english_segments):
                target = segment.get('text', ''); k_text = target
                if target and not target.isspace():
                    prev = english_segments[idx-1].get('text') if idx > 0 else None
                    next_ = english_segments[idx+1].get('text') if idx < len(english_segments) - 1 else None
                    k_text = translate_text_to_korean(target, client, prev_context=prev, next_context=next_)
                    if k_text is None: k_text = target; failures += 1; logging.error(f"Translation None seg {idx+1}")
                    elif k_text == target: failures += 1; logging.warning(f"Translation fallback seg {idx+1}")
                new_seg = segment.copy(); new_seg['text'] = k_text; korean_segments_for_clip.append(new_seg)
            logging.info(f"Finished translating. {failures} failures/fallbacks.")
            logging.info(f"--- Phase 6a Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 6b: Add Subtitles to Video Section
            phase_start_time = time.time(); logging.info("--- Phase 6b: Add Subtitles to Video Section ---")
            subtitle_success = create_subtitled_video(video_section_path, korean_segments_for_clip, key_start_time, final_clip_path)
            if not subtitle_success: logging.warning("Failed to create final subtitled video.")
            logging.info(f"--- Phase 6b Finished ({time.time() - phase_start_time:.2f}s) ---")
            # Phase 7: Generate Social Post
            phase_start_time = time.time(); logging.info("--- Phase 7: Generate Social Post ---")
            social_post_text = generate_social_post(korean_segments_for_clip, reasoning, client)
            try:
                with open(social_post_path, 'w', encoding='utf-8') as f: f.write(social_post_text)
                logging.info(f"Social post saved: {social_post_path}")
            except IOError as e: logging.error(f"Failed to write social post: {e}")
            logging.info(f"--- Phase 7 Finished ({time.time() - phase_start_time:.2f}s) ---")
            logging.info(f"--- Successfully finished URL {i+1}/{total_urls} ---")
            logging.info(f"Video: {final_clip_path if subtitle_success and os.path.exists(final_clip_path) else '(Failed/Not Found)'}")
            logging.info(f"Post: {social_post_path if os.path.exists(social_post_path) else '(Not Found)'}")
        except Exception as e:
            logging.error(f"--- FAILED processing URL {i+1}/{total_urls}: {youtube_url} ---")
            logging.error(f"Error: {e}"); logging.exception("Traceback:")
        finally:
            cleanup([TEMP_FOLDER]) # Clean temp folder after each URL
            logging.info(f"--- URL {i+1} finished/failed in {time.time() - url_start_time:.2f}s ---")
            if i < total_urls - 1 and DELAY_BETWEEN_VIDEOS_SECONDS > 0:
                logging.info(f"Waiting {DELAY_BETWEEN_VIDEOS_SECONDS}s..."); time.sleep(DELAY_BETWEEN_VIDEOS_SECONDS)
    logging.info(f"--- Finished processing all {total_urls} URLs ---")

# --- Main Entry Point ---
if __name__ == "__main__":
    script_start_time = time.time(); logging.info("Script execution started.")
    print("Performing pre-flight checks...")
    deps_ok = True
    if not shutil.which('yt-dlp'): logging.critical("❌ Missing: yt-dlp"); deps_ok = False
    if not shutil.which('ffmpeg'): logging.critical("❌ Missing: ffmpeg"); deps_ok = False
    if not os.path.exists(SUBTITLE_FONT): logging.critical(f"❌ Missing Font: {os.path.abspath(SUBTITLE_FONT)}"); deps_ok = False
    if not deps_ok: print("🚨 Critical dependencies missing. Exiting."); exit(1)
    print(f"✅ Dependencies and font OK."); logging.info("Dependencies checked successfully.")
    urls_to_process = []
    if not os.path.exists(URL_INPUT_FILE): logging.critical(f"❌ Input file '{URL_INPUT_FILE}' not found. Create it with URLs. Exiting."); exit(1)
    try:
        logging.info(f"Reading URLs from '{URL_INPUT_FILE}'..."); count = 0
        with open(URL_INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'): urls_to_process.append(url); count += 1
        logging.info(f"Found {count} URLs to process.")
    except Exception as e: logging.critical(f"❌ Error reading '{URL_INPUT_FILE}': {e}. Exiting."); exit(1)
    if not urls_to_process: logging.warning(f"No valid URLs found in '{URL_INPUT_FILE}'. Exiting."); exit(0)
    load_dotenv(); main_processing_loop(urls_to_process)
    logging.info(f"Script execution finished. Total wall time: {time.time() - script_start_time:.2f} seconds.")