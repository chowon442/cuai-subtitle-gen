from fastapi import APIRouter
import json
import re
import tempfile
import time
from pathlib import Path

import replicate
import soundfile as sf

from app.config import REPLICATE_API_TOKEN

router = APIRouter(
    prefix="/whisper",
    tags=["whisper"],
)


MODEL_VERSION = "jacksoby/whisperx:b484c1fc8bb7096df7fea8c9628adee66cedc6088d1cbcc56a72674df05c5c24"
AUDIO_PATH = Path("app/assets/E_EA_10001.wav")
OUTPUT_DIR = Path("app/assets")
MAX_DURATION_SECONDS = 30  # Replicate 파일 업로드 제한을 피하기 위해 길이 제한


def build_trimmed_wav(path: Path) -> Path:
    """원본 오디오를 모노/길이 제한 후 임시 WAV 파일 경로로 반환."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if MAX_DURATION_SECONDS:
        max_samples = int(MAX_DURATION_SECONDS * sr)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    sf.write(tmp_path, audio, sr, format="WAV")
    return tmp_path


@router.post("/")
async def whisper_endpoint():
    start_time = time.time()
    if not REPLICATE_API_TOKEN:
        raise RuntimeError(
            "Missing REPLICATE_API_TOKEN. Please set it in app.config or environment variables."
        )
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    snippet_path = build_trimmed_wav(AUDIO_PATH)

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    try:
        with snippet_path.open("rb") as audio_file:
            prediction = client.run(
                MODEL_VERSION,
                input={"audio_file": audio_file},
            )
    finally:
        snippet_path.unlink(missing_ok=True)

    segments = prediction.get("segments", []) if isinstance(prediction, dict) else []

    results = []
    for idx, segment in enumerate(segments):
        text = (segment.get("text") or "").strip()
        results.append(
            {
                "idx": idx,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "nbest": [text],
            }
        )

    def clean_cutoff_words_smart(segments):
        cleaned = []
        for i, seg in enumerate(segments):
            text = seg["nbest"][0].strip()
            if text.endswith("-") and i + 1 < len(segments):
                next_text = segments[i + 1]["nbest"][0].strip()
                m = re.search(r"(\w+)-$", text)
                if m:
                    prefix = m.group(1)
                    next_first_word = re.match(r"([A-Za-z]+)", next_text)
                    if next_first_word:
                        candidate = next_first_word.group(1)
                        if candidate.lower().startswith(prefix.lower()):
                            text = re.sub(
                                r"\b" + re.escape(prefix) + r"-$", "", text
                            ).strip()
            seg["nbest"][0] = text
            cleaned.append(seg)
        return cleaned


    cleaned_results = clean_cutoff_words_smart(results)

    # ------------------------
    # 8. JSON 파일 저장
    # ------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = OUTPUT_DIR / "cleaned_results.json"
    llm_input_path = OUTPUT_DIR / "llm_input.json"

    with cleaned_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)

    # LLM 입력용 JSON 변환
    llm_input = []
    for seg in cleaned_results:
        llm_input.append(
            {
                "segment_id": seg["idx"],
                "start": seg["start"],
                "end": seg["end"],
                "n_best": seg["nbest"],
            }
        )

    with llm_input_path.open("w", encoding="utf-8") as f:
        json.dump(llm_input, f, ensure_ascii=False, indent=2)

    elapsed_sec = time.time() - start_time
    return {"message": llm_input, "elapsed_sec": elapsed_sec}
