from __future__ import annotations

from pathlib import Path
from typing import Optional

import edge_tts
from faster_whisper import WhisperModel
from config import (
    TTS_VOICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
    WHISPER_USE_VAD,
)


class VoiceService:
    """
    Сервис для:
    - распознавания речи из аудиофайла
    - озвучивания текста в аудиофайл
    """

    def __init__(
        self,
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        whisper_device: str = WHISPER_DEVICE,
        whisper_compute_type: str = WHISPER_COMPUTE_TYPE,
        tts_voice: str = TTS_VOICE,
    ) -> None:
        self.model = WhisperModel(
            whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        self.tts_voice = tts_voice

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Распознаёт речь из аудиофайла и возвращает текст.
        """
        segments, _info = self.model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            vad_filter=WHISPER_USE_VAD,
        )

        parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)

        return " ".join(parts).strip()

    async def synthesize_to_mp3(
        self,
        text: str,
        output_path: str | Path,
    ) -> Optional[Path]:
        """
        Озвучивает текст и сохраняет MP3.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return None

        communicate = edge_tts.Communicate(
            text=cleaned_text,
            voice=self.tts_voice,
        )
        await communicate.save(str(output_path))
        return Path(output_path)
