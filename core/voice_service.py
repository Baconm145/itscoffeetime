from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import edge_tts
from faster_whisper import WhisperModel


class VoiceService:
    """
    Сервис для:
    - распознавания речи из аудиофайла
    - озвучивания текста в аудиофайл
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        tts_voice: str = "ru-RU-SvetlanaNeural",
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
            language="ru",
            vad_filter=True,
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

    def synthesize_to_mp3_sync(
        self,
        text: str,
        output_path: str | Path,
    ) -> Optional[Path]:
        """
        Синхронная обёртка над async TTS.
        """
        return asyncio.run(self.synthesize_to_mp3(text, output_path))