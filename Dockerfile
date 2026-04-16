FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY bot ./bot
COPY core ./core
COPY data ./data
COPY loaders ./loaders
COPY ml ./ml
COPY scripts ./scripts
COPY voice ./voice
COPY config.py ./config.py
COPY token_config.py ./token_config.py
COPY main.py ./main.py

CMD ["sh", "-c", "python scripts/expand_intents.py && python scripts/prepare_smalltalk_from_siberian_persona_chat.py && python ml/train_model.py && python main.py"]
