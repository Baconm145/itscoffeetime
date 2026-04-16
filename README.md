# PythonProject

Telegram-бот на Python для свободного диалога и подбора кофемашин. Проект использует:

- intent-классификатор на `scikit-learn`
- smalltalk-датасет с готовыми парами вопрос-ответ
- сценарную логику диалога
- обработку голосовых сообщений через `faster-whisper`
- озвучивание ответов через `edge-tts`

## Что есть в проекте

- [main.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/main.py) — точка входа
- [config.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/config.py) — основные настройки и пути
- [token_config.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/token_config.py) — токен Telegram-бота
- [data](/C:/Users/Baconm145/PycharmProjects/PythonProject/data) — JSON-данные для диалога
- [ml/artifacts](/C:/Users/Baconm145/PycharmProjects/PythonProject/ml/artifacts) — обученные артефакты модели
- [scripts](/C:/Users/Baconm145/PycharmProjects/PythonProject/scripts) — скрипты подготовки датасетов

## Требования

- Python 3.12
- `pip`
- `ffmpeg`

`ffmpeg` нужен для голосовых сообщений.

## Быстрый запуск локально

### 1. Создать виртуальное окружение

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Установить зависимости

```powershell
pip install -r requirements.txt
```

### 3. Указать токен бота

Создай token_config.py и укажи Telegram token своего бота:

```python
BOT_TOKEN = "your-telegram-bot-token"
```

### 4. Проверить, что артефакты и данные уже есть

Для старта приложения должны существовать:

- [data/products.json](/C:/Users/Baconm145/PycharmProjects/PythonProject/data/products.json)
- [data/dialogue_routes.json](/C:/Users/Baconm145/PycharmProjects/PythonProject/data/dialogue_routes.json)
- [data/dialogue_texts.json](/C:/Users/Baconm145/PycharmProjects/PythonProject/data/dialogue_texts.json)
- [data/smalltalk_dataset_siberian_persona_chat.json](/C:/Users/Baconm145/PycharmProjects/PythonProject/data/smalltalk_dataset_siberian_persona_chat.json)
- [ml/artifacts/model.pkl](/C:/Users/Baconm145/PycharmProjects/PythonProject/ml/artifacts/model.pkl)
- [ml/artifacts/vectorizer.pkl](/C:/Users/Baconm145/PycharmProjects/PythonProject/ml/artifacts/vectorizer.pkl)
- [ml/artifacts/label_encoder.pkl](/C:/Users/Baconm145/PycharmProjects/PythonProject/ml/artifacts/label_encoder.pkl)

### 5. Запустить бота

```powershell
.\.venv\Scripts\python.exe main.py
```

Если все в порядке, в консоли появится:

```text
Telegram bot is running...
```

## Запуск через Docker

В проекте есть [Dockerfile](/C:/Users/Baconm145/PycharmProjects/PythonProject/Dockerfile).

### Сборка образа

```bash
docker build -t pythonproject-bot .
```

### Запуск контейнера

```bash
docker run --rm pythonproject-bot
```

Важно: сейчас токен лежит в [token_config.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/token_config.py), поэтому при Docker-сборке он попадет в образ. Для реального деплоя безопаснее вынести токен в переменную окружения.

## Обучение модели

Если нужно пересобрать артефакты intent-классификатора:

```powershell
.\.venv\Scripts\python.exe ml\train_model.py
```

Скрипт:

- загружает `data/intents_expanded.json`
- обучает `TfidfVectorizer + LogisticRegression`
- сохраняет артефакты в `ml/artifacts`

## Подготовка датасета smalltalk

В проекте есть скрипт подготовки smalltalk-датасета:

- [scripts/prepare_smalltalk_from_siberian_persona_chat.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/scripts/prepare_smalltalk_from_siberian_persona_chat.py)

Пример запуска:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_smalltalk_from_siberian_persona_chat.py
```

Результат сохраняется в `data/smalltalk_dataset_siberian_persona_chat.json`.

## Структура запуска

При старте [main.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/main.py):

1. Загружает данные из `data/`
2. Загружает обученные ML-артефакты из `ml/artifacts/`
3. Создает `DialogueManager`
4. Создает `VoiceService`
5. Поднимает Telegram polling-бота

## Возможные проблемы

### Не найден `ffmpeg`

Голосовые сообщения могут не работать. Установи `ffmpeg` и проверь, что он доступен из командной строки.

### Не найдены `model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`

Нужно запустить:

```powershell
.\.venv\Scripts\python.exe ml\train_model.py
```

### Не найден smalltalk-датасет

Нужно либо вернуть существующий JSON в `data/`, либо пересобрать датасет:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_smalltalk_from_siberian_persona_chat.py
```

### Бот не запускается из-за токена

Проверь [token_config.py](/C:/Users/Baconm145/PycharmProjects/PythonProject/token_config.py) и убедись, что там указан валидный токен Telegram-бота.

## Команды

Локальный запуск:

```powershell
.\.venv\Scripts\python.exe main.py
```

Обучение модели:

```powershell
.\.venv\Scripts\python.exe ml\train_model.py
```

Подготовка smalltalk-датасета:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_smalltalk_from_siberian_persona_chat.py
```

Сборка Docker:

```bash
docker build -t pythonproject-bot .
```

Запуск Docker:

```bash
docker run --rm pythonproject-bot
```
