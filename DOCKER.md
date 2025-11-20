## Docker Workflow

Этот документ описывает, как запускать мультимодальный пайплайн в контейнере.

### 1. Предварительные требования
- Docker Desktop 4.33+ (или любая актуальная версия Docker Engine)
- 25+ GB свободного места на диске (для моделей, кешей и артефактов)
- (Опционально) NVIDIA GPU + CUDA-драйверы + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) для ускорения WhisperX

### 2. Структура
```
opensmile-whisperX/
├── Dockerfile
├── docker-compose.yml   # CPU и GPU сервисы
├── docker/requirements.txt
├── data/                # Расшаривается в контейнер (результаты/артефакты)
└── cache/               # Общий кэш моделей (HF/transformers/torch)
```

`./data` пробрасывается внутрь контейнера как `/app/data`, а `./cache` — как `/root/.cache`. Любой другой сервис может примонтировать тот же каталог и повторно использовать загруженные модели/чекпоинты.

### 3. Сборка образа
```bash
docker compose build        # по умолчанию собирает CPU-образ
docker compose --profile gpu build    # собирает CUDA-вариант
```

CPU-образ использует `python:3.10-slim` и устанавливает PyTorch из `https://download.pytorch.org/whl/cpu`. GPU-образ собирается с аргументами:

- `BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
- `TORCH_CHANNEL=cu121` (в Dockerfile выбирает правильные CUDA-колёса)

Оба образа включают:
- Python 3.10 + PyTorch 2.4.1 (CPU или CUDA в зависимости от канала)
- WhisperX + Faster-Whisper + SciPy/Sklearn/Streamlit
- FFmpeg и все системные зависимости
- Автоматическую сборку openSMILE (по умолчанию берётся официальный тег `v3.0.1`). Бинарь `SMILExtract` кладётся в `/app/opensmile/bin/SMILExtract`.
- WhisperX закреплён на версии `3.7.4` (требует `faster-whisper>=1.1.1` и `numpy 2.0.x`). Эта связка протестирована с пайплайном и поддерживает новые модели/таймстемпы.

### 4. Запуск контейнера

**CPU режим (дефолт):**
```bash
docker compose up -d pipeline
docker compose exec pipeline bash
```

**GPU режим (если есть NVIDIA):**
```bash
docker compose --profile gpu up -d pipeline-gpu
docker compose --profile gpu exec pipeline-gpu bash
```

> Убедитесь, что `docker info` показывает `nvidia` в списке рантаймов.

Если хотите собрать другой релиз openSMILE, передайте аргумент во время билда (работает для CPU и GPU профилей):

```bash
docker compose build --build-arg OPENSMILE_VERSION=v3.1.0
```

### 5. Запуск пайплайна внутри контейнера
```bash
# Внутри контейнера
python pipeline/run_full_pipeline.py \
    --whisper-device cpu \
    --skip-training
```

Для GPU добавьте `--whisper-device cuda` и другие параметры из README.

### 6. Полезные команды
```bash
# Остановить контейнеры
docker compose down

# Очистить собранный образ
docker image rm opensmile-whisperx:latest

# Логи
docker compose logs -f pipeline
```

### 7. Кеширование моделей
Каталог `./cache` автоматически монтируется в `/root/.cache`, а переменные `HF_HOME` и `TRANSFORMERS_CACHE` указывают внутрь него. Поэтому:

- Hugging Face / Faster-Whisper / PyTorch скачивают модели и checkpoint’ы на хост.
- Другие контейнеры могут использовать тот же кеш, если добавить volume `- ./cache:/root/.cache` (или подкаталоги) и те же переменные окружения.
- При очистке достаточно удалить содержимое `cache/`.

Пример для другого docker-compose сервиса:

```yaml
  other-service:
    image: your/image
    volumes:
      - ./cache:/root/.cache
    environment:
      - HF_HOME=/root/.cache/huggingface
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
```

