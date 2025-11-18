# Быстрый старт на Windows с CUDA 12.6

Краткая инструкция для запуска пайплайна на Windows с использованием CUDA.

## Быстрая установка

```powershell
# 1. Создание и активация venv
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Установка PyTorch с CUDA 12.4 (совместимо с 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Установка WhisperX и зависимостей
pip install whisperx>=3.1.1
pip install -r requirements.txt

# 4. Проверка CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Установка FFmpeg

```powershell
# Через Chocolatey
choco install ffmpeg

# Или скачайте вручную с https://www.gyan.dev/ffmpeg/builds/
```

## Запуск пайплайна

### Для мощных GPU (RTX 3090/4090):

```powershell
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model large `
    --whisper-batch-size 64 `
    --whisper-compute-type float16 `
    --mode server
```

### Для средних GPU (RTX 3060/3070/3080):

```powershell
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model medium `
    --whisper-batch-size 32 `
    --whisper-compute-type float16 `
    --mode server
```

### Для слабых GPU:

```powershell
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model base `
    --whisper-batch-size 16 `
    --whisper-compute-type float16 `
    --mode server
```

### Минимальная команда (автоопределение параметров):

```powershell
python pipeline/run_full_pipeline.py --skip-training --whisper-device cuda --mode server
```

## Полная документация

Подробная инструкция: [RUN_WINDOWS_CUDA.md](RUN_WINDOWS_CUDA.md)

