# Запуск проекта на Windows с CUDA 12.6

Полное руководство по настройке и запуску всего пайплайна на Windows с использованием CUDA 12.6 для ускорения обработки.

## Требования

- Windows 10/11 (64-bit)
- NVIDIA GPU с поддержкой CUDA 12.6
- Python 3.10 или 3.11 (рекомендуется 3.11)
- Минимум 16 GB RAM (рекомендуется 32 GB)
- Минимум 20 GB свободного места на диске

## Шаг 1: Установка CUDA 12.6

### 1.1. Проверка текущей версии CUDA

```powershell
nvidia-smi
```

Проверьте версию CUDA в выводе. Если версия 12.6 или выше, можно продолжать.

### 1.2. Установка CUDA Toolkit 12.6 (если не установлен)

1. Скачайте CUDA Toolkit 12.6 с официального сайта NVIDIA:
   https://developer.nvidia.com/cuda-12-6-0-download-archive

2. Выберите:
   - **OS:** Windows
   - **Architecture:** x86_64
   - **Version:** 12.6
   - **Installer Type:** exe (local)

3. Запустите установщик и следуйте инструкциям

4. После установки перезапустите компьютер

5. Проверьте установку:
```powershell
nvcc --version
nvidia-smi
```

## Шаг 2: Создание виртуального окружения

```powershell
# Перейдите в папку проекта
cd C:\path\to\opensmile_with_mywhisper

# Создайте виртуальное окружение
python -m venv venv

# Активируйте его
.\venv\Scripts\Activate.ps1

# Если ошибка ExecutionPolicy, выполните:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# Обновите pip
python -m pip install --upgrade pip
```

## Шаг 3: Установка PyTorch с CUDA 12.6

Для CUDA 12.6 используйте PyTorch с CUDA 12.4 (обратно совместимо) или CUDA 12.1.

### Вариант 1: PyTorch с CUDA 12.4 (рекомендуется для CUDA 12.6)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Вариант 2: PyTorch с CUDA 12.1 (также совместимо)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Вариант 3: Официальный установщик PyTorch

1. Перейдите на: https://pytorch.org/get-started/locally/
2. Выберите:
   - **OS:** Windows
   - **Package:** Pip
   - **Language:** Python
   - **Compute Platform:** CUDA 12.4 или CUDA 12.1
3. Скопируйте команду и выполните в PowerShell

### Проверка установки PyTorch

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Ожидаемый вывод:
- PyTorch: 2.x.x+cu124 (или cu121)
- CUDA доступна: True
- CUDA версия: 12.4 (или 12.1)
- GPU: [Название вашей видеокарты]

## Шаг 4: Установка WhisperX

```powershell
pip install whisperx>=3.1.1
```

**Примечание:** WhisperX может автоматически установить совместимую версию PyTorch, если требуется.

### Проверка WhisperX

```powershell
python -c "import whisperx; import torch; print(f'WhisperX: OK'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Шаг 5: Установка остальных зависимостей

```powershell
pip install -r requirements.txt
```

Если возникают проблемы, установите по одной:

```powershell
pip install ffmpeg-python>=0.2.0
pip install pydub>=0.25.1
pip install librosa>=0.10.0
pip install soundfile>=0.12.1
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0
pip install xgboost>=2.0.0
pip install joblib>=1.3.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install plotly>=5.14.0
pip install streamlit>=1.28.0
pip install tqdm>=4.65.0
pip install python-dotenv>=1.0.0
pip install psutil>=5.9.0
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.0
```

## Шаг 6: Установка FFmpeg

### Вариант 1: Через Chocolatey (рекомендуется)

```powershell
# Установите Chocolatey (если еще не установлен)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Установите FFmpeg
choco install ffmpeg

# Перезапустите PowerShell после установки
```

### Вариант 2: Вручную

1. Скачайте FFmpeg: https://www.gyan.dev/ffmpeg/builds/
2. Распакуйте в `C:\ffmpeg`
3. Добавьте `C:\ffmpeg\bin` в PATH системы:
   - Откройте "Система" → "Дополнительные параметры системы" → "Переменные среды"
   - Найдите переменную `Path` в "Системные переменные"
   - Добавьте `C:\ffmpeg\bin`
4. Перезапустите PowerShell

### Проверка FFmpeg

```powershell
ffmpeg -version
```

## Шаг 7: Установка openSMILE

### Вариант 1: Готовые бинарники (рекомендуется)

1. Скачайте с https://github.com/audeering/opensmile/releases
2. Найдите `opensmile-3.0.0-win64.zip` (или последнюю версию)
3. Распакуйте в папку `opensmile/` проекта
4. Убедитесь, что `SMILExtract.exe` находится в `opensmile/bin/` или `opensmile/build/progsrc/smilextract/Release/`

### Вариант 2: Сборка из исходников

```powershell
# Установите CMake (если еще не установлен)
# Скачайте с https://cmake.org/download/
# При установке выберите "Add CMake to system PATH"

# Перейдите в папку opensmile
cd opensmile

# Запустите сборку
powershell -ExecutionPolicy Bypass -File build.ps1

# После сборки проверьте
.\build\progsrc\smilextract\Release\SMILExtract.exe -h
```

## Шаг 8: Подготовка данных

Убедитесь, что структура данных правильная:

```
data/
├── audio_wav/
│   ├── 0/          # Контрольные аудио файлы (WAV)
│   └── 1/          # Суицидные аудио файлы (WAV)
└── metadata.csv    # Опционально: метаданные видео
```

Если у вас есть видео файлы, сначала извлеките аудио:

```powershell
python pipeline/extract_audio.py --input-dir data/raw_videos --output-dir data/audio_wav --sample-rate 16000
```

## Шаг 9: Запуск полного пайплайна с CUDA

### Базовый запуск с GPU:

```powershell
python pipeline/run_full_pipeline.py --skip-training --whisper-device cuda
```

### Оптимальные настройки для мощных GPU (RTX 3090/4090):

```powershell
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model large `
    --whisper-batch-size 64 `
    --whisper-compute-type float16 `
    --mode server
```

### Оптимальные настройки для средних GPU (RTX 3060/3070/3080):

```powershell
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model medium `
    --whisper-batch-size 32 `
    --whisper-compute-type float16 `
    --mode server
```

### Оптимальные настройки для слабых GPU (GTX 1660/RTX 2060):

```powershell
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model base `
    --whisper-batch-size 16 `
    --whisper-compute-type float16 `
    --mode server
```

### Полный запуск с обучением модели:

```powershell
python pipeline/run_full_pipeline.py `
    --whisper-device cuda `
    --whisper-model large `
    --whisper-batch-size 64 `
    --whisper-compute-type float16 `
    --mode server
```

## Параметры запуска

### Основные параметры:

- `--whisper-device cuda` - использовать GPU для транскрипции
- `--whisper-model {tiny|base|small|medium|large}` - модель Whisper
  - `tiny` - самая быстрая, низкая точность
  - `base` - быстрая, средняя точность
  - `small` - средняя скорость, хорошая точность
  - `medium` - медленная, высокая точность (рекомендуется)
  - `large` - очень медленная, максимальная точность
- `--whisper-batch-size N` - размер батча (больше = быстрее, но больше VRAM)
  - RTX 4090: 64-128
  - RTX 3090: 64
  - RTX 3080: 32-64
  - RTX 3070: 32
  - RTX 3060: 16-32
- `--whisper-compute-type {int8|float16|float32}` - тип вычислений
  - `int8` - самое быстрое, но может быть менее точным (только для CPU)
  - `float16` - оптимально для GPU (рекомендуется)
  - `float32` - максимальная точность, но медленнее
- `--mode {lightweight|server}` - режим работы
  - `lightweight` - для слабых систем
  - `server` - для мощных систем (рекомендуется для GPU)

### Пропуск этапов:

- `--skip-transcription` - пропустить транскрипцию
- `--skip-segmentation` - пропустить сегментацию
- `--skip-features` - пропустить извлечение признаков
- `--skip-merge` - пропустить объединение признаков
- `--skip-training` - пропустить обучение модели

## Проверка работы CUDA

### Тест 1: Проверка PyTorch

```powershell
python -c "import torch; print('CUDA доступна:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Тест 2: Проверка WhisperX с CUDA

```powershell
python -c "import whisperx; import torch; device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f'Устройство: {device}'); model = whisperx.load_model('base', device=device, compute_type='float16' if device == 'cuda' else 'int8'); print('WhisperX с CUDA работает!')"
```

### Тест 3: Быстрая транскрипция одного файла

```powershell
python pipeline/transcribe_whisperx.py `
    --input-dir data/audio_wav `
    --output-dir data/transcripts_test `
    --model base `
    --device cuda `
    --compute-type float16 `
    --batch-size 16
```

## Решение проблем

### Ошибка: "CUDA not available"

**Решение:**
1. Проверьте версию CUDA: `nvidia-smi`
2. Убедитесь, что установлен правильный PyTorch:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. Перезапустите PowerShell
4. Проверьте снова: `python -c "import torch; print(torch.cuda.is_available())"`

### Ошибка: "Out of memory" (OOM)

**Решение:**
1. Уменьшите batch_size:
   ```powershell
   --whisper-batch-size 16  # вместо 64
   ```
2. Используйте меньшую модель:
   ```powershell
   --whisper-model medium  # вместо large
   ```
3. Используйте int8 вместо float16 (менее точно):
   ```powershell
   --whisper-compute-type int8
   ```

### Ошибка: "ModuleNotFoundError: No module named 'whisperx'"

**Решение:**
```powershell
# Убедитесь, что venv активирован
.\venv\Scripts\Activate.ps1

# Переустановите WhisperX
pip install --upgrade whisperx>=3.1.1
```

### Ошибка: "ffmpeg не найден"

**Решение:**
```powershell
# Установите через Chocolatey
choco install ffmpeg

# Или добавьте в PATH вручную
```

### Ошибка: "SMILExtract не найден"

**Решение:**
- Используйте готовые бинарники с GitHub Releases
- Или соберите из исходников (см. Шаг 7)

### Медленная работа на GPU

**Проверьте:**
1. Используется ли GPU:
   ```powershell
   nvidia-smi
   ```
   Должна быть видна активность GPU во время выполнения.

2. Правильный compute_type:
   - Для CUDA используйте `float16`, не `int8`

3. Достаточный batch_size:
   - Увеличьте batch_size для лучшей утилизации GPU

## Ожидаемая производительность

### На RTX 4090 (24 GB VRAM):
- **Модель large, batch_size=64:**
  - Скорость: ~30-50x быстрее CPU
  - Обработка 1 часа аудио: ~2-3 минуты
  - Использование VRAM: ~10-14 GB

### На RTX 3090 (24 GB VRAM):
- **Модель large, batch_size=64:**
  - Скорость: ~25-40x быстрее CPU
  - Обработка 1 часа аудио: ~3-5 минут
  - Использование VRAM: ~10-14 GB

### На RTX 3080 (10 GB VRAM):
- **Модель medium, batch_size=32:**
  - Скорость: ~20-30x быстрее CPU
  - Обработка 1 часа аудио: ~5-8 минут
  - Использование VRAM: ~6-8 GB

### На RTX 3070 (8 GB VRAM):
- **Модель medium, batch_size=16:**
  - Скорость: ~15-25x быстрее CPU
  - Обработка 1 часа аудио: ~8-12 минут
  - Использование VRAM: ~4-6 GB

## Полная последовательность команд (быстрый старт)

Скопируйте и выполните все команды последовательно:

```powershell
# 1. Создание и активация venv
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Установка PyTorch с CUDA 12.4 (совместимо с 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Установка WhisperX
pip install whisperx>=3.1.1

# 4. Установка остальных зависимостей
pip install -r requirements.txt

# 5. Проверка установки
python -c "import torch; import whisperx; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print('Все установлено!')"

# 6. Запуск пайплайна с CUDA (настройте параметры под вашу GPU)
python pipeline/run_full_pipeline.py `
    --skip-training `
    --whisper-device cuda `
    --whisper-model large `
    --whisper-batch-size 64 `
    --whisper-compute-type float16 `
    --mode server
```

## Готово!

После выполнения всех шагов пайплайн должен работать на вашей GPU с максимальной производительностью.

Для визуализации результатов запустите:

```powershell
streamlit run visualization_app/app.py
```

