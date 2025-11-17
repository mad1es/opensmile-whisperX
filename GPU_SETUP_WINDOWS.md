# Настройка GPU для Windows (RTX 4090)

## Быстрый старт для RTX 4090

Пайплайн автоматически определит RTX 4090 и настроит оптимальные параметры:
- Модель: `large` (лучшая точность)
- Batch size: `64` (максимальная скорость)
- Compute type: `float16` (оптимальный баланс скорости и точности)
- Device: `cuda` (GPU ускорение)

### Просто запустите:

```powershell
python pipeline/run_full_pipeline.py
```

Система автоматически определит GPU и настроит параметры.

---

## Ручная настройка для максимальной производительности

Если хотите явно указать параметры:

```powershell
python pipeline/run_full_pipeline.py \
    --whisper-device cuda \
    --whisper-model large \
    --whisper-batch-size 64 \
    --whisper-compute-type float16 \
    --mode gpu
```

### Параметры для RTX 4090:

- **`--whisper-device cuda`** - использование GPU
- **`--whisper-model large`** - самая точная модель (рекомендуется для RTX 4090)
- **`--whisper-batch-size 64`** - оптимальный размер батча для RTX 4090
- **`--whisper-compute-type float16`** - быстрее чем float32, точнее чем int8
- **`--mode gpu`** - режим для мощных GPU

---

## Требования для GPU

### 1. Установка CUDA и PyTorch с GPU поддержкой

**Важно:** Установите PyTorch с поддержкой CUDA для Windows:

```powershell
# Проверьте версию CUDA на вашей системе
nvidia-smi

# Установите PyTorch с CUDA (замените cu118 на вашу версию CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Или используйте официальный установщик с [pytorch.org](https://pytorch.org/get-started/locally/):
- Выберите: Windows, CUDA 11.8 (или вашу версию), pip

### 2. Проверка установки

```powershell
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести:
```
CUDA доступна: True
GPU: NVIDIA GeForce RTX 4090
```

### 3. Установка WhisperX

```powershell
pip install whisperx>=3.1.1
```

---

## Оптимизация производительности

### Рекомендуемые настройки для RTX 4090:

| Параметр | Значение | Описание |
|----------|----------|----------|
| Модель | `large` | Максимальная точность |
| Batch size | `64` | Оптимально для 24GB VRAM |
| Compute type | `float16` | Баланс скорости и точности |
| Device | `cuda` | GPU ускорение |

### Альтернативные настройки:

**Если не хватает памяти (OOM ошибки):**
```powershell
--whisper-batch-size 32  # Уменьшите batch size
```

**Если нужна максимальная скорость (меньше точность):**
```powershell
--whisper-model medium \
--whisper-batch-size 128 \
--whisper-compute-type float16
```

**Если нужна максимальная точность:**
```powershell
--whisper-model large \
--whisper-batch-size 32 \
--whisper-compute-type float32
```

---

## Мониторинг GPU

Во время работы можно мониторить использование GPU:

```powershell
# В отдельном окне PowerShell
nvidia-smi -l 1
```

Или используйте встроенный мониторинг Python:
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

---

## Решение проблем

### Ошибка: "CUDA out of memory"

**Решение:**
1. Уменьшите batch size:
   ```powershell
   --whisper-batch-size 32
   ```

2. Используйте меньшую модель:
   ```powershell
   --whisper-model medium
   ```

3. Очистите кэш GPU:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Ошибка: "CUDA not available"

**Решение:**
1. Проверьте установку CUDA:
   ```powershell
   nvidia-smi
   ```

2. Переустановите PyTorch с CUDA:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Проверьте версию CUDA:
   ```powershell
   nvcc --version
   ```

### Медленная работа на GPU

**Возможные причины:**
1. Используется CPU вместо GPU - проверьте `--whisper-device cuda`
2. Слишком маленький batch size - увеличьте до 64
3. Используется int8 вместо float16 - используйте `--whisper-compute-type float16`

---

## Ожидаемая производительность на RTX 4090

С оптимальными настройками (large модель, batch_size=64, float16):

- **Скорость транскрипции:** ~10-30x быстрее чем CPU
- **Обработка 1 часа аудио:** ~2-5 минут (зависит от сложности)
- **Использование VRAM:** ~8-12 GB для модели large
- **Точность:** Максимальная (модель large)

---

## Дополнительные оптимизации

### 1. Использование TensorRT (опционально)

Для еще большей скорости можно использовать TensorRT:
```powershell
pip install nvidia-tensorrt
```

### 2. Оптимизация памяти

Добавьте в начало скрипта:
```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### 3. Многопоточная обработка

Для обработки нескольких файлов параллельно можно использовать несколько процессов (осторожно с памятью GPU).

---

## Примеры команд

### Полный пайплайн с GPU:
```powershell
python pipeline/run_full_pipeline.py \
    --whisper-device cuda \
    --whisper-model large \
    --whisper-batch-size 64 \
    --mode gpu
```

### Только транскрипция с GPU:
```powershell
python pipeline/transcribe_whisperx.py \
    --input-dir data/audio_wav \
    --output-dir data/transcripts \
    --device cuda \
    --model large \
    --batch-size 64 \
    --compute-type float16 \
    --mode gpu
```

---

## Проверка готовности

Перед запуском проверьте:

1. ✅ CUDA установлена и работает (`nvidia-smi`)
2. ✅ PyTorch с CUDA установлен (`torch.cuda.is_available()`)
3. ✅ WhisperX установлен (`pip list | findstr whisperx`)
4. ✅ Достаточно места на диске (модели занимают ~3-6 GB)
5. ✅ Достаточно VRAM (рекомендуется 12+ GB для модели large)

После проверки можно запускать пайплайн!

