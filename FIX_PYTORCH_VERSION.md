# Исправление конфликта версий PyTorch и WhisperX

## Проблема

WhisperX 3.7.4 требует `torch~=2.8.0`, но установлена версия 2.7.1.

## Решение

### Вариант 1: Установить PyTorch 2.8.0 с CUDA (рекомендуется для GPU)

```powershell
# Проверьте версию CUDA
nvidia-smi

# Для CUDA 11.8:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu118

# Для CUDA 12.1:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Для CUDA 12.4 или 12.6 (обратно совместимо):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124
```

**Примечание:** CUDA 12.6 обратно совместима с CUDA 12.4, поэтому используйте `cu124`.

### Вариант 2: Использовать официальный установщик PyTorch

1. Перейдите на https://pytorch.org/get-started/locally/
2. Выберите:
   - OS: Windows
   - Package: Pip
   - Language: Python
   - Compute Platform: CUDA 11.8 (или ваша версия)
3. Скопируйте команду и выполните в PowerShell

### Вариант 3: Для CPU (без GPU)

```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

## Проверка установки

После установки проверьте:

```powershell
# Проверьте версии
python -c "import torch; import torchaudio; print(f'PyTorch: {torch.__version__}'); print(f'TorchAudio: {torchaudio.__version__}')"

# Проверьте CUDA (для GPU)
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Проверьте совместимость с WhisperX
python -c "import whisperx; import torch; print(f'WhisperX: {whisperx.__version__}'); print(f'PyTorch: {torch.__version__}'); print('Совместимость: OK')"
```

## Если проблемы с установкой

### Ошибка: "No matching distribution found"

Попробуйте установить без указания конкретной версии CUDA:

```powershell
# Установите последнюю версию PyTorch 2.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

Затем проверьте, какая версия CUDA установлена, и установите соответствующую версию.

### Ошибка: Конфликт зависимостей

```powershell
# Удалите все версии PyTorch
pip uninstall torch torchvision torchaudio -y

# Установите правильную версию
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu118
```

## Для RTX 4090 (CUDA 12.x)

RTX 4090 обычно работает с CUDA 12.x. Проверьте версию:

```powershell
nvidia-smi
```

**Для CUDA 12.1-12.3:**
```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121
```

**Для CUDA 12.4-12.6 (рекомендуется):**
```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124
```

CUDA 12.6 обратно совместима с CUDA 12.4, поэтому `cu124` будет работать.

## Полная последовательность для исправления

```powershell
# 1. Убедитесь, что venv активирован
.\venv\Scripts\Activate.ps1

# 2. Удалите старые версии PyTorch
pip uninstall torch torchvision torchaudio -y

# 3. Проверьте версию CUDA
nvidia-smi

# 4. Установите PyTorch 2.8.0 с правильной версией CUDA
# Для CUDA 11.8:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu118
# Для CUDA 12.4-12.6:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124

# 5. Проверьте установку
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 6. Проверьте совместимость с WhisperX
python -c "import whisperx; import torch; print('Все OK!')"

# 7. Запустите пайплайн
python pipeline/run_full_pipeline.py --skip-training
```

