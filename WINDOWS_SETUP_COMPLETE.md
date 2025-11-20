# Полная установка для Windows с RTX 4090 (CUDA 12.6)

## Требования

- Windows 10/11
- RTX 4090
- CUDA 12.6
- Python 3.10 или 3.11 (рекомендуется 3.11)

---

## Шаг 1: Подготовка окружения

### 1.1. Создайте виртуальное окружение

```powershell
# Перейдите в папку проекта
cd C:\Users\proje pc 13900k\Desktop\madi_olzhas\opensmile-whisperX

# Создайте виртуальное окружение
python -m venv venv

# Активируйте его
.\venv\Scripts\Activate.ps1

# Если ошибка ExecutionPolicy, выполните:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### 1.2. Обновите pip

```powershell
python -m pip install --upgrade pip
```

---

## Шаг 2: Установка PyTorch с CUDA 12.6

**ВАЖНО:** Для CUDA 12.6 используйте PyTorch с CUDA 12.4 (обратно совместимо) или официальный установщик.

### Вариант 1: Официальный установщик PyTorch (РЕКОМЕНДУЕТСЯ)

1. Перейдите на: https://pytorch.org/get-started/locally/
2. Выберите:
   - **OS:** Windows
   - **Package:** Pip
   - **Language:** Python
   - **Compute Platform:** CUDA 12.4 (или CUDA 12.1, если 12.4 недоступен)
3. Скопируйте команду и выполните в PowerShell

Обычно команда будет:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Вариант 2: Установка PyTorch для CUDA 12.4 (совместимо с 12.6)

```powershell
# Установите PyTorch с CUDA 12.4 (CUDA 12.6 обратно совместима)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Вариант 3: Установка без указания версии CUDA

```powershell
# Установите PyTorch без указания CUDA (автоматически определит и установит совместимую версию)
pip install torch torchvision torchaudio
```

**Примечание:** Если WhisperX требует PyTorch 2.8.0, он может установить его автоматически при установке WhisperX.

**Проверка:**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести:
- PyTorch: 2.x.x+cu121 (или другая версия)
- CUDA доступна: True
- GPU: NVIDIA GeForce RTX 4090

---

## Шаг 3: Установка WhisperX

```powershell
# Установите WhisperX
pip install whisperx>=3.1.1
```

**ВАЖНО:** Если WhisperX требует PyTorch 2.8.0, но он недоступен, WhisperX может установить совместимую версию PyTorch автоматически. Проверьте после установки.

**Проверка:**
```powershell
python -c "import whisperx; import torch; print(f'WhisperX: OK'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Шаг 4: Установка остальных зависимостей

```powershell
# Установите все зависимости из requirements.txt
pip install -r requirements.txt
```

**Или установите по одной (если requirements.txt не работает):**

```powershell
pip install ffmpeg-python>=0.2.0
pip install pydub>=0.25.1
pip install librosa>=0.10.0
pip install soundfile>=1.12.1
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

---

## Шаг 5: Установка FFmpeg

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
3. Добавьте `C:\ffmpeg\bin` в PATH системы
4. Перезапустите PowerShell

**Проверка:**
```powershell
ffmpeg -version
```

---

## Шаг 6: Установка openSMILE

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

---

## Шаг 7: Финальная проверка

Выполните все проверки:

```powershell
# 1. Виртуальное окружение активировано (должно быть (venv) в начале строки)

# 2. Python из venv
python -c "import sys; print('Python:', sys.executable)"

# 3. PyTorch с CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 4. WhisperX
python -c "import whisperx; print('WhisperX: OK')"

# 5. FFmpeg
ffmpeg -version

# 6. openSMILE (если установлен)
.\opensmile\bin\SMILExtract.exe -h
```

---

## Шаг 8: Запуск пайплайна

### Базовый запуск с GPU:

```powershell
python pipeline/run_full_pipeline.py --skip-training --whisper-device cuda
```

### Оптимальные настройки для RTX 4090:

```powershell
python pipeline/run_full_pipeline.py \
    --skip-training \
    --whisper-device cuda \
    --whisper-model large \
    --whisper-batch-size 64 \
    --whisper-compute-type float16 \
    --mode gpu
```

---

## Полная последовательность команд (скопируйте и выполните)

```powershell
# 1. Создание и активация venv
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Установка PyTorch с CUDA 12.6
# Для CUDA 12.6 используйте CUDA 12.4 (обратно совместимо):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Или используйте официальный установщик: https://pytorch.org/get-started/locally/

# 3. Установка WhisperX
pip install whisperx>=3.1.1

# 4. Установка остальных зависимостей
pip install -r requirements.txt

# 5. Проверка
python -c "import torch; import whisperx; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print('Все установлено!')"

# 6. Запуск пайплайна
python pipeline/run_full_pipeline.py --skip-training --whisper-device cuda --whisper-model large --whisper-batch-size 64 --whisper-compute-type float16 --mode gpu
```

---

## Решение проблем

### Ошибка: "ModuleNotFoundError: No module named 'whisperx'"

**Решение:**
```powershell
# Пересоздайте venv
deactivate
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install whisperx>=3.1.1
pip install -r requirements.txt
```

### Ошибка: "CUDA not available"

**Решение:**
1. Проверьте версию CUDA: `nvidia-smi`
2. Для CUDA 12.6 установите PyTorch с CUDA 12.4 (обратно совместимо):
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. Или используйте официальный установщик: https://pytorch.org/get-started/locally/

### Ошибка: "Could not find a version that satisfies the requirement torch==2.8.0"

**Решение:**
PyTorch 2.8.0 может быть недоступен для вашей версии CUDA. Используйте:
```powershell
# Для CUDA 12.6 используйте CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Или установите без указания версии (WhisperX установит совместимую версию):
```powershell
pip install torch torchvision torchaudio
pip install whisperx>=3.1.1
```
Или используйте официальный установщик PyTorch: https://pytorch.org/get-started/locally/

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
- Или соберите из исходников (см. Шаг 6)

---

## Ожидаемая производительность на RTX 4090

С оптимальными настройками:
- **Скорость транскрипции:** ~10-30x быстрее чем CPU
- **Обработка 1 часа аудио:** ~2-5 минут
- **Использование VRAM:** ~8-12 GB для модели large
- **Batch size:** 64 (оптимально для RTX 4090)

---

## Готово!

После выполнения всех шагов пайплайн должен работать на RTX 4090 с максимальной производительностью.

