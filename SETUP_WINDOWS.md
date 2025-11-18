# Быстрая установка на Windows

## Шаг 1: Установка зависимостей Python

```powershell
# Убедитесь, что виртуальное окружение активировано
# Если нет, создайте и активируйте:
python -m venv venv
.\venv\Scripts\Activate.ps1

# Установите зависимости
pip install -r requirements.txt
```

## Шаг 2: Установка PyTorch с CUDA (для GPU)

Если у вас есть GPU (например, RTX 4090):

```powershell
# Проверьте версию CUDA
nvidia-smi

# Установите PyTorch с CUDA (замените cu118 на вашу версию CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Или используйте официальный установщик: https://pytorch.org/get-started/locally/

## Шаг 3: Установка FFmpeg

### Вариант 1: Через Chocolatey (рекомендуется)

```powershell
# Установите Chocolatey (если еще не установлен)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Установите FFmpeg
choco install ffmpeg
```

### Вариант 2: Вручную

1. Скачайте FFmpeg: https://ffmpeg.org/download.html
2. Распакуйте архив
3. Добавьте папку `bin` в PATH:
   ```powershell
   # Временно (для текущей сессии)
   $env:PATH += ";C:\path\to\ffmpeg\bin"
   
   # Постоянно (через системные настройки)
   # Система -> Переменные среды -> PATH -> Добавить путь к bin
   ```

### Проверка установки FFmpeg:

```powershell
ffmpeg -version
```

## Шаг 4: Установка openSMILE

См. [INSTALL_OPENSMILE_WINDOWS.md](INSTALL_OPENSMILE_WINDOWS.md)

**Быстрый способ:** Используйте готовые бинарники с GitHub Releases.

## Шаг 5: Проверка установки

```powershell
# Проверьте Python зависимости
python -c "import whisperx; print('whisperx OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Проверьте FFmpeg
ffmpeg -version

# Проверьте openSMILE (если установлен)
.\opensmile\bin\SMILExtract.exe -h
```

## Решение проблем

### Ошибка: "ModuleNotFoundError: No module named 'whisperx'"

**Решение:**
```powershell
pip install whisperx>=3.1.1
```

Или установите все зависимости:
```powershell
pip install -r requirements.txt
```

### Ошибка: "ffmpeg не найден"

**Решение:** Установите FFmpeg (см. Шаг 3 выше)

### Ошибка: "CUDA not available" (для GPU)

**Решение:**
1. Установите PyTorch с CUDA (см. Шаг 2)
2. Проверьте: `python -c "import torch; print(torch.cuda.is_available())"`

### Ошибка при установке зависимостей

**Решение:**
```powershell
# Обновите pip
python -m pip install --upgrade pip

# Установите зависимости по одной
pip install whisperx
pip install torch torchvision torchaudio
pip install pandas numpy scipy
# и т.д.
```

## Полная последовательность установки

```powershell
# 1. Создайте виртуальное окружение (если еще не создано)
python -m venv venv

# 2. Активируйте виртуальное окружение
.\venv\Scripts\Activate.ps1

# 3. Обновите pip
python -m pip install --upgrade pip

# 4. Установите PyTorch с CUDA (для GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Установите остальные зависимости
pip install -r requirements.txt

# 6. Установите FFmpeg (через Chocolatey или вручную)
choco install ffmpeg

# 7. Проверьте установку
python -c "import whisperx; import torch; print('Все зависимости установлены!')"
```

## Готово!

После установки всех зависимостей можно запускать пайплайн:

```powershell
python pipeline/run_full_pipeline.py
```

Для GPU:
```powershell
python pipeline/run_full_pipeline.py --whisper-device cuda
```

