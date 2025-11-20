# Решение проблем на Windows

## Проблема: "ModuleNotFoundError: No module named 'whisperx'" при установленном пакете

### Причина:
Python использует другое окружение или виртуальное окружение не активировано правильно.

### Решение:

#### 1. Проверьте, какое окружение используется:

```powershell
# Проверьте путь к Python
python -c "import sys; print(sys.executable)"

# Должен быть путь к venv:
# C:\Users\...\opensmile-whisperX\venv\Scripts\python.exe
```

#### 2. Убедитесь, что виртуальное окружение активировано:

```powershell
# Деактивируйте текущее (если есть)
deactivate

# Активируйте заново
.\venv\Scripts\Activate.ps1

# Проверьте, что вы в venv (должно быть (venv) в начале строки)
```

#### 3. Если ActivationPolicy блокирует скрипты:

```powershell
# Разрешите выполнение скриптов
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Затем активируйте venv
.\venv\Scripts\Activate.ps1
```

#### 4. Проверьте установку пакетов:

```powershell
# Проверьте список установленных пакетов
pip list | findstr whisperx

# Проверьте импорт
python -c "import whisperx; print(whisperx.__version__)"
```

#### 5. Переустановите пакет в правильном окружении:

```powershell
# Убедитесь, что venv активирован
# (должно быть (venv) в начале строки)

# Переустановите whisperx
pip uninstall whisperx -y
pip install whisperx>=3.1.1

# Или установите все зависимости заново
pip install -r requirements.txt
```

#### 6. Если ничего не помогает - пересоздайте виртуальное окружение:

```powershell
# Деактивируйте текущее окружение
deactivate

# Удалите старое окружение
Remove-Item -Recurse -Force venv

# Создайте новое
python -m venv venv

# Активируйте
.\venv\Scripts\Activate.ps1

# Обновите pip
python -m pip install --upgrade pip

# Установите зависимости
pip install -r requirements.txt
```

---

## Проблема: FFmpeg не найден

### Решение:

#### Вариант 1: Через Chocolatey

```powershell
# Установите Chocolatey (если еще не установлен)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Установите FFmpeg
choco install ffmpeg

# Перезапустите PowerShell после установки
```

#### Вариант 2: Вручную

1. Скачайте FFmpeg: https://www.gyan.dev/ffmpeg/builds/ (Windows builds)
2. Распакуйте архив (например, в `C:\ffmpeg`)
3. Добавьте в PATH:
   - Откройте "Переменные среды" (System Properties -> Environment Variables)
   - Добавьте `C:\ffmpeg\bin` в переменную PATH
   - Перезапустите PowerShell

#### Проверка:

```powershell
ffmpeg -version
```

---

## Проблема: Python использует системный Python вместо venv

### Решение:

```powershell
# Проверьте, какой Python используется
where python

# Должен быть путь к venv:
# C:\Users\...\opensmile-whisperX\venv\Scripts\python.exe

# Если нет, активируйте venv правильно:
.\venv\Scripts\Activate.ps1

# Или используйте полный путь:
.\venv\Scripts\python.exe pipeline/run_full_pipeline.py
```

---

## Проблема: Ошибка при активации venv

### Ошибка: "cannot be loaded because running scripts is disabled"

### Решение:

```powershell
# Разрешите выполнение скриптов (только для текущего пользователя)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Затем активируйте venv
.\venv\Scripts\Activate.ps1
```

---

## Полная последовательность для исправления проблем:

```powershell
# 1. Откройте PowerShell в корне проекта
cd C:\Users\proje pc 13900k\Desktop\madi_olzhas\opensmile-whisperX

# 2. Разрешите выполнение скриптов
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Активируйте виртуальное окружение
.\venv\Scripts\Activate.ps1

# 4. Проверьте, что вы в venv (должно быть (venv) в начале)
# 5. Проверьте Python
python -c "import sys; print(sys.executable)"

# 6. Установите/переустановите зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 7. Проверьте установку
python -c "import whisperx; print('whisperx OK')"

# 8. Запустите пайплайн
python pipeline/run_full_pipeline.py --skip-training
```

---

## Проверка готовности системы:

Выполните все проверки:

```powershell
# 1. Виртуальное окружение активировано
# (должно быть (venv) в начале строки)

# 2. Python из venv
python -c "import sys; print('Python:', sys.executable)"

# 3. WhisperX установлен
python -c "import whisperx; print('WhisperX:', whisperx.__version__)"

# 4. PyTorch установлен
python -c "import torch; print('PyTorch:', torch.__version__)"

# 5. FFmpeg доступен
ffmpeg -version

# 6. Все зависимости
pip list
```

Если все проверки пройдены, пайплайн должен работать!

