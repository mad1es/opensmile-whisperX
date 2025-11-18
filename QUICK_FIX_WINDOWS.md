# Быстрое решение проблем на Windows

## Проблема: "cmake не найден" и "make не найден"

### ✅ Правильное решение:

На Windows **НЕ используйте** команды `cmake` и `make` вручную! Используйте готовый скрипт.

### Шаг 1: Установите CMake

1. Скачайте CMake: https://cmake.org/download/
2. Выберите "Windows x64 Installer"
3. При установке **обязательно** выберите "Add CMake to system PATH"
4. Перезапустите PowerShell

### Шаг 2: Правильная сборка

```powershell
# 1. Перейдите в папку opensmile (НЕ в build!)
cd C:\Users\proje pc 13900k\Desktop\madi_olzhas\opensmile-whisperX\opensmile

# 2. Запустите скрипт сборки (он все сделает сам)
powershell -ExecutionPolicy Bypass -File build.ps1
```

**ВАЖНО:**
- ❌ НЕ запускайте `cmake ..` вручную
- ❌ НЕ запускайте `make` вручную  
- ✅ Используйте `build.ps1` - он автоматически создаст папку build и настроит все

### Шаг 3: Проверка

После сборки проверьте:
```powershell
.\build\progsrc\smilextract\Release\SMILExtract.exe -h
```

---

## Альтернатива: Используйте готовые бинарники (рекомендуется)

Если сборка вызывает проблемы, используйте готовые бинарники:

1. Скачайте с https://github.com/audeering/opensmile/releases
2. Найдите `opensmile-3.0.0-win64.zip` (или последнюю версию)
3. Распакуйте в папку `opensmile/` проекта
4. Убедитесь, что `SMILExtract.exe` находится в `opensmile/bin/` или `opensmile/build/progsrc/smilextract/Release/`

---

## Если все еще не работает

### Проверьте установку CMake:
```powershell
cmake --version
```

Если команда не работает:
1. Переустановите CMake с опцией "Add to PATH"
2. Перезапустите PowerShell
3. Проверьте переменную PATH: `$env:PATH`

### Проверьте Visual Studio:
```powershell
# Откройте Visual Studio Developer Command Prompt и попробуйте там
```

Или установите Visual Studio Build Tools:
- Скачайте: https://visualstudio.microsoft.com/downloads/
- Выберите "Build Tools for Visual Studio"
- Установите "C++ build tools"

---

## Резюме правильных команд:

```powershell
# 1. Установите CMake (если еще не установлен)
# Скачайте с https://cmake.org/download/

# 2. Перейдите в папку opensmile
cd opensmile

# 3. Запустите скрипт (НЕ cmake и НЕ make!)
powershell -ExecutionPolicy Bypass -File build.ps1

# 4. Дождитесь завершения (10-30 минут)

# 5. Проверьте результат
.\build\progsrc\smilextract\Release\SMILExtract.exe -h
```

