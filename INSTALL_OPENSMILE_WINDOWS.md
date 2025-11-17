# Установка openSMILE для Windows

## Способ 1: Использование готовых бинарников (рекомендуется)

Самый простой способ - скачать готовый бинарник с GitHub Releases.

### Шаги:

1. **Скачайте последнюю версию openSMILE для Windows:**
   - Перейдите на [GitHub Releases](https://github.com/audeering/opensmile/releases)
   - Найдите последний релиз (например, `opensmile-3.0.0-win64.zip`)
   - Скачайте архив для Windows (x64)

2. **Распакуйте архив:**
   ```powershell
   # Распакуйте в папку opensmile/ проекта
   Expand-Archive -Path opensmile-3.0.0-win64.zip -DestinationPath .
   ```

3. **Проверьте структуру:**
   После распаковки должна быть структура:
   ```
   opensmile/
   ├── bin/
   │   └── SMILExtract.exe
   ├── config/
   │   └── egemaps/
   │       └── v02/
   │           └── eGeMAPSv02.conf
   └── ...
   ```

4. **Проверьте работу:**
   ```powershell
   .\opensmile\bin\SMILExtract.exe -h
   ```

5. **Обновите путь в коде:**
   Код автоматически найдет `SMILExtract.exe` в следующих местах:
   - `opensmile/build/progsrc/smilextract/SMILExtract.exe` (если собрано из исходников)
   - `opensmile/build/bin/SMILExtract.exe` (если собрано из исходников)
   - `opensmile/bin/SMILExtract.exe` (если использованы готовые бинарники)
   - `SMILExtract.exe` (если добавлен в PATH)

---

## Способ 2: Сборка из исходников

Если нужна кастомная сборка или готовые бинарники недоступны.

### Требования:

1. **Visual Studio 2017 или выше** с компонентами C++:
   - Visual Studio Community (бесплатная версия)
   - Убедитесь, что установлены "Desktop development with C++" компоненты

2. **CMake 3.15 или выше:**
   - Скачайте с [официального сайта](https://cmake.org/download/)
   - Установите и добавьте в PATH
   - Или используйте установщик с опцией "Add CMake to system PATH"

3. **PowerShell** (обычно уже установлен в Windows 10/11)

### Шаги сборки:

1. **Откройте PowerShell** в директории проекта:
   ```powershell
   cd opensmile
   ```

2. **Настройте параметры сборки (опционально):**
   Отредактируйте `build_flags.ps1` если нужно:
   ```powershell
   # Откройте build_flags.ps1 в текстовом редакторе
   # По умолчанию сборка настроена правильно
   ```

3. **Запустите сборку:**
   ```powershell
   # Разрешите выполнение скриптов (только один раз)
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   
   # Запустите сборку
   powershell -ExecutionPolicy Bypass -File build.ps1
   ```

4. **Дождитесь завершения сборки:**
   - Сборка может занять 10-30 минут в зависимости от компьютера
   - После завершения бинарник будет в `opensmile/build/progsrc/smilextract/Release/SMILExtract.exe`

5. **Проверьте работу:**
   ```powershell
   .\build\progsrc\smilextract\Release\SMILExtract.exe -h
   ```

### Возможные проблемы при сборке:

**Ошибка: "CMake не найден"**
- Убедитесь, что CMake установлен и добавлен в PATH
- Перезапустите PowerShell после установки CMake

**Ошибка: "Visual Studio не найден"**
- Установите Visual Studio 2017 или выше
- Убедитесь, что установлены компоненты C++

**Ошибка: "ExecutionPolicy"**
- Выполните: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Или используйте: `powershell -ExecutionPolicy Bypass -File build.ps1`

**Ошибка при компиляции:**
- Убедитесь, что используется Visual Studio 2017 или выше
- Проверьте, что установлены все необходимые компоненты C++

---

## Способ 3: Использование vcpkg (для продвинутых пользователей)

Если вы используете vcpkg для управления зависимостями:

1. **Установите vcpkg** (если еще не установлен)

2. **Отредактируйте `build_flags.ps1`:**
   Раскомментируйте и укажите путь к vcpkg:
   ```powershell
   "-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
   ```

3. **Запустите сборку** как в Способе 2

---

## Проверка установки

После установки проверьте работу openSMILE:

```powershell
# Если использовали готовые бинарники
.\opensmile\bin\SMILExtract.exe -h

# Если собрали из исходников
.\opensmile\build\progsrc\smilextract\Release\SMILExtract.exe -h
```

Должна появиться справка по использованию SMILExtract.

---

## Интеграция с проектом

Код проекта автоматически найдет `SMILExtract.exe` в следующих местах:

1. `opensmile/build/progsrc/smilextract/SMILExtract.exe`
2. `opensmile/build/bin/SMILExtract.exe`
3. `opensmile/bin/SMILExtract.exe`
4. `SMILExtract.exe` (если добавлен в PATH)

Если бинарник находится в другом месте, вы можете:
- Добавить его в PATH системы
- Или создать символическую ссылку в одном из ожидаемых мест

---

## Дополнительные ресурсы

- [Официальная документация openSMILE](https://audeering.github.io/opensmile/)
- [GitHub репозиторий](https://github.com/audeering/opensmile)
- [Страница релизов](https://github.com/audeering/opensmile/releases)

