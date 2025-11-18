"""
Главный скрипт для запуска полного пайплайна обработки данных.

Автоматически выполняет все этапы обработки от транскрипции аудио
до обучения модели и создания визуализаций.

Ожидаемая структура данных:
- data/audio_wav/0/ - контрольные аудио файлы (WAV)
- data/audio_wav/1/ - суицидные аудио файлы (WAV)

Метки автоматически определяются из структуры папок.
"""

import os
import sys
import argparse
import subprocess
import platform
import psutil
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """
    Выполняет команду и выводит результат.
    
    Args:
        cmd: Список аргументов команды
        description: Описание выполняемой операции
        
    Returns:
        True если успешно, False иначе
    """
    print(f"\n{'='*60}")
    print(f"Выполняется: {description}")
    print(f"Команда: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        print(f"✓ {description} завершено успешно\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при выполнении {description}")
        print(f"Код возврата: {e.returncode}\n")
        return False
    except FileNotFoundError:
        print(f"✗ Команда не найдена: {cmd[0]}")
        print("Убедитесь, что все зависимости установлены\n")
        return False


def detect_environment():
    """Определяет тип окружения."""
    is_mac = platform.system() == 'Darwin'
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    if is_mac or total_memory_gb < 16:
        return {
            'mode': 'lightweight',
            'default_model': 'medium',
            'description': 'MacBook / ограниченная память'
        }
    else:
        return {
            'mode': 'server',
            'default_model': 'medium',
            'description': 'Сервер / мощный компьютер'
        }


def check_dependencies() -> bool:
    """Проверяет наличие необходимых зависимостей."""
    print("Проверка зависимостей...")
    
    dependencies = {
        'python': ['python', '--version'],
        'ffmpeg': ['ffmpeg', '-version'],
    }
    
    all_ok = True
    for name, cmd in dependencies.items():
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True
            )
            print(f"✓ {name} найден")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ {name} не найден")
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Запуск полного пайплайна обработки данных'
    )
    parser.add_argument(
        '--skip-transcription',
        action='store_true',
        help='Пропустить транскрипцию'
    )
    parser.add_argument(
        '--skip-segmentation',
        action='store_true',
        help='Пропустить сегментацию'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Пропустить извлечение признаков'
    )
    parser.add_argument(
        '--skip-merge',
        action='store_true',
        help='Пропустить объединение признаков'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Пропустить обучение модели'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Базовая директория с данными'
    )
    parser.add_argument(
        '--whisper-model',
        type=str,
        default=None,
        help='Модель Whisper для транскрипции (None = автоопределение по окружению)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['lightweight', 'server'],
        help='Режим работы: lightweight (MacBook) или server (мощный сервер). None = автоопределение'
    )
    parser.add_argument(
        '--whisper-language',
        type=str,
        default='ru',
        help='Язык для транскрипции'
    )
    parser.add_argument(
        '--whisper-device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Устройство для Whisper'
    )
    parser.add_argument(
        '--whisper-compute-type',
        type=str,
        default=None,
        choices=['int8', 'float16', 'float32'],
        help='Тип вычислений для Whisper (None = авто: int8 для CPU, float16 для CUDA)'
    )
    parser.add_argument(
        '--whisper-batch-size',
        type=int,
        default=None,
        help='Размер батча для Whisper (None = автоопределение)'
    )
    
    args = parser.parse_args()
    
    env = detect_environment()
    
    # Проверка наличия CUDA для определения оптимального batch_size
    try:
        import torch
        has_cuda = torch.cuda.is_available() and args.whisper_device == 'cuda'
    except ImportError:
        has_cuda = False
    
    if args.mode:
        env['mode'] = args.mode
        if args.mode == 'lightweight':
            env['default_model'] = 'medium'
            env['batch_size'] = 4
        else:
            env['default_model'] = 'medium'
            # Для серверного режима с CUDA используем больший batch_size
            env['batch_size'] = 64 if has_cuda else 16
    
    if args.whisper_model is None:
        args.whisper_model = env['default_model']
    
    print("="*60)
    print("МУЛЬТИМОДАЛЬНЫЙ ПАЙПЛАЙН ОБРАБОТКИ ДАННЫХ")
    print("="*60)
    print(f"Окружение: {env['description']}")
    print(f"Режим: {env['mode']}")
    print(f"Модель Whisper: {args.whisper_model}")
    print(f"Устройство: {args.whisper_device}")
    if args.whisper_compute_type:
        print(f"Compute type: {args.whisper_compute_type}")
    if args.whisper_batch_size:
        print(f"Batch size: {args.whisper_batch_size}")
    elif has_cuda:
        print(f"Batch size: {env['batch_size']} (авто для CUDA)")
    if has_cuda:
        try:
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except:
            pass
    print("="*60)
    
    if not check_dependencies():
        print("\n⚠ Некоторые зависимости не найдены. Продолжить? (y/n)")
        response = input().lower()
        if response != 'y':
            sys.exit(1)
    
    base_dir = Path(args.data_dir)
    pipeline_dir = Path('pipeline')
    
    audio_wav_dir = base_dir / 'audio_wav'
    
    if not audio_wav_dir.exists():
        print(f"Ошибка: директория {audio_wav_dir} не найдена")
        print("Убедитесь, что аудио файлы находятся в data/audio_wav/0/ и data/audio_wav/1/")
        sys.exit(1)
    
    label_dirs = [audio_wav_dir / '0', audio_wav_dir / '1']
    found_labels = [d for d in label_dirs if d.exists() and d.is_dir()]
    
    if not found_labels:
        print(f"Ошибка: не найдены подпапки 0 и 1 в {audio_wav_dir}")
        print("Структура должна быть: data/audio_wav/0/ и data/audio_wav/1/")
        sys.exit(1)
    
    print(f"Найдены метки: {[d.name for d in found_labels]}")
    
    success = True
    
    if not args.skip_transcription:
        transcribe_cmd = [
            'python', str(pipeline_dir / 'transcribe_whisperx.py'),
            '--input-dir', str(base_dir / 'audio_wav'),
            '--output-dir', str(base_dir / 'transcripts'),
            '--model', args.whisper_model,
            '--language', args.whisper_language,
            '--device', args.whisper_device
        ]
        if args.whisper_compute_type:
            transcribe_cmd.extend(['--compute-type', args.whisper_compute_type])
        if args.whisper_batch_size:
            transcribe_cmd.extend(['--batch-size', str(args.whisper_batch_size)])
        if args.mode:
            transcribe_cmd.extend(['--mode', args.mode])
        
        success &= run_command(transcribe_cmd, 'Транскрипция аудио')
    
    if success and not args.skip_segmentation:
        success &= run_command(
            [
                'python', str(pipeline_dir / 'segment_audio.py'),
                '--audio-dir', str(base_dir / 'audio_wav'),
                '--transcript-dir', str(base_dir / 'transcripts'),
                '--output-dir', str(base_dir / 'segments')
            ],
            'Сегментация аудио'
        )
    
    if success and not args.skip_features:
        success &= run_command(
            [
                'python', str(pipeline_dir / 'extract_opensmile_features.py'),
                '--segments-dir', str(base_dir / 'segments'),
                '--output-dir', str(base_dir / 'features'),
                '--output-csv', str(base_dir / 'features' / 'opensmile_features.csv')
            ],
            'Извлечение признаков openSMILE'
        )
    
    if success and not args.skip_merge:
        merge_cmd = [
            'python', str(pipeline_dir / 'merge_features.py'),
            '--segments-metadata', str(base_dir / 'segments' / 'segments_metadata.csv'),
            '--opensmile-features', str(base_dir / 'features' / 'opensmile_features.csv'),
            '--output', str(base_dir / 'features' / 'merged_features.csv'),
            '--language', args.whisper_language,
            '--aggregate'
        ]
        
        metadata_path = base_dir / 'metadata.csv'
        if metadata_path.exists():
            merge_cmd.extend(['--metadata', str(metadata_path)])
            print(f"Найден metadata.csv, будет использован для дополнительных меток")
        
        success &= run_command(
            merge_cmd,
            'Объединение признаков'
        )
    
    if success and not args.skip_training:
        merged_features = base_dir / 'features' / 'merged_features.csv'
        if merged_features.exists():
            success &= run_command(
                [
                    'python', str(pipeline_dir / 'train_baseline.py'),
                    '--data', str(merged_features),
                    '--model-type', 'logistic',
                    '--n-splits', '5'
                ],
                'Обучение базовой модели'
            )
        else:
            print(f"⚠ Файл {merged_features} не найден. Пропуск обучения.")
    
    print("\n" + "="*60)
    if success:
        print("✓ ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО")
        print("="*60)
        print("\nСледующие шаги:")
        print("1. Проверьте результаты в data/features/merged_features.csv")
        print("2. Запустите визуализацию: streamlit run visualization_app/app.py")
    else:
        print("✗ ПАЙПЛАЙН ЗАВЕРШЕН С ОШИБКАМИ")
        print("="*60)
        print("\nПроверьте логи выше для деталей.")
        sys.exit(1)


if __name__ == '__main__':
    main()

