## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone <url_репозитория>
cd ARCHITECT_CLASSIFIER
```

### 2. Создание виртуального окружения

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

**Для GPU (CUDA 12.4):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Для CPU:**

```bash
pip install torch torchvision torchaudio
pip install albumentations streamlit opencv-python numpy Pillow tqdm scikit-learn
```

### 4. Подготовка датасета

Структура папки `dataset`:

```
dataset/
├── train/
│   ├── Classical/
│   ├── Cottage/
│   ├── Gothic/
│   ├── Hitech/
│   ├── Japanese/
│   └── Minimalism/
├── val/
│   └── [те же папки]
└── test/
    └── [те же папки]
```

**Если у вас все изображения в одной папке для каждого класса:**

```bash
python scripts/split_dataset.py
```

## Обучение модели

```bash
python train.py
```

Параметры обучения можно изменить в `src/config.py`:

- `BATCH_SIZE` - размер батча (по умолчанию 16)
- `LEARNING_RATE` - скорость обучения (по умолчанию 1e-4)
- `NUM_EPOCHS` - количество эпох (по умолчанию 20)
- `IMAGE_SIZE` - размер изображения (по умолчанию 256)

Обученная модель сохраняется в `models/best_model.pth`

## Запуск веб-приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`

## Вспомогательные скрипты

### Поиск поврежденных изображений

```bash
python scripts/find_bad_images.py
```

### Поиск дубликатов

```bash
python scripts/find_duplicates.py
```

### Поиск слишком маленьких изображений

```bash
python scripts/find_small_images.py
```

### Разделение датасета на train/val/test

```bash
python scripts/split_dataset.py
```

## Структура проекта

```
ARCHITECT_CLASSIFIER/
├── dataset/              # Датасет с изображениями
├── models/               # Сохраненные модели
│   └── best_model.pth
├── scripts/              # Вспомогательные скрипты
│   ├── find_bad_images.py
│   ├── find_duplicates.py
│   ├── find_small_images.py
│   └── split_dataset.py
├── src/                  # Исходный код
│   ├── config.py         # Конфигурация
│   ├── dataset.py        # Класс датасета
│   └── transforms.py     # Аугментации
├── app.py                # Streamlit приложение
├── train.py              # Скрипт обучения
├── requirements.txt      # Зависимости
└── README.md
```

## Требования

- Python 3.8+
- CUDA 12.4 (для GPU)
- 8GB+ RAM
- 2GB+ свободного места на диске

## Примечания

1. **GPU vs CPU**: Обучение на GPU в ~10-20 раз быстрее
2. **Размер датасета**: Рекомендуется минимум 100 изображений на класс
3. **Аугментация**: Автоматически применяется при обучении
4. **Сохранение модели**: Автоматически сохраняется лучшая модель по validation accuracy

## Возможные проблемы

### Ошибка "CUDA out of memory"

Уменьшите `BATCH_SIZE` в `src/config.py`

### Ошибка "FileNotFoundError: models/best_model.pth"

Сначала запустите обучение: `python train.py`

### Медленное обучение

- Проверьте, что используется GPU: модель выведет "Using device: cuda"
- Увеличьте `NUM_WORKERS` в `src/config.py` (по умолчанию 2)
