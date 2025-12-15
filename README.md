## Быстрый старт

### 1. Создание виртуального окружения

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

### 2. Установка зависимостей

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

### 3. Запуск веб-приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`

## Требования

- Python 3.8+
- CUDA 12.4 (для GPU)
- 8GB+ RAM
- 2GB+ свободного места на диске

## Возможные проблемы

### Ошибка "CUDA out of memory"

Уменьшите `BATCH_SIZE` в `src/config.py`

### Ошибка "FileNotFoundError: models/best_model.pth"

Сначала запустите обучение: `python train.py`

### Медленное обучение

- Проверьте, что используется GPU: модель выведет "Using device: cuda"
- Увеличьте `NUM_WORKERS` в `src/config.py` (по умолчанию 2)
