# Entropy and Complexity Analysis

Анализ респираторных звуков с использованием энтропии и статистической сложности.

## 🎯 Архитектура проекта

**Google Colab** - Работа с большим датасетом (3.69GB), обучение моделей
**VS Code** - Разработка функций, работа с обученными моделями

## 📁 Структура проекта

```
course_paper/
├── app/                    # 🧠 Основные функции анализа
│   ├── entropy_complexity.py  # Расчет энтропии и сложности
│   ├── feature_extraction.py  # Извлечение признаков
│   └── data_helpers.py     # Работа с данными и признаками
│
├── colab/
│   └── colab_notebook.ipynb    # 🚀 Всё для Colab: загрузка данных, анализ, визуализация
├── COLAB_GUIDE.md          # 📖 Подробная инструкция для Colab
└── main.py                 # 💻 Локальный запуск (показывает доступные функции)
```

**Философия:**
- `app/` - функции анализа (используются локально и в Colab)
- `colab/colab_notebook.ipynb` - содержит ВСЁ: загрузку данных, анализ, визуализацию
- Датасет остается в Colab (кэшируется в Google Drive), результаты скачиваются в VS Code

## 💾 Кэширование в Google Drive

**Датасет и kaggle.json автоматически сохраняются в Google Drive!**

При первом запуске `colab/colab_notebook.ipynb`:
- Вы загружаете `kaggle.json` → сохраняется в Drive
- Датасет (3.69GB) загружается с Kaggle → сохраняется в Drive

При следующих запусках:
- `kaggle.json` берется из Drive автоматически
- Датасет копируется из Drive за 1-2 минуты (вместо 5-10 минут)

**Преимущества:**
- ⚡ Не нужно каждый раз загружать kaggle.json
- ⚡ Быстрый старт при повторных запусках датасета
- 💰 Экономия квоты Kaggle API
- 🔄 Работает между разными Colab сессиями

---

## �🚀 Быстрый старт

### Вариант 1: Google Colab (Рекомендуется)

1. **Откройте ноутбук в Colab:**
   - Загрузите `colab_notebook.ipynb` в Google Colab
   - Или откройте напрямую: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/incRED1bl/course_paper/blob/main/colab_notebook.ipynb)

2. **Следуйте инструкциям в ноутбуке:**
   - Настройка Kaggle API
   - Загрузка датасета (3.69GB)
   - Анализ данных
   - Обучение модели
   - Сохранение результатов

3. **Скачайте результаты:**
   - `features.pkl` - извлеченные признаки
   - `model.pkl` - обученная модель

### Вариант 2: Локальная разработка

```bash
# Установка зависимостей
poetry install

# Запуск (только информация, без датасета)
poetry run python main.py
```

## 📦 Зависимости

```bash
# Основные пакеты
poetry add numpy scipy matplotlib

# Для работы в Colab
pip install kaggle
```

## 🔑 Настройка Kaggle (для Colab)

1. Зайдите на https://www.kaggle.com/settings
2. Нажмите "Create New Token" → скачается `kaggle.json`
3. Загрузите файл в Colab (см. инструкции в ноутбуке)

## 💡 Особенности

- ✅ **Датасет остается в Colab** - не нужно качать 3.69GB локально
- ✅ **Все функции готовы к использованию** - импортируйте в Colab
- ✅ **Обученные модели работают локально** - скачайте из Colab в VS Code
- ✅ **Модульная архитектура** - легко добавлять новые функции

## 📊 Dataset

**Respiratory Sound Database** от Kaggle (3.69GB)
- Ссылка: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- Содержит аудиозаписи дыхательных звуков
- Используется для анализа и классификации

## 🎓 Использование

### В Google Colab:

Все функции уже в [colab_notebook.ipynb](colab/colab_notebook.ipynb) - просто запускайте ячейки:

```python
# Загрузка данных (функция в ноутбуке)
signals = load_respiratory_sounds('/content/respiratory_sound_dataset')

# Извлечение признаков
from app.data_helpers import extract_all_features
features_dict, whistle_results, entropy_complexity = extract_all_features(
    signal_data, sample_rate, m=3, tau=1
)

# Визуализация (функции в ноутбуке)
plot_entropy_complexity_plane(disease_features)
```

### В VS Code (с готовыми данными):

```python
import pickle

# Загрузка результатов из Colab
with open('features.pkl', 'rb') as f:
    data = pickle.load(f)

# Работа с моделью
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 🤝 Contributing

Этот проект для курсовой работы. Все функции можно использовать и модифицировать.

## 📝 License

MIT License
