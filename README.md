# Разработка системы генерации и управления текстовыми репрезентациями персонажей и миров на основе современных технологий естественной обработки языка.
---

## 🚀 Инструкция по установке

### 1. Клонируйте репозиторий

```bash
git clone <URL_репозитория>
cd <название_проекта>
```

### 2. Создайте и активируйте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

---

## 🧠 Настройка базы данных Neo4j

1. Установите Neo4j (например, через [официальный сайт](https://neo4j.com/download/)).
2. Запустите Neo4j и создайте две базы данных:

   * `staticdb` — для хранения статических знаний (предзагруженные факты и сущности).
   * `dynamicdb` — для хранения динамически создаваемых сущностей и фактов в процессе работы системы.

**Важно:** Названия баз можно изменить, если вы скорректируете конфигурацию в `config.yaml` или в коде проекта.

---

## ⚙️ Настройка конфигурационного файла

В конфигурации `config.yaml` или в `.env` файле укажите:

```yaml
agentgraph:
  generated_mode: True         # Автоматическое создание сущностей на основе чата
  provider: google_genai       # Провайдер модели: 'openai' или 'google_genai'
  load_entities: False         # Не загружать сущности из документов при старте
```

---

## 🧾 Настройка переменных окружения

Создайте `.env` файл в корне проекта и укажите следующие переменные:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Пути
WORKING_DIR=/path/to/project_root
DATA_PATH=/path/to/documents

# LLM конфигурация
RAG_LLM_MODEL=gpt-4
RAG_LLM_MODEL_SUMMARIZER=gpt-3.5-turbo
RAG_BASE_URL=https://api.openai.com/v1

# Ключи API
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_genai_key
```

---

### ✅ Готово!

Теперь можно запускать приложение или использовать CLI-инструменты в проекте.
