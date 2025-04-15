# Системные диаграммы

## 1. Диаграмма потоков данных (Data Flow Diagram)
```mermaid
graph TD
    A[Пользователь] -->|Ввод текста| B[Интерфейс ввода]
    B -->|Текст| C[Модуль обработки ввода]
    C -->|Анализ намерения| D[LLM]
    D -->|Выбор действия| E[KnowledgeBaseModule]
    E -->|Обновить динамический граф| F[Динамический граф]
    E -->|Запись события| G[Логи событий]
    E -->|Чтение из статического графа| H[Статический граф]
    F -->|Актуальное состояние| I[Модуль генерации ответа]
    H -->|Справочные данные| I
    G -->|История изменений| I
    I -->|Ответ| J[Интерфейс вывода]
    J -->|Текст| A
```
**Описание:**  
- **Статический граф** содержит неизменяемые данные о мире (например, "Меч — оружие кузнеца Джона").
- **Динамический граф** хранит текущее состояние (например, "Игрок имеет меч").
- **Логи событий** сохраняют историю изменений с временными метками.
- **KnowledgeBaseModule** управляет взаимодействием между всеми компонентами.

---

## 2. Диаграмма классов (Class Diagram)
```mermaid
classDiagram
    class KnowledgeBaseModule {
        -VectorDBModule vector_db
        -GraphDB graph_db
        +search_classic(query: str) List[Dict]
        +search_extended(query: str) List[Dict]
        +search_fast(query: str) List[Dict]
        +search_deep(query: str) List[Dict]
        +update_dynamic_graph(action: Action) bool
        +query_static_graph(entity: str) Dict
    }
    class VectorDBModule {
        +query_names(query: str) Dict
        +query_descriptions(query: str) Dict
    }
    class GraphDB {
        +get_node_with_relationships(name: str) Dict
        +update_dynamic_node(entity: str, relations: Dict) bool
        +get_static_node(entity: str) Dict
        +ping() bool
    }
    class LLM {
        +analyze_intent(text: str) Intent
        +select_action(intent: Intent) Action
    }
    class ResponseGenerator {
        +generate_response(dynamic_data: Dict, static_data: Dict, logs: List) str
    }
    LLM --> KnowledgeBaseModule : использует
    VectorDBModule --> KnowledgeBaseModule : содержит
    GraphDB --> KnowledgeBaseModule : содержит
    KnowledgeBaseModule --> ResponseGenerator : передает данные
```
**Описание:**  
- **KnowledgeBaseModule** интегрирует векторную БД (ChromaDB), графовую БД (Neo4j) и логи.
- **GraphDB** поддерживает методы для работы с **статическим** (справочные данные) и **динамическим** графом (изменяемые состояния).
- **ResponseGenerator** формирует ответы на основе данных из всех источников.

---

## 3. Диаграмма вариантов использования (Use Case Diagram)
```mermaid
graph LR
    subgraph Система
        A[Выполнить действие] 
        B[Получить справочную информацию]
        C[Просмотреть историю]
        D[Сгенерировать ответ]
    end
    U[Пользователь] --> A
    U --> B
    U --> C
    A --> D
    B --> D
    C --> D
```
**Описание:**  
- Пользователь может:
  - Выполнять действия (например, "Взять меч"), обновляющие динамический граф.
  - Запрашивать справочные данные из статического графа.
  - Просматривать историю изменений через логи.
  - Получать ответы, объединяющие данные из всех компонентов.

---

## 4. Диаграмма последовательности (Sequence Diagram)
```mermaid
sequenceDiagram
    participant U as Пользователь
    participant I as Интерфейс ввода
    participant K as KnowledgeBaseModule
    participant V as VectorDB
    participant G as GraphDB
    participant S as Статический граф
    participant D as Динамический граф
    participant L as Логи
    participant R as Модуль генерации ответа
    U->>I: Ввод "Где меч?"
    I->>K: Вызов search_deep("местоположение меча")
    K->>V: query_descriptions("местоположение меча")
    V-->>K: Результаты (например, "Меч в кузнице")
    K->>G: get_node_with_relationships("Меч")
    G->>D: Чтение динамических связей
    D-->>G: Текущая связь: "Меч у игрока"
    G-->>K: Данные: {статус: "у игрока"}
    K->>S: get_static_node("Меч") 
    S-->>K: Описание: "Меч — оружие кузнеца Джона"
    K->>R: Передача данных (статика + динамика)
    R->>U: Ответ: "Меч сейчас у вас. Это оружие кузнеца Джона."
```
**Описание:**  
- Поиск информации о местоположении меча:
  - **Векторная БД** находит описание "Меч в кузнице".
  - **Динамический граф** показывает, что меч сейчас у игрока.
  - **Статический граф** предоставляет описание меча.
  - Ответ объединяет актуальное состояние и справочные данные.

---

## Итог
Диаграммы отражают полную систему с:
1. **Статическим графом** — неизменяемые данные.
2. **Динамическим графом** — текущее состояние мира.
3. **Логами** — история изменений.
4. **KnowledgeBaseModule** — центральный компонент для интеграции данных и генерации ответов.
