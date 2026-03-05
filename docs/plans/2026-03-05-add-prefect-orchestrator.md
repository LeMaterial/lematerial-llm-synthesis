# Интеграция Prefect как оркестратора пайплайнов

Created: 2026-03-05
Status: PENDING
Approved: Yes
Iterations: 1
Worktree: Yes
Type: Feature
Plan-Review: auto-human
Code-Review: auto
Linear: LEM-1

## Резюме

**Цель:** Интегрировать Prefect как оркестратор пайплайнов в проект lematerial-llm-synthesis: обернуть существующие пайплайны экстракции в Prefect flow/tasks, добавить retry-логику и структурированное логирование, сохранив Hydra для конфигурации.

**Архитектура:** Новый модуль `orchestration/` предоставляет `@task`-обёртки для каждого шага пайплайна (загрузка данных, экстракция материалов, экстракция синтеза, оценка, сохранение) и `@flow` для оркестрации. Hydra остаётся слоем конфигурации — Prefect управляет выполнением, ретраями и наблюдаемостью.

**Стек:** Prefect 3.x, DSPy, Hydra, Pydantic, LiteLLM, fsspec

## Скоуп

### В скоупе
- Исследование архитектуры и документ по дизайну интеграции
- Добавление `prefect` как зависимости проекта
- Создание модуля `src/llm_synthesis/orchestration/` с Prefect tasks и flows
- Оборачивание пайплайна экстракции синтеза (экстракция материалов → экстракция синтеза → оценка → сохранение)
- Hydra config-группа для настроек оркестрации (retries, concurrency)
- Интеграция логирования через Prefect (`get_run_logger`)
- Retry-логика на LLM-вызывающих задачах
- Prefect-aware точка входа (entry point скрипт)
- Unit-тесты модуля оркестрации

### Вне скоупа
- Деплой Prefect server/cloud (только локальное выполнение)
- Миграция `SynthesisPerformancePipeline` (линковка performance) — отложено
- Миграция `batch_run_tc_new_snippet.py` и других case study скриптов
- Настройка Prefect UI дашборда
- Запуск по расписанию/cron
- Отдельный `material_extraction_flow` — экстракция материалов реализована как `@task` внутри комбинированного flow, что соответствует текущей архитектуре

## Контекст для имплементатора

> Написано для разработчика, который видит кодовую базу впервые.

### Текущая архитектура

Проект извлекает структурированные процедуры синтеза из научных статей с помощью LLM.

**Основной flow выполнения** (в `examples/scripts/deployment/extract_synthesis_procedure_from_text.py`):
1. **Загрузка статей** — `PaperLoaderInterface.load()` → `list[Paper]` (HuggingFace или файловая система)
2. **Экстракция материалов** — `MaterialExtractorInterface.forward(text)` → строка с названиями материалов через запятую
3. **Экстракция синтеза** — `SynthesisExtractorInterface.forward((text, material))` → `GeneralSynthesisOntology`
4. **Оценка** — `DspyGeneralSynthesisJudge.forward((text, json, material))` → оценка
5. **Сохранение** — `ResultGatherInterface.gather(paper_with_results)`

Все компоненты инстанцируются через `hydra.utils.instantiate()` из YAML-конфигов. Статьи обрабатываются параллельно через `ThreadPoolExecutor(max_workers=4)`.

### Ключевые паттерны

- **Extractors** наследуют `ExtractorInterface(dspy.Module, Generic[T, R])` с методом `forward()` (`src/llm_synthesis/transformers/base.py:16`)
- **Pipelines** наследуют `BasePipeline(ABC)` с методом `run()` (`src/llm_synthesis/services/pipelines/base_pipeline.py:4`)
- **Hydra-конфиги** используют `_target_` для инстанцирования классов. Корень конфигов: `examples/config/`. Компоненты: `data_loader`, `synthesis_extraction`, `material_extraction`, `judge`, `result_save`
- **LLM Registry** в `src/llm_synthesis/utils/llms.py:44` — `LLMConfig` dataclass, `SystemPrefixedLM` оборачивает dspy.LM с трекингом стоимости
- **DSPy utils** в `src/llm_synthesis/utils/dspy_utils.py` — `get_llm_from_name()`, `configure_dspy()`
- **Модели данных** — Pydantic: `Paper`, `SynthesisEntry`, `PaperWithSynthesisOntologies` в `src/llm_synthesis/models/paper.py`
- **Хранение результатов** через fsspec для абстракции файловой системы (локальная + GCS) в `src/llm_synthesis/result_gather/synthesis_results/fs_result_gather.py`

### Соглашения
- Длина строки: 80 символов (ruff)
- Импорты: standard → third-party → local (ruff авто-сортирует)
- Type hints: современный синтаксис (`list[str]`, `str | None`)
- Тестов нет — директория `tests/` пуста

### Ключевые файлы
- `examples/scripts/deployment/extract_synthesis_procedure_from_text.py` — основная точка входа для оборачивания
- `examples/config/config.yaml` — корневой Hydra-конфиг
- `src/llm_synthesis/transformers/base.py` — базовый класс ExtractorInterface
- `src/llm_synthesis/services/pipelines/base_pipeline.py` — BasePipeline ABC
- `src/llm_synthesis/services/pipelines/synthesis_performance_pipeline.py` — наиболее полный пайплайн (6 шагов)
- `src/llm_synthesis/utils/dspy_utils.py` — утилиты конфигурации LLM
- `src/llm_synthesis/utils/llms.py` — LLM registry и SystemPrefixedLM
- `src/llm_synthesis/models/paper.py` — модели Paper, SynthesisEntry

### Подводные камни
- Hydra меняет CWD (`hydra.job.chdir: true`), поэтому относительные пути требуют `get_original_cwd()`. Это касается `system_prompt` путей, `data_dir` И `result_dir` — все должны быть преобразованы в абсолютные перед передачей в Prefect flow
- Пути `system_prompt` в конфигах относительные — разрешаются вручную в entry-скрипте (строки 64-85)
- `result_dir` относительный (`results`) — проверка уже обработанных статей (`os.listdir(result_dir)`) зависит от chdir Hydra. Нужно преобразовать в абсолютный перед flow
- `dspy.settings.context()` используется per-call для изоляции настроек — это потокобезопасно. Однако `SystemPrefixedLM._cumulative_cost_usd` — разделяемый мутабельный float без блокировки — НЕ потокобезопасен при конкурентных Prefect-задачах
- `DspyGeneralSynthesisJudge` требует `signature: type[dspy.Signature]` при создании — не может быть инстанцирован без Hydra-конфига. В тестах мокать через `MagicMock(spec=DspyGeneralSynthesisJudge)`
- `MaterialExtractorInterface.forward()` принимает `input: str` как keyword-аргумент и возвращает строку через запятую, НЕ список. Вызывать как `extractor.forward(input=clean_text(text))` и парсить: `[m.strip() for m in result.replace('\n', ',').split(',') if m.strip()]` (см. строки 142-155 entry-скрипта)
- Нет тестовой инфраструктуры — нужен `pytest` в dev-зависимостях, `tests/`, `conftest.py`, конфигурация pytest-маркеров

### Точки интеграции Prefect
- `@task` оборачивает каждый шаг: загрузка, экстракция материалов, экстракция синтеза, оценка, сохранение
- `@flow` заменяет монолитную функцию `main()`
- `ThreadPoolTaskRunner` (`from prefect.task_runners import ThreadPoolTaskRunner`) заменяет `ThreadPoolExecutor` — требует Prefect >= 3.1
- `get_run_logger()` заменяет `logging.getLogger()`
- Параметры retry: `retries=3, retry_delay_seconds=5` на LLM-задачах
- Hydra инстанцирует компоненты, Prefect оркестрирует выполнение
- Компоненты инстанцируются внутри flow из сериализуемого Hydra-конфиг dict (НЕ передаются как pre-built объекты) — чтобы избежать проблем сериализации Prefect с non-picklable объектами типа `dspy.LM` с httpx connection pools

### Потокобезопасность трекинга стоимости
- `SystemPrefixedLM._cumulative_cost_usd` мутируется через `_extract_and_accumulate_cost()` без блокировки
- При конкурентных Prefect-задачах, разделяющих один экземпляр LM, это приводит к повреждению данных о стоимости
- Решение: добавить `threading.Lock` в `SystemPrefixedLM`, защищающий инкремент стоимости в `_extract_and_accumulate_cost()` (`src/llm_synthesis/utils/llms.py:139`)

## Трекинг прогресса

- [x] Задача 1: Исследование архитектуры и дизайн интеграции
- [x] Задача 2: Добавление Prefect и структура модуля
- [ ] Задача 3: Создание Prefect-задач для шагов пайплайна
- [ ] Задача 4: Создание основного flow экстракции
- [ ] Задача 5: Hydra-конфигурация оркестрации
- [ ] Задача 6: Entry point скрипт с Prefect
- [ ] Задача 7: Unit-тесты

**Всего задач:** 7 | **Выполнено:** 2 | **Осталось:** 5

## Задачи реализации

### Задача 1: Исследование архитектуры и дизайн интеграции [LEM-2]

**Цель:** Задокументировать полную архитектуру проекта и создать документ по дизайну интеграции с Prefect, зафиксировав все существующие паттерны и ограничения перед началом реализации.

**Зависимости:** Нет

**Файлы:**
- Создать: `docs/prefect-integration-design.md`

**Ключевые решения / Заметки:**
- Сопоставить каждый компонент пайплайна экстракции с Prefect task/flow
- Документировать типы данных между задачами (какие Pydantic-модели передаются)
- Определить, какие шаги нуждаются в retry (LLM-вызовы) vs детерминированные (загрузка, сохранение)
- Ссылаться на `src/llm_synthesis/services/pipelines/synthesis_performance_pipeline.py` как на наиболее полный паттерн пайплайна
- Документировать проблему потокобезопасности `SystemPrefixedLM._cumulative_cost_usd` и необходимый фикс с `threading.Lock`
- Документировать, что `DspyGeneralSynthesisJudge` требует `signature` при создании — не может быть инстанцирован без Hydra-конфига
- Документировать, что компоненты должны инстанцироваться внутри flow из сериализуемого конфига (не передаваться как pre-built объекты) для избежания pickle-ошибок на httpx/dspy-объектах

**Definition of Done:**
- [ ] `docs/prefect-integration-design.md` существует с: диаграммой архитектуры (ASCII), таблицей маппинга компонентов, типами данных между задачами, стратегией retry, анализом потокобезопасности, стратегией сериализации
- [ ] Все точки интеграции документированы со ссылками file:line
- [ ] Нет ошибок диагностики

**Верификация:**
- Файл существует и содержит все необходимые секции

---

### Задача 2: Добавление Prefect и структура модуля [LEM-3]

**Цель:** Добавить Prefect 3.x как зависимость проекта и создать скелет модуля оркестрации.

**Зависимости:** Нет

**Файлы:**
- Изменить: `pyproject.toml`
- Создать: `src/llm_synthesis/orchestration/__init__.py`
- Создать: `src/llm_synthesis/orchestration/tasks.py`
- Создать: `src/llm_synthesis/orchestration/flows.py`

**Ключевые решения / Заметки:**
- Добавить `prefect>=3.1` в `[project.dependencies]` pyproject.toml (3.1+ нужен для `ThreadPoolTaskRunner`)
- Добавить `pytest` и `pytest-asyncio` в `[dependency-groups] dev`
- Добавить `[tool.pytest.ini_options]` с `markers = ["unit: Unit tests", "integration: Integration tests"]`
- Создать пустые файлы модуля с docstrings и экспортом публичного API
- `tasks.py` будет содержать все `@task`-декорированные функции
- `flows.py` будет содержать все `@flow`-декорированные функции

**Definition of Done:**
- [ ] `uv pip install -e .` успешно с установленным prefect
- [ ] `python -c "from llm_synthesis.orchestration import tasks, flows"` успешно
- [ ] `from prefect.task_runners import ThreadPoolTaskRunner` успешно
- [ ] `from prefect.testing.utilities import prefect_test_harness` успешно
- [ ] `uv run pytest --version` успешно
- [ ] Нет ошибок диагностики

**Верификация:**
- `uv run python -c "import prefect; print(prefect.__version__)"`
- `uv run python -c "from prefect.task_runners import ThreadPoolTaskRunner; from prefect.testing.utilities import prefect_test_harness; print('ok')"`
- `uv run python -c "from llm_synthesis.orchestration import tasks, flows"`

---

### Задача 3: Создание Prefect-задач для шагов пайплайна [LEM-4]

**Цель:** Обернуть каждый шаг пайплайна как Prefect `@task` с соответствующей retry-логикой и логированием.

**Зависимости:** Задача 2

**Файлы:**
- Изменить: `src/llm_synthesis/orchestration/tasks.py`

**Ключевые решения / Заметки:**
- Следовать шагам пайплайна из `extract_synthesis_procedure_from_text.py:44-344`
- Задачи для создания:
  - `load_papers(data_loader) -> list[Paper]` — без retry (детерминированная)
  - `extract_materials(extractor, paper_text) -> list[str]` — retries=3 (LLM-вызов). **Важно:** вызывать как `extractor.forward(input=clean_text(paper_text))`, парсить CSV-результат: `[m.strip() for m in result.replace('\n', ',').split(',') if m.strip()]`, возвращать пустой список если falsy. Ссылка: `extract_synthesis_procedure_from_text.py:142-155`
  - `extract_synthesis(extractor, paper_text, material) -> GeneralSynthesisOntology` — retries=3 (LLM-вызов). Вызывать как `extractor.forward(input=(clean_text(paper_text), material))`
  - `evaluate_synthesis(judge, paper_text, synthesis_json, material) -> evaluation` — retries=2 (LLM-вызов). Вызывать как `judge.forward((clean_text(paper_text), synthesis_json, material))`
  - `save_paper_results(result_gather, paper_with_results) -> None` — retries=1 (I/O)
- Каждая задача использует `get_run_logger()` для структурированного логирования
- Задачи получают инстанцированные компоненты как аргументы — компоненты создаются внутри flow из Hydra-конфига (см. Задачу 4)
- Возвращать Pydantic-модели напрямую (Prefect их сериализует)
- Использовать `clean_text()` из `llm_synthesis.utils` перед всеми вызовами extractors

**Definition of Done:**
- [ ] 5 Prefect-задач определены с корректными сигнатурами и type hints
- [ ] LLM-вызывающие задачи имеют `retries` и `retry_delay_seconds`
- [ ] Каждая задача использует `get_run_logger()` для логирования
- [ ] Нет ошибок диагностики

**Верификация:**
- `ruff check src/llm_synthesis/orchestration/tasks.py`
- `uv run python -c "from llm_synthesis.orchestration.tasks import load_papers, extract_materials, extract_synthesis, evaluate_synthesis, save_paper_results"`

---

### Задача 4: Создание основного flow экстракции [LEM-5]

**Цель:** Создать Prefect `@flow`, оркестрирующий полный пайплайн экстракции синтеза, используя задачи из Задачи 3.

**Зависимости:** Задача 3

**Файлы:**
- Изменить: `src/llm_synthesis/orchestration/flows.py`
- Изменить: `src/llm_synthesis/utils/llms.py` (добавление threading.Lock)

**Ключевые решения / Заметки:**
- `synthesis_extraction_flow` заменяет логику `extract_synthesis_procedure_from_text.py:44-344`
- Использует `ThreadPoolTaskRunner` (`from prefect.task_runners import ThreadPoolTaskRunner`) для конкурентной обработки статей
- **Стратегия сериализации:** Flow получает plain `dict`-конфиг (преобразованный из Hydra DictConfig через `OmegaConf.to_container(cfg, resolve=True)`) и инстанцирует компоненты внутри flow через `hydra.utils.instantiate()`. Это избегает pickle non-serializable объектов (dspy.LM содержит httpx connection pools)
- Внутренняя функция `process_single_paper()` оркестрирует шаги для каждой статьи
- Каждая задача возвращает дельту стоимости — flow агрегирует общую стоимость (избегает shared mutable state)
- Добавить `threading.Lock` в `SystemPrefixedLM._extract_and_accumulate_cost()` для потокобезопасного трекинга стоимости

**Definition of Done:**
- [ ] `synthesis_extraction_flow` определён с `@flow` декоратором
- [ ] Использует `ThreadPoolTaskRunner` для конкурентности
- [ ] Инстанцирует компоненты из config dict внутри flow (не pre-instantiated аргументы)
- [ ] Обрабатывает статьи: экстракция материалов → экстракция синтеза → оценка → сохранение
- [ ] Все пути (result_dir, system_prompt, data_dir) преобразованы в абсолютные перед использованием
- [ ] Агрегирует и логирует общую стоимость через возвращаемые значения задач
- [ ] `threading.Lock` добавлен в `SystemPrefixedLM._extract_and_accumulate_cost()`
- [ ] Нет ошибок диагностики

**Верификация:**
- `ruff check src/llm_synthesis/orchestration/flows.py`
- `uv run python -c "from llm_synthesis.orchestration.flows import synthesis_extraction_flow"`

---

### Задача 5: Hydra-конфигурация оркестрации [LEM-6]

**Цель:** Добавить Hydra config-группу для настроек оркестрации (retries, concurrency, task runner).

**Зависимости:** Задача 2

**Файлы:**
- Создать: `examples/config/orchestration/default.yaml`
- Изменить: `examples/config/config.yaml` — добавить `orchestration: default` в defaults

**Ключевые решения / Заметки:**
- Структура конфига:
  ```yaml
  max_workers: 4
  retries:
    material_extraction: 3
    synthesis_extraction: 3
    judge_evaluation: 2
    result_save: 1
  retry_delay_seconds: 5
  log_level: INFO
  ```
- Без `_target_` для оркестрации (не класс, а настройки)
- Сохраняет обратную совместимость: существующие Hydra-скрипты игнорируют неизвестные config-группы

**Definition of Done:**
- [ ] `examples/config/orchestration/default.yaml` существует с документированными настройками
- [ ] `examples/config/config.yaml` включает `orchestration: default` в defaults
- [ ] Hydra резолвит конфиг без ошибок
- [ ] Нет ошибок диагностики

**Верификация:**
- `cd examples && uv run python -c "import hydra; hydra.initialize(config_path='config', version_base=None); cfg = hydra.compose('config'); print(cfg.orchestration); hydra.core.global_hydra.GlobalHydra.instance().clear()"`

---

### Задача 6: Entry point скрипт с Prefect [LEM-7]

**Цель:** Создать новый entry point скрипт, использующий Hydra для конфигурации и Prefect для выполнения, заменяя подход с ThreadPoolExecutor.

**Зависимости:** Задача 4, Задача 5

**Файлы:**
- Создать: `examples/scripts/deployment/extract_synthesis_prefect.py`

**Ключевые решения / Заметки:**
- Следует паттерну `extract_synthesis_procedure_from_text.py`, но делегирует Prefect flow
- `@hydra.main()` управляет загрузкой конфига и CWD
- Внутри `main()`: преобразовать Hydra DictConfig в plain dict через `OmegaConf.to_container(cfg, resolve=True)`, затем вызвать `synthesis_extraction_flow()`
- Передать настройки оркестрации (retries, workers) из Hydra-конфига в flow
- Преобразовать ВСЕ относительные пути в абсолютные перед передачей в flow: пути `system_prompt` (строки 64-85), `data_dir` (строки 47-57) И `result_dir` — через `os.path.join(get_original_cwd(), path)`
- Оригинальный скрипт не трогаем — это аддитивное изменение

**Definition of Done:**
- [ ] `extract_synthesis_prefect.py` существует и запускается
- [ ] Использует `@hydra.main()` для конфигурации
- [ ] Вызывает `synthesis_extraction_flow()` с сериализуемым config dict
- [ ] `result_dir` преобразован в абсолютный путь перед вызовом flow
- [ ] Передаёт retry/concurrency настройки из Hydra orchestration конфига
- [ ] Скрипт можно запустить: `uv run python examples/scripts/deployment/extract_synthesis_prefect.py`
- [ ] Нет ошибок диагностики

**Верификация:**
- `ruff check examples/scripts/deployment/extract_synthesis_prefect.py`
- `uv run python examples/scripts/deployment/extract_synthesis_prefect.py --help` (Hydra help output)

---

### Задача 7: Unit-тесты [LEM-8]

**Цель:** Создать unit-тесты для модуля оркестрации, замокав все LLM и I/O зависимости.

**Зависимости:** Задача 4

**Файлы:**
- Создать: `tests/__init__.py`
- Создать: `tests/conftest.py`
- Создать: `tests/test_orchestration_tasks.py`
- Создать: `tests/test_orchestration_flows.py`

**Ключевые решения / Заметки:**
- Использовать `prefect.testing.utilities.prefect_test_harness` для тестирования flow/task (импорт верифицирован в Задаче 2)
- Мокать все DSPy extractors и LLM-вызовы — тесты не должны делать реальных API-вызовов
- Мокать judge через `MagicMock(spec=DspyGeneralSynthesisJudge)` — он требует `signature` при создании
- Тестировать каждую задачу отдельно с замоканными входами
- Тестировать оркестрацию flow с замоканными задачами
- Тестировать конкурентный трекинг стоимости: верифицировать что `SystemPrefixedLM` lock предотвращает повреждение данных при конкурентном доступе
- Использовать `pytest` с `caplog` для верификации Prefect-логирования
- Паттерн из Prefect docs: оборачивать тесты в `with prefect_test_harness():`

**Definition of Done:**
- [ ] Все 5 задач имеют индивидуальные unit-тесты
- [ ] Flow имеет интеграционный тест с замоканными задачами
- [ ] Все тесты проходят: `uv run pytest tests/ -q`
- [ ] Нет реальных LLM/API вызовов в тестах (полностью замокано)
- [ ] Нет ошибок диагностики

**Верификация:**
- `uv run pytest tests/ -q`

## Стратегия тестирования

- **Unit-тесты:** Каждая Prefect-задача тестируется с замоканными extractors/judges/loaders. Верификация корректных типов возврата, поведения retry, вывода логов.
- **Интеграционные тесты:** Flow тестируется с замоканными задачами, верифицируя логику оркестрации (фильтрация статей, агрегация стоимости, конкурентное выполнение).
- **Ручная верификация:** `extract_synthesis_prefect.py --help` выводит Hydra-конфиг. Dry run возможен с небольшим локальным датасетом.

## Риски и митигации

| Риск | Вероятность | Влияние | Митигация |
|------|-----------|---------|-----------|
| Prefect не может сериализовать non-picklable объекты (dspy.LM, httpx) | Средняя | Высокое | Передавать сериализуемый config dict в flow, инстанцировать компоненты внутри flow. Тестировать picklability в Задаче 7 |
| Смена CWD Hydra конфликтует с рабочей директорией Prefect | Средняя | Среднее | Преобразовать ВСЕ пути (system_prompt, data_dir, result_dir) в абсолютные в entry point перед вызовом flow |
| ThreadPoolTaskRunner ведёт себя иначе, чем ThreadPoolExecutor | Низкая | Среднее | Тестировать конкурентное выполнение в Задаче 7; fallback на последовательное если проблемы |
| DSPy settings не потокобезопасны | Низкая | Низкое | Уже обработано: код использует `dspy.settings.context()` per-call, что потокобезопасно |
| Трекинг стоимости SystemPrefixedLM не потокобезопасен | Высокая | Среднее | Добавить `threading.Lock` в `_extract_and_accumulate_cost()` в Задаче 4. Тестировать конкурентный доступ в Задаче 7 |

## Верификация цели

### Истины
1. `prefect` установлен и импортируется в проекте
2. `from llm_synthesis.orchestration import tasks, flows` работает
3. Prefect flow может обрабатывать статьи через полный пайплайн экстракции
4. Каждая LLM-вызывающая задача повторяется при ошибке согласно конфигу
5. Настройки оркестрации конфигурируемы через Hydra YAML
6. Все unit-тесты проходят с замоканными зависимостями
7. Существующая кодовая база не изменена — интеграция чисто аддитивная

### Артефакты
- `src/llm_synthesis/orchestration/tasks.py` — 5 Prefect-задач
- `src/llm_synthesis/orchestration/flows.py` — основной flow экстракции
- `examples/config/orchestration/default.yaml` — Hydra-конфиг
- `examples/scripts/deployment/extract_synthesis_prefect.py` — точка входа
- `tests/test_orchestration_tasks.py` — тесты задач
- `tests/test_orchestration_flows.py` — тесты flow
- `docs/prefect-integration-design.md` — документ архитектуры

### Ключевые связи
- `orchestration/tasks.py` импортирует из `transformers/base.py`, `models/paper.py`, `result_gather/base.py`
- `orchestration/flows.py` импортирует из `orchestration/tasks.py`
- `extract_synthesis_prefect.py` использует Hydra для инстанцирования + вызывает `synthesis_extraction_flow()`
- `examples/config/config.yaml` включает `orchestration: default`

## Отложенные идеи
- Обернуть `SynthesisPerformancePipeline` (6-шаговый пайплайн с экстракцией фигур и линковкой performance) как отдельный Prefect flow
- Добавить Prefect deployment конфиги для запусков по расписанию
- Добавить Prefect server UI для мониторинга
- Реализовать кэширование через `@task(cache_key_fn=...)` для идемпотентных перезапусков
- Добавить `@flow`-обёртку для `batch_run_tc_new_snippet.py` case study
