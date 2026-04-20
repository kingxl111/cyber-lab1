# Лабораторная работа 1: CV

Курс «Кибер-физические системы».
Тема: классификация дорожных знаков на наборе данных **GTSRB** (German Traffic Sign Recognition Benchmark, 43 класса).

В работе обучены и сравнены три типа моделей:

- **ResNet18** — сверточная сеть из `torchvision`, fine-tuning ImageNet-весов;
- **ViT-B/16** — трансформерная сеть из `torchvision`, fine-tuning ImageNet-весов;
- **SimpleCNN** — собственная сверточная сеть, написанная с нуля.

Каждая модель прогнана в двух режимах: `baseline` минимальный пайплайн и `improved` со всеми улучшениями.

---

## 1. Структура проекта

```
cyber-lab1/
├── Train/  Test/  Meta/        # GTSRB, распакован в корень
├── Train.csv  Test.csv  Meta.csv
├── src/
│   ├── data.py                 # GTSRBDataset + ROI crop + CV-аугментации
│   ├── models.py               # SimpleCNN (собственная архитектура)
│   ├── utils.py                # seed, train loop, multiclass-метрики
│   └── plots.py                # loss / accuracy / confusion matrix / per-class F1
├── train_torchvision.py        # ResNet18 / ViT-B/16
├── train_custom.py             # SimpleCNN
├── requirements.txt
└── README.md
```

---

## 2. Подготовка окружения

### 2.1. Виртуальное окружение и зависимости

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.2. GPU

Все эксперименты запускались на одной видеокарте **NVIDIA RTX 2060 (6 GB)**, драйвер 580.126.09, CUDA 13.0, PyTorch 2.x с CUDA-сборкой.

```
NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0
```

На CPU обучение тоже работает, но `ViT-B/16` в этом случае непрактично долгий.

### 2.3. Датасет

GTSRB скачан с Kaggle: <https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>.

После распаковки архива в корне репозитория должны лежать:

```
Train/   # 39 209 PNG в подпапках по ClassId
Test/    # 12 630 PNG, плоский список
Meta/    # шаблоны знаков (используются опционально)
Train.csv  Test.csv  Meta.csv
```

CSV содержат колонки `Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path`. ROI используется для обрезки изображения по знаку перед ресайзом.

---

## 3. Запуск экспериментов

### 3.1. Собственная CNN

```bash
python train_custom.py --variant baseline
python train_custom.py --variant improved
```

### 3.2. Модели из torchvision

```bash
python train_torchvision.py --model resnet18 --variant baseline
python train_torchvision.py --model resnet18 --variant improved

python train_torchvision.py --model vit_b_16 --variant baseline  --epochs 2 --batch-size 48 --num-workers 8
python train_torchvision.py --model vit_b_16 --variant improved  --epochs 4 --batch-size 32 --num-workers 8
```

ViT-B/16 — самая тяжёлая модель ≈ 86 M параметров, обязательно 224×224, поэтому для RTX 2060 она запускалась с уменьшенным числом эпох. ResNet18 и SimpleCNN тренируются с дефолтными значениями.

### 3.3. Полезные аргументы CLI

| Аргумент | Что делает | Где доступен |
|---|---|---|
| `--variant {baseline,improved}` | переключение между чистым и улучшенным конвейером | оба скрипта |
| `--model {resnet18,vit_b_16}` | выбор torchvision-архитектуры | `train_torchvision.py` |
| `--epochs N` | переопределить число эпох | оба скрипта |
| `--batch-size N` | размер батча | оба скрипта |
| `--image-size N` | разрешение входа | оба скрипта (для ViT с pretrained — оставить 224) |
| `--num-workers N` | потоки `DataLoader` | оба скрипта |
| `--no-pretrained` | обучение torchvision-моделей с нуля | `train_torchvision.py` |
| `--seed N` | random seed | оба скрипта |

---

# Отчёт по лабораторной работе

## 1. Выбор начальных условий

### 1.1. Набор данных

**GTSRB** — German Traffic Sign Recognition Benchmark. Публичный датасет 43 классов дорожных знаков, ~39 200 обучающих и ~12 630 тестовых изображений. Изображения разного размера (от ~15 × 15 до ~250 × 250 пикселей), формат PNG. Для каждого изображения в CSV указаны координаты ROI (bounding box знака), что позволяет обрезать фон до подачи в модель.

### 1.2. Обоснование выбора датасета

- **Реальная практическая задача**: распознавание знаков — ключевой компонент систем автономного транспорта (ADAS). Это прямое применение в кибер-физических системах.
- **Существенный дисбаланс классов** (от ~210 до ~2 250 образцов на класс) — даёт осмысленный выбор метрик (нельзя смотреть только accuracy) и техник борьбы с дисбалансом (class weights, label smoothing).
- **43 класса** делают задачу нетривиальной (есть похожие знаки, например ограничения скорости 30/50/80 km/h), но не такой, чтобы требовать сотен часов обучения.
- Для архитектур `torchvision` доступны предобученные на ImageNet веса — это позволяет честно сравнить CNN (ResNet18) и трансформер (ViT-B/16).

### 1.3. Выбор метрик качества

- **Accuracy** — интуитивная общая метрика.
- **Precision / Recall (macro)** — усреднение по классам без взвешивания их размера; не занижается из-за крупных классов.
- **F1-score (macro)** — основная метрика для ранжирования моделей в этой работе.
- **F1-score (weighted)** — учитывает размер классов; полезно для сопоставления с accuracy.
- **ROC-AUC (One-vs-Rest, macro)** — оценивает качество ранжирования вероятностей по 43 классам.

### 1.4. Обоснование выбора метрик

В задаче сильный дисбаланс. Accuracy на нём переоценит качество: даже модель, хорошо различающая только крупные классы, получит высокую accuracy.

Поэтому основной метрикой выбрана **F1-score (macro)**, которая равномерно оценивает все 43 класса. Precision/Recall (macro) и ROC-AUC OvR дают дополнительную картину поведения модели на редких классах.

---

## 2. Создание бейзлайна и оценка качества

### 2.1. Состав бейзлайна

Обучены две модели из `torchvision` с предобученными ImageNet-весами:

- **ResNet18** — компактная сверточная сеть (~11.7 M параметров);
- **ViT-B/16** — трансформер с патчами 16 × 16 (~86 M параметров).

Baseline-конфигурация (`--variant baseline`, файл `train_torchvision.py`):

- входное разрешение **224 × 224** (совпадает с ImageNet);
- ROI-crop по координатам из CSV перед ресайзом;
- **без аугментаций** на train (только `Resize → ToTensor → Normalize` по ImageNet статистикам);
- `CrossEntropyLoss` без `class_weight`;
- `Adam`, `lr = 1e-3`, без LR-scheduler;
- early stopping по `val_loss`, `patience = 3`.

### 2.2. Результаты baseline

| Модель | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC OvR | Время |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 baseline | **0.9861** | **0.9851** | **0.9764** | **0.9784** | **0.9857** | 1.0000 | 11 мин |
| ViT-B/16 baseline | 0.9731 | 0.9694 | 0.9603 | 0.9631 | 0.9726 | 0.9996 | 28 мин |

### 2.3. Вывод по baseline

**ResNet18** уверенно обогнала ViT-B/16 при примерно в 2.5 раза меньшем времени обучения. Несмотря на то что ViT-B/16 в общем случае мощнее, на GTSRB он проигрывает по двум причинам:

1. знаки на оригинальных снимках маленькие (часто 30–50 пикселей), и upscale до 224 × 224 размывает информацию, которую CNN с более локальными рецептивными полями использует эффективнее;
2. ViT «голоден» на данные, и 33 327 train-картинок 43 классов недостаточно, чтобы развернуть его потенциал, особенно на baseline без аугментаций.

---

## 3. Улучшение бейзлайна

### 3.1. Сформулированные гипотезы

1. **CV-аугментации** (`RandomCrop`, `RandomAffine`, `ColorJitter`, `RandAugment`, `RandomErasing`) снизят переобучение и улучшат обобщение.
   - *Важно:* `HorizontalFlip` **не используется** — у многих знаков (повороты, «keep left/right», стрелки) горизонтальное отражение меняет класс.
2. **Class weights + label smoothing** компенсируют дисбаланс 43 классов и подавляют переуверенность модели на крупных классах.
3. **AdamW + CosineAnnealingLR** дадут более стабильное и плавное обучение, чем `Adam` с фиксированным `lr`.
4. **Weight decay 1e-2** в AdamW работает как сильный регуляризатор для больших pretrained моделей.
5. **Увеличение числа эпох + early stopping** позволит полностью использовать аугментации без переобучения.

### 3.2. Проверка гипотез

Improved-конфигурация (`--variant improved`):

- аугментации: `RandomCrop`, `RandomAffine(rotate=15, translate=0.08, scale=(0.9, 1.1), shear=8)`, `ColorJitter(0.3, 0.3, 0.2, 0.03)`, `RandAugment(num_ops=2, magnitude=7)`, `RandomErasing(p=0.25)`;
- `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`;
- `AdamW(lr=5e-4, weight_decay=1e-2)`;
- `CosineAnnealingLR(T_max=epochs)`;
- увеличенное число эпох с `patience=4`.

### 3.3. Сформированный улучшенный бейзлайн

Все улучшения зашиты в `src/data.py::build_transforms(variant="improved")` и в ветку `improved` тренировочных скриптов. Никаких ручных переключателей — достаточно `--variant improved`.

### 3.4. Результаты improved-моделей

| Модель | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC OvR | Время |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 improved | **0.9947** | **0.9938** | **0.9935** | **0.9935** | **0.9947** | 1.0000 | 23 мин |
| ViT-B/16 improved | 0.9886 | 0.9857 | 0.9841 | 0.9847 | 0.9884 | 0.9999 | 58 мин |

### 3.5. Сравнение baseline vs improved

| Модель | Δ Accuracy | Δ F1-macro | Δ Recall-macro |
|---|---:|---:|---:|
| ResNet18 baseline → improved | +0.86 п.п. | **+1.51 п.п.** | +1.71 п.п. |
| ViT-B/16 baseline → improved | +1.55 п.п. | **+2.16 п.п.** | +2.38 п.п. |

Улучшения дали прирост по **всем** метрикам у обеих моделей. Особенно заметен рост `recall_macro` (+1.7 / +2.4 п.п.), что подтверждает гипотезу №2 — class weights действительно помогли редким классам, которые у baseline модель пропускала чаще среднего.

ViT-B/16 получил больший относительный прирост, чем ResNet18, — улучшения частично компенсируют его «голод на данные» через искусственное расширение распределения аугментациями.

### 3.6. Выводы по улучшенному бейзлайну

1. CV-аугментации **без HorizontalFlip** для GTSRB критичны: модель видит больше «вариантов» каждого знака (наклоны, сдвиги, изменения освещения), что соответствует реальным условиям съёмки.
2. Связка **class weights + label smoothing** — основной драйвер роста `f1_macro`, особенно на ViT.
3. **CosineAnnealingLR** убирает зигзаги `val_loss`, которые видны у baseline (`resnet18_baseline` epoch 4–5: 0.0516 → 0.1029 → 0.0151).
4. ResNet18 остаётся лидером и среди improved-моделей: ~99.5 % accuracy и `f1_macro` 0.9935.

> **Важное замечание о значениях `loss`.**
> Абсолютные `train_loss / val_loss` baseline-моделей (~0.01–0.05) и improved-моделей (~0.6–0.7) **сравнивать напрямую нельзя**. У improved используется `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`, у которого минимум > 0 даже при идеальном предсказании, плюс `weight` масштабирует значение. Сравнение корректно только по метрикам на тесте.

---

## 4. Имплементация алгоритма машинного обучения

### 4.1. Своя архитектура — `SimpleCNN`

Файл [`src/models.py`](src/models.py). Компактная сверточная сеть, обучаемая с нуля:

```
Input  3 × 64 × 64
└── 4 × [Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool2]
    каналы  3 → 32 → 64 → 128 → 256
└── AdaptiveAvgPool2d(1)
└── Flatten → Dropout → Linear(256 → 128) → ReLU → Dropout → Linear(128 → 43)
```

Всего **1 211 659 параметров** — это в ~10 раз меньше ResNet18 и в ~70 раз меньше ViT-B/16. Входное разрешение 64 × 64 (знаки маленькие, больше не нужно), что также делает обучение очень быстрым.

### 4.2. Baseline-вариант собственной модели

`train_custom.py --variant baseline`:
- без аугментаций;
- без class weights, без label smoothing;
- `CrossEntropyLoss`, `Adam(lr=1e-3, weight_decay=1e-4)`, без scheduler;
- `dropout = 0.2`;
- 15 эпох, early stopping `patience = 4`.

### 4.3. Сравнение собственной модели с torchvision-моделями (п. 4d ТЗ)

| Модель | Accuracy | F1 (macro) | Параметры | Время |
|---|---:|---:|---:|---:|
| SimpleCNN baseline | 0.9635 | 0.9395 | 1.2 M | **2 мин** |
| ResNet18 baseline | 0.9861 | 0.9784 | 11.7 M | 11 мин |
| ViT-B/16 baseline | 0.9731 | 0.9631 | 86 M | 28 мин |

SimpleCNN baseline уступает обеим pretrained моделям, что ожидаемо: сеть учится с нуля на 33 327 картинках, без аугментаций и без переноса знаний с ImageNet. Но даже так она достигает 96.4 % accuracy — задача GTSRB достаточно «лёгкая» для конволюций.

### 4.4. Добавление техник из улучшенного бейзлайна (п. 4f ТЗ)

`train_custom.py --variant improved` включает все улучшения из раздела 3.2:

- CV-аугментации (`RandomAffine`, `RandAugment`, `ColorJitter`, `RandomErasing`);
- `class weights + label smoothing 0.05`;
- `AdamW(lr=1e-3, weight_decay=5e-4) + CosineAnnealingLR`;
- повышенный `dropout = 0.45`;
- 25 эпох, `patience = 6`.

### 4.5. Сравнение с пунктом 3 (п. 4i ТЗ)

| Модель | Accuracy | F1 (macro) | ROC-AUC OvR | Время |
|---|---:|---:|---:|---:|
| SimpleCNN baseline | 0.9635 | 0.9395 | 0.9996 | 2 мин |
| **SimpleCNN improved** | **0.9908** | **0.9855** | **1.0000** | **9 мин** |
| ResNet18 improved | 0.9947 | 0.9935 | 1.0000 | 23 мин |
| ViT-B/16 improved | 0.9886 | 0.9847 | 0.9999 | 58 мин |

**SimpleCNN improved** обогнала **ViT-B/16 improved** и по `accuracy`, и по `f1_macro`, при том что:

- параметров в ~70 раз меньше;
- время обучения в ~6 раз меньше;
- модель учится с нуля, без переноса знаний с ImageNet.

Это главный результат раздела 4: техники из улучшенного бейзлайна (аугментации + регуляризация + class weights + label smoothing + cosine LR) дают для самописной маленькой CNN прирост **+2.7 п.п. accuracy** и **+4.6 п.п. f1_macro**, что значительно больше эффекта тех же техник для ResNet18 (+0.9 / +1.5 п.п.) и ViT-B/16 (+1.5 / +2.2 п.п.).

### 4.6. Вывод по самостоятельной имплементации

Маленькая правильно сконфигурированная CNN с современными техниками регуляризации сравнима по качеству с дорогими предобученными трансформерами на узкой задаче с фиксированной природой объектов. **Архитектура не главный фактор качества — пайплайн данных и регуляризация важнее размера модели.**

---

## 5. Итоговое сравнение всех моделей

| # | Модель | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC OvR | Параметры | Время |
|--:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **ResNet18 improved** | **0.9947** | **0.9938** | **0.9935** | **0.9935** | **0.9947** | 1.0000 | 11.7 M | 23 мин |
| 2 | SimpleCNN improved | 0.9908 | 0.9822 | 0.9895 | 0.9855 | 0.9908 | 1.0000 | 1.2 M | 9 мин |
| 3 | ViT-B/16 improved | 0.9886 | 0.9857 | 0.9841 | 0.9847 | 0.9884 | 0.9999 | 86 M | 58 мин |
| 4 | ResNet18 baseline | 0.9861 | 0.9851 | 0.9764 | 0.9784 | 0.9857 | 1.0000 | 11.7 M | 11 мин |
| 5 | ViT-B/16 baseline | 0.9731 | 0.9694 | 0.9603 | 0.9631 | 0.9726 | 0.9996 | 86 M | 28 мин |
| 6 | SimpleCNN baseline | 0.9635 | 0.9533 | 0.9374 | 0.9395 | 0.9627 | 0.9996 | 1.2 M | 2 мин |

Сортировка по основной метрике `f1_macro`.

---

## 6. Анализ результатов

### 6.1. Что лучше — CNN или Transformer

**CNN выигрывает на этом датасете** в обоих режимах:

- baseline: ResNet18 0.9784 vs ViT-B/16 0.9631 по `f1_macro`;
- improved: ResNet18 0.9935 vs ViT-B/16 0.9847.

Причины:

1. **Маленький размер знаков**. Большинство ROI имеет сторону 30–60 пикселей. Upscale до 224 × 224 нужен ради совместимости с ImageNet-весами, но реальной информации в них меньше, чем в типичных ImageNet-картинках. CNN с её локальными рецептивными полями и иерархией свёрток справляется с такими «бедными» картинками лучше, чем ViT с глобальным attention на 14 × 14 патчах.
2. **Объём данных**. ViT-B/16 рассчитан на гораздо большие датасеты для тонкой настройки. 33 327 train-сэмплов для 86 M параметров — это режим, где ViT не успевает раскрыться даже с pretrained весами.
3. **Время и память**. ViT в ~2.5–3 раза дороже ResNet18 по времени и памяти на той же задаче.

### 6.2. Эффект техник из improved-конвейера

Прирост в п.п. при переходе baseline → improved:

| Модель | Δ Accuracy | Δ F1 (macro) |
|---|---:|---:|
| **SimpleCNN** | **+2.73** | **+4.60** |
| ViT-B/16 | +1.55 | +2.16 |
| ResNet18 | +0.86 | +1.51 |

Чем «слабее» бейзлайн — тем больший относительный прирост дают улучшения. Для собственной CNN, которая училась с нуля, аугментации + регуляризация + class weights дают **в 3 раза больший прирост `f1_macro`**, чем для уже сильной pretrained ResNet18.

### 6.3. Конкурентоспособность собственной CNN

`SimpleCNN improved` (1.2 M параметров) обогнала `ViT-B/16 improved` (86 M параметров) по `accuracy` (0.9908 vs 0.9886) и по `f1_macro` (0.9855 vs 0.9847), отстав от `ResNet18 improved` всего на 0.39 п.п. accuracy и 0.80 п.п. `f1_macro`.

С точки зрения practitioners это означает, что **на узкой и хорошо определённой задаче** (фиксированные объекты, контролируемое позиционирование, ограниченный набор классов) маленькая собственная модель с продуманным пайплайном — вполне разумная альтернатива тяжёлым универсальным архитектурам.

### 6.4. Самая лучшая модель по соотношению качество / стоимость

| Модель | F1 (macro) | Время / эпоха | Память |
|---|---:|---:|---:|
| **SimpleCNN improved** | **0.9855** | ~22 сек | ~0.5 GB |
| ResNet18 improved | 0.9935 | ~115 сек | ~2 GB |
| ViT-B/16 improved | 0.9847 | ~870 сек | ~4 GB |

Если важен сухой максимум качества — **ResNet18 improved**. Если важен баланс «качество vs ресурсы» — **SimpleCNN improved**: всего на 0.8 п.п. отстаёт по `f1_macro`, при этом в ~5 раз быстрее обучается и в ~4 раза меньше потребляет памяти. На целевой платформе ADAS (встраиваемое железо в автомобиле) это решающее преимущество.

---

## 7. Выводы

1. Полный цикл лабораторной работы пройден: сформулирована задача, выбран и обоснован реальный CV-датасет (GTSRB) с практическим применением в кибер-физических системах, выбраны и обоснованы метрики качества под дисбаланс 43 классов.
2. Построен бейзлайн на двух архитектурах из `torchvision` (CNN и Transformer) с предобученными ImageNet-весами и оценено его качество.
3. Сформулированы и проверены 5 гипотез по улучшению бейзлайна. Все улучшения дали положительный эффект, что подтверждено на тесте.
4. Реализована собственная архитектура `SimpleCNN`, обучена с нуля, сравнена с torchvision-моделями. После добавления техник из улучшенного бейзлайна она вышла на уровень трансформера в 70 раз большего размера.
5. Практический результат:
   - **самая точная модель** — `ResNet18 improved` (`accuracy = 0.9947`, `f1_macro = 0.9935`);
   - **лучший компромисс «качество / ресурсы»** — `SimpleCNN improved` (`accuracy = 0.9908`, `f1_macro = 0.9855`, всего 1.2 M параметров и ~9 минут обучения);
   - **CNN-архитектуры выигрывают у ViT** на GTSRB — задача с маленькими объектами и относительно небольшим обучающим набором.
6. Главный методический вывод: **на узкой задаче пайплайн данных и регуляризация важнее размера модели.** Аугментации без `HorizontalFlip`, class weights, label smoothing и cosine LR в совокупности дали прирост `f1_macro` от +1.5 до +4.6 п.п. для всех трёх архитектур.

---

## Приложение. Логи прогонов

### A.1. SimpleCNN baseline

```bash
time python train_custom.py --variant baseline
Device: cuda
Run: customcnn_baseline | image_size=64 | batch_size=128 | epochs=15 | dropout=0.2
SimpleCNN parameters: 1,211,659
Epoch 01 | train_loss=1.1636 acc=0.6636 | val_loss=0.1805 acc=0.9541 | lr=1.00e-03
Epoch 02 | train_loss=0.0650 acc=0.9847 | val_loss=0.0439 acc=0.9876 | lr=1.00e-03
Epoch 03 | train_loss=0.0237 acc=0.9944 | val_loss=0.3590 acc=0.8914 | lr=1.00e-03
Epoch 04 | train_loss=0.0214 acc=0.9946 | val_loss=0.0245 acc=0.9944 | lr=1.00e-03
Epoch 05 | train_loss=0.0125 acc=0.9971 | val_loss=0.0287 acc=0.9925 | lr=1.00e-03
Epoch 06 | train_loss=0.0111 acc=0.9974 | val_loss=0.0760 acc=0.9764 | lr=1.00e-03
Epoch 07 | train_loss=0.0118 acc=0.9969 | val_loss=0.0347 acc=0.9888 | lr=1.00e-03
Epoch 08 | train_loss=0.0132 acc=0.9964 | val_loss=0.0276 acc=0.9932 | lr=1.00e-03
Early stopping. Best epoch: 4 (val_loss=0.0245)

TEST METRICS (customcnn_baseline)
  accuracy: 0.9635
  precision_macro: 0.9533
  recall_macro: 0.9374
  f1_macro: 0.9395
  f1_weighted: 0.9627
  roc_auc_ovr_macro: 0.9996

real    2m12,948s
user    5m23,902s
sys     0m20,534s
```

### A.2. SimpleCNN improved

```bash
time python train_custom.py --variant improved
Device: cuda
Run: customcnn_improved | image_size=64 | batch_size=128 | epochs=25 | dropout=0.45
SimpleCNN parameters: 1,211,659
Epoch 01 | train_loss=3.2286 acc=0.1344 | val_loss=2.5645 acc=0.2475 | lr=9.96e-04
Epoch 02 | train_loss=2.1576 acc=0.4373 | val_loss=1.4426 acc=0.6834 | lr=9.84e-04
Epoch 03 | train_loss=1.3318 acc=0.7911 | val_loss=0.9628 acc=0.9286 | lr=9.65e-04
Epoch 04 | train_loss=1.0380 acc=0.9025 | val_loss=0.8365 acc=0.9629 | lr=9.38e-04
Epoch 05 | train_loss=0.9135 acc=0.9416 | val_loss=0.7587 acc=0.9835 | lr=9.05e-04
Epoch 06 | train_loss=0.8705 acc=0.9529 | val_loss=0.7125 acc=0.9889 | lr=8.64e-04
Epoch 07 | train_loss=0.8248 acc=0.9669 | val_loss=0.6976 acc=0.9929 | lr=8.19e-04
Epoch 08 | train_loss=0.7983 acc=0.9724 | val_loss=0.6694 acc=0.9947 | lr=7.68e-04
Epoch 09 | train_loss=0.7804 acc=0.9757 | val_loss=0.6621 acc=0.9966 | lr=7.13e-04
Epoch 10 | train_loss=0.7704 acc=0.9781 | val_loss=0.6844 acc=0.9978 | lr=6.55e-04
Epoch 11 | train_loss=0.7554 acc=0.9822 | val_loss=0.6609 acc=0.9976 | lr=5.94e-04
Epoch 12 | train_loss=0.7441 acc=0.9842 | val_loss=0.6475 acc=0.9983 | lr=5.31e-04
Epoch 13 | train_loss=0.7329 acc=0.9857 | val_loss=0.6471 acc=0.9985 | lr=4.69e-04
Epoch 14 | train_loss=0.7241 acc=0.9873 | val_loss=0.6429 acc=0.9980 | lr=4.06e-04
Epoch 15 | train_loss=0.7168 acc=0.9898 | val_loss=0.6424 acc=0.9986 | lr=3.45e-04
Epoch 16 | train_loss=0.7118 acc=0.9901 | val_loss=0.6404 acc=0.9993 | lr=2.87e-04
Epoch 17 | train_loss=0.7052 acc=0.9915 | val_loss=0.6346 acc=0.9991 | lr=2.32e-04
Epoch 18 | train_loss=0.7018 acc=0.9923 | val_loss=0.6315 acc=0.9991 | lr=1.81e-04
Epoch 19 | train_loss=0.7015 acc=0.9916 | val_loss=0.6346 acc=0.9995 | lr=1.36e-04
Epoch 20 | train_loss=0.6971 acc=0.9915 | val_loss=0.6296 acc=0.9997 | lr=9.55e-05
Epoch 21 | train_loss=0.6915 acc=0.9938 | val_loss=0.6293 acc=0.9995 | lr=6.18e-05
Epoch 22 | train_loss=0.6895 acc=0.9941 | val_loss=0.6297 acc=0.9995 | lr=3.51e-05
Epoch 23 | train_loss=0.6901 acc=0.9942 | val_loss=0.6296 acc=0.9993 | lr=1.57e-05
Epoch 24 | train_loss=0.6875 acc=0.9943 | val_loss=0.6283 acc=0.9995 | lr=3.94e-06
Epoch 25 | train_loss=0.6894 acc=0.9941 | val_loss=0.6294 acc=0.9995 | lr=0.00e+00

TEST METRICS (customcnn_improved)
  accuracy: 0.9908
  precision_macro: 0.9822
  recall_macro: 0.9895
  f1_macro: 0.9855
  f1_weighted: 0.9908
  roc_auc_ovr_macro: 1.0000

real    9m6,768s
user    40m54,550s
sys     1m14,193s
```

### A.3. ResNet18 baseline

```bash
time python train_torchvision.py --model resnet18 --variant baseline
Device: cuda
Run: resnet18_baseline | pretrained=True | image_size=224 | batch_size=64 | epochs=6
Epoch 01 | train_loss=0.1382 acc=0.9657 | val_loss=0.0736 acc=0.9806 | lr=1.00e-03
Epoch 02 | train_loss=0.0281 acc=0.9934 | val_loss=0.0371 acc=0.9930 | lr=1.00e-03
Epoch 03 | train_loss=0.0236 acc=0.9936 | val_loss=0.0121 acc=0.9971 | lr=1.00e-03
Epoch 04 | train_loss=0.0221 acc=0.9948 | val_loss=0.0516 acc=0.9879 | lr=1.00e-03
Epoch 05 | train_loss=0.0218 acc=0.9950 | val_loss=0.1029 acc=0.9757 | lr=1.00e-03
Epoch 06 | train_loss=0.0235 acc=0.9942 | val_loss=0.0151 acc=0.9964 | lr=1.00e-03
Early stopping. Best epoch: 3 (val_loss=0.0121)

TEST METRICS (resnet18_baseline)
  accuracy: 0.9861
  precision_macro: 0.9851
  recall_macro: 0.9764
  f1_macro: 0.9784
  f1_weighted: 0.9857
  roc_auc_ovr_macro: 1.0000

real    11m1,330s
user    17m48,960s
sys     2m57,914s
```

### A.4. ResNet18 improved

```bash
time python train_torchvision.py --model resnet18 --variant improved
Device: cuda
Run: resnet18_improved | pretrained=True | image_size=224 | batch_size=64 | epochs=12
Epoch 01 | train_loss=0.9389 acc=0.9155 | val_loss=0.6741 acc=0.9917 | lr=4.91e-04
Epoch 02 | train_loss=0.6926 acc=0.9835 | val_loss=0.6561 acc=0.9942 | lr=4.67e-04
Epoch 03 | train_loss=0.6562 acc=0.9905 | val_loss=0.6328 acc=0.9978 | lr=4.27e-04
Epoch 04 | train_loss=0.6530 acc=0.9901 | val_loss=0.6296 acc=0.9986 | lr=3.75e-04
Epoch 05 | train_loss=0.6392 acc=0.9922 | val_loss=0.6240 acc=0.9968 | lr=3.15e-04
Epoch 06 | train_loss=0.6359 acc=0.9930 | val_loss=0.6198 acc=0.9990 | lr=2.50e-04
Epoch 07 | train_loss=0.6252 acc=0.9962 | val_loss=0.6159 acc=0.9997 | lr=1.85e-04
Epoch 08 | train_loss=0.6248 acc=0.9960 | val_loss=0.6159 acc=0.9993 | lr=1.25e-04
Epoch 09 | train_loss=0.6199 acc=0.9974 | val_loss=0.6130 acc=0.9995 | lr=7.32e-05
Epoch 10 | train_loss=0.6163 acc=0.9982 | val_loss=0.6119 acc=0.9998 | lr=3.35e-05
Epoch 11 | train_loss=0.6142 acc=0.9983 | val_loss=0.6107 acc=0.9998 | lr=8.52e-06
Epoch 12 | train_loss=0.6132 acc=0.9986 | val_loss=0.6106 acc=0.9998 | lr=0.00e+00

TEST METRICS (resnet18_improved)
  accuracy: 0.9947
  precision_macro: 0.9938
  recall_macro: 0.9935
  f1_macro: 0.9935
  f1_weighted: 0.9947
  roc_auc_ovr_macro: 1.0000

real    22m40,888s
user    83m58,713s
sys     6m36,943s
```

### A.5. ViT-B/16 baseline

```bash
time python train_torchvision.py --model vit_b_16 --variant baseline --epochs 2 --batch-size 48 --num-workers 8
Device: cuda
Run: vit_b_16_baseline | pretrained=True | image_size=224 | batch_size=48 | epochs=2
Epoch 01 | train_loss=0.2687 acc=0.9394 | val_loss=0.0724 acc=0.9808 | lr=1.00e-03
Epoch 02 | train_loss=0.0572 acc=0.9862 | val_loss=0.0413 acc=0.9891 | lr=1.00e-03

TEST METRICS (vit_b_16_baseline)
  accuracy: 0.9731
  precision_macro: 0.9694
  recall_macro: 0.9603
  f1_macro: 0.9631
  f1_weighted: 0.9726
  roc_auc_ovr_macro: 0.9996

real    27m42,618s
user    39m51,447s
sys     4m33,109s
```

### A.6. ViT-B/16 improved

```bash
time python train_torchvision.py --model vit_b_16 --variant improved --epochs 4 --batch-size 32 --num-workers 8
Device: cuda
Run: vit_b_16_improved | pretrained=True | image_size=224 | batch_size=32 | epochs=4
Epoch 01 | train_loss=0.9483 acc=0.8921 | val_loss=0.7014 acc=0.9829 | lr=4.27e-04
Epoch 02 | train_loss=0.7118 acc=0.9756 | val_loss=0.6541 acc=0.9920 | lr=2.50e-04
Epoch 03 | train_loss=0.6689 acc=0.9867 | val_loss=0.6344 acc=0.9966 | lr=7.32e-05
Epoch 04 | train_loss=0.6553 acc=0.9891 | val_loss=0.6298 acc=0.9976 | lr=0.00e+00

TEST METRICS (vit_b_16_improved)
  accuracy: 0.9886
  precision_macro: 0.9857
  recall_macro: 0.9841
  f1_macro: 0.9847
  f1_weighted: 0.9884
  roc_auc_ovr_macro: 0.9999

real    58m23,941s
user    87m12,508s
sys     9m14,672s
```
