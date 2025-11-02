# ML5 — Decision Trees (CART), Random Forest, GBDT

Проект посвящён деревьям решений и ансамблевым методам (Random Forest, Gradient Boosting): их принципам, применению и практической реализации на Python с нуля.

## 🗂 Структура проекта

```
ML5_Decision-trees/
├── src/                     
│   └── algos/algoritms.py      # Реализации алгоритмов моделей
│   │   
│   └── helpers/
│   │   └── datasplit.py        # Логика разделения датасета на части train, val и test
│   │   │
│   │   └── pipeline_manager.py # Препроцессинг данных, создание пайплайна, оценка модели и подбор гиперпараметров
│   │
│   └── main.ipynb
│
├── datasets/
│   └── data/                   # Датасеты 
```

## Установка зависимостей
Windows:
```powershell
cd path\to\ML_project4
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
cd /path/to/ML_project4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirments.txt
```
