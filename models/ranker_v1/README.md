# Зафиксированная модель для `answer_ranker.csv`

В комплекте сохранены выбранные веса `text_model.cbm`, альтернативные веса
`history_model.cbm`, полный обученный сигнал истории и популярности
`all_train_signals.joblib`, а также `config.json`. Для отправленного файла
используется `text_model.cbm` на общем пуле из пяти поисковых каналов; история
даёт кандидатов, но её признаки не подаются в модель. `history_model.cbm`
сохранён для воспроизведения сравнения моделей.

`config.json` фиксирует выбор по dev, параметры CatBoost, порядок признаков,
поисковую конфигурацию, SHA256 каждого артефакта и трёх исходных Parquet-файлов,
версию `uv.lock` и ожидаемый SHA256 ответа. Исходные Parquet-файлы должны
находиться в `src/avito_test/`. Поисковый индекс строится из них локально или
загружается из проверенного кэша; он не содержит обученных меток.

Из корня проекта:

```powershell
uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli predict-ranker --ranker-package models/ranker_v1 --ranker-answer answer_ranker_from_package.csv
```

Команда проверяет хэши данных, кода, весов и сигналов, а затем формат CSV и его
SHA256. Ожидаемый SHA256: `1844e8dc91fd5d562aed1ca79f34598105bc3d6e3c8497a12ebb3c0dcd2be44e`.
Сетевые API не используются. Для пересоздания комплекта из исходного полного
запуска служит команда `avito_test.cli save-ranker-bundle`; её аргументы
описаны в основном `README.md`.
