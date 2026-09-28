# Первый эксперимент с CatBoostRanker

Контрольные RRF совпали с расширенной проверкой: 284112 парных строк, расхождений 0.

Цель — проверить, улучшает ли обучаемый отбор Recall@50 при том же поисковом индексе. Модели обучены на out-of-fold примерах внутри исходного train; dev содержит полный пул. Test рассчитан после решения по двум сценариям dev и служит диагностикой.

## Исходные условия

- Контрольные варианты: отправленный `rrf_history_half` и текстовый `rrf_geo16`.
- Текстовый пул: топ-500 BM25 и символьного поиска с географией, а также их вариантов со статистиками. Общий пул добавляет топ-500 исторического канала. Повторы item_id удалены.
- Для каждого пула показан собственный RRF с равными весами его источников. Отдельно сравниваются два CatBoostRanker: текстовый набор признаков и полный набор с историей. Текстовая модель применяется к двум пулам без переобучения.
- Обучение: 3000 OOF-запросов, 2691 групп с найденным ответом; 309 групп без него исключены только из PairLogit. Macro полнота обучающего пула по всем выбранным OOF-запросам: 0.8877.
- На запрос оставлены все найденные положительные ответы и не более 150 альтернатив. Нули — неразмеченные объявления, часть из них может быть подходящей.
- CatBoostRanker 1.2.10, CPU, PairLogit: 240 деревьев, глубина 6, learning_rate 0.06, seed 20260926. [Документация PairLogit](https://catboost.ai/docs/en/concepts/loss-functions-ranking).

## Метрики

C@pool — macro полнота объединения выбранных каналов до сокращения до 50. Потери отбора — положительные объявления, которые есть в пуле, но не попали в финальные 50.

| Часть | Корпус | Вариант | Recall@50 | C@pool | Потери отбора | Пропуски поиска |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| dev | benchmark_plus_positives | catboost_full_no_history_features | 0.7890 | 0.9269 | 6796 | 3175 |
| dev | benchmark_plus_positives | catboost_full_with_history_features | 0.7734 | 0.9269 | 7578 | 3175 |
| dev | benchmark_plus_positives | catboost_text_pool | 0.7885 | 0.9213 | 6513 | 3485 |
| dev | benchmark_plus_positives | rrf_full_pool | 0.7601 | 0.9269 | 7904 | 3175 |
| dev | benchmark_plus_positives | rrf_geo16 | 0.7529 | 0.9185 | 7853 | 3638 |
| dev | benchmark_plus_positives | rrf_history_half | 0.7604 | 0.9262 | 7856 | 3211 |
| dev | benchmark_plus_positives | rrf_text_pool | 0.7577 | 0.9213 | 7745 | 3485 |
| dev | union | catboost_full_no_history_features | 0.7180 | 0.8818 | 8269 | 5332 |
| dev | union | catboost_full_with_history_features | 0.7247 | 0.8818 | 7914 | 5332 |
| dev | union | catboost_text_pool | 0.7177 | 0.8741 | 7898 | 5723 |
| dev | union | rrf_full_pool | 0.6675 | 0.8818 | 10266 | 5332 |
| dev | union | rrf_geo16 | 0.6735 | 0.8706 | 9485 | 5900 |
| dev | union | rrf_history_half | 0.6602 | 0.8749 | 10258 | 5648 |
| dev | union | rrf_text_pool | 0.6744 | 0.8741 | 9621 | 5723 |
| test | benchmark_plus_positives | catboost_full_no_history_features | 0.7943 | 0.9282 | 6521 | 3066 |
| test | benchmark_plus_positives | catboost_full_with_history_features | 0.7771 | 0.9282 | 7326 | 3066 |
| test | benchmark_plus_positives | catboost_text_pool | 0.7939 | 0.9225 | 6252 | 3356 |
| test | benchmark_plus_positives | rrf_full_pool | 0.7654 | 0.9282 | 7703 | 3066 |
| test | benchmark_plus_positives | rrf_geo16 | 0.7568 | 0.9195 | 7756 | 3521 |
| test | benchmark_plus_positives | rrf_history_half | 0.7655 | 0.9274 | 7660 | 3105 |
| test | benchmark_plus_positives | rrf_text_pool | 0.7621 | 0.9225 | 7641 | 3356 |
| test | union | catboost_full_no_history_features | 0.7252 | 0.8850 | 8097 | 5046 |
| test | union | catboost_full_with_history_features | 0.7331 | 0.8850 | 7693 | 5046 |
| test | union | catboost_text_pool | 0.7249 | 0.8772 | 7719 | 5439 |
| test | union | rrf_full_pool | 0.6709 | 0.8850 | 10268 | 5046 |
| test | union | rrf_geo16 | 0.6780 | 0.8738 | 9451 | 5615 |
| test | union | rrf_history_half | 0.6639 | 0.8785 | 10260 | 5347 |
| test | union | rrf_text_pool | 0.6776 | 0.8772 | 9622 | 5439 |

Первичный максимум union/dev был у `catboost_full_with_history_features` (0.7247), но на benchmark + ответы он дал 0.7734 против 0.7890 у выбранной модели. Для переноса зафиксировано правило: среди моделей в пределах 0.01 Recall@50 от лучшей во втором сценарии dev выбрать максимум union/dev. Это решение сохранено в `dev_decision.json` до просмотра test.

Зафиксированный обученный вариант выбран по dev:

`catboost_full_no_history_features` — union/dev 0.7180; лучший контроль `rrf_geo16` — 0.6735. Разница во втором сценарии dev относительно лучшего контроля: +0.0287.

## Парные сравнения

Разница считается по одним и тем же запросам. Интервал — 1000 bootstrap-повторов с выборкой целых кластеров нормализованного текста, без поправки на несколько сравнений.

| Часть | Корпус | Вариант − контроль | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |
| --- | --- | --- | ---: | --- | ---: |
| dev | benchmark_plus_positives | catboost_full_no_history_features − catboost_text_pool | +0.0005 | [+0.0003, +0.0008] | 27 / 0 |
| dev | benchmark_plus_positives | catboost_full_no_history_features − rrf_full_pool | +0.0289 | [+0.0241, +0.0337] | 3500 / 2297 |
| dev | benchmark_plus_positives | catboost_full_no_history_features − rrf_geo16 | +0.0361 | [+0.0316, +0.0411] | 3525 / 2010 |
| dev | benchmark_plus_positives | catboost_full_no_history_features − rrf_history_half | +0.0287 | [+0.0238, +0.0336] | 3536 / 2346 |
| dev | benchmark_plus_positives | catboost_full_with_history_features − catboost_full_no_history_features | -0.0157 | [-0.0177, -0.0137] | 458 / 1153 |
| dev | union | catboost_full_no_history_features − catboost_text_pool | +0.0003 | [+0.0002, +0.0006] | 20 / 0 |
| dev | union | catboost_full_no_history_features − rrf_full_pool | +0.0506 | [+0.0458, +0.0552] | 4212 / 2222 |
| dev | union | catboost_full_no_history_features − rrf_geo16 | +0.0446 | [+0.0400, +0.0492] | 3985 / 2199 |
| dev | union | catboost_full_no_history_features − rrf_history_half | +0.0578 | [+0.0529, +0.0627] | 4422 / 2126 |
| dev | union | catboost_full_with_history_features − catboost_full_no_history_features | +0.0067 | [+0.0045, +0.0088] | 1153 / 852 |
| test | benchmark_plus_positives | catboost_full_no_history_features − catboost_text_pool | +0.0004 | [+0.0002, +0.0007] | 22 / 1 |
| test | benchmark_plus_positives | catboost_full_no_history_features − rrf_full_pool | +0.0289 | [+0.0239, +0.0337] | 3411 / 2187 |
| test | benchmark_plus_positives | catboost_full_no_history_features − rrf_geo16 | +0.0375 | [+0.0330, +0.0419] | 3492 / 1897 |
| test | benchmark_plus_positives | catboost_full_no_history_features − rrf_history_half | +0.0288 | [+0.0237, +0.0335] | 3460 / 2225 |
| test | benchmark_plus_positives | catboost_full_with_history_features − catboost_full_no_history_features | -0.0172 | [-0.0193, -0.0153] | 428 / 1160 |
| test | union | catboost_full_no_history_features − catboost_text_pool | +0.0003 | [+0.0001, +0.0005] | 17 / 2 |
| test | union | catboost_full_no_history_features − rrf_full_pool | +0.0543 | [+0.0493, +0.0587] | 4214 / 2069 |
| test | union | catboost_full_no_history_features − rrf_geo16 | +0.0472 | [+0.0423, +0.0516] | 3954 / 2076 |
| test | union | catboost_full_no_history_features − rrf_history_half | +0.0613 | [+0.0564, +0.0658] | 4417 / 1999 |
| test | union | catboost_full_with_history_features − catboost_full_no_history_features | +0.0079 | [+0.0060, +0.0099] | 1119 / 775 |

## Срезы правильных объявлений

Для каждого сегмента ниже приведён macro Recall@50 по запросам, в которых есть хотя бы один ответ этого типа. `seen_item` означает выбор этого item_id в истории обучающей части; `history_source` означает попадание в топ-500 исторического канала. Это диагностические разбиения ответов, не признаки доступности в benchmark.

| Часть | Корпус | Вариант | С историей item | Без истории item | Локация совпадает | Локация различается | Исторический канал | Вне канала |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | benchmark_plus_positives | catboost_full_no_history_features | 0.8047 | 0.7849 | 0.9201 | 0.2543 | 0.8283 | 0.7816 |
| dev | benchmark_plus_positives | catboost_full_with_history_features | 0.7565 | 0.7884 | 0.9109 | 0.2139 | 0.7833 | 0.7739 |
| dev | benchmark_plus_positives | catboost_text_pool | 0.8032 | 0.7849 | 0.9195 | 0.2540 | 0.8253 | 0.7816 |
| dev | benchmark_plus_positives | rrf_full_pool | 0.8089 | 0.7369 | 0.8564 | 0.3658 | 0.9067 | 0.7245 |
| dev | benchmark_plus_positives | rrf_geo16 | 0.7664 | 0.7499 | 0.8566 | 0.3289 | 0.8328 | 0.7350 |
| dev | benchmark_plus_positives | rrf_history_half | 0.8172 | 0.7324 | 0.8551 | 0.3724 | 0.9161 | 0.7224 |
| dev | benchmark_plus_positives | rrf_text_pool | 0.7813 | 0.7490 | 0.8594 | 0.3420 | 0.8491 | 0.7368 |
| dev | union | catboost_full_no_history_features | 0.7229 | 0.7193 | 0.8627 | 0.1315 | 0.7659 | 0.7120 |
| dev | union | catboost_full_with_history_features | 0.6896 | 0.7492 | 0.8653 | 0.1542 | 0.7343 | 0.7251 |
| dev | union | catboost_text_pool | 0.7218 | 0.7193 | 0.8623 | 0.1315 | 0.7629 | 0.7120 |
| dev | union | rrf_full_pool | 0.6950 | 0.6574 | 0.7883 | 0.1764 | 0.8427 | 0.6408 |
| dev | union | rrf_geo16 | 0.6681 | 0.6821 | 0.7946 | 0.1809 | 0.7598 | 0.6616 |
| dev | union | rrf_history_half | 0.6987 | 0.6438 | 0.7815 | 0.1676 | 0.8479 | 0.6314 |
| dev | union | rrf_text_pool | 0.6815 | 0.6754 | 0.7966 | 0.1773 | 0.7813 | 0.6590 |
| test | benchmark_plus_positives | catboost_full_no_history_features | 0.8076 | 0.7906 | 0.9252 | 0.2596 | 0.8316 | 0.7868 |
| test | benchmark_plus_positives | catboost_full_with_history_features | 0.7582 | 0.7926 | 0.9157 | 0.2134 | 0.7895 | 0.7766 |
| test | benchmark_plus_positives | catboost_text_pool | 0.8063 | 0.7906 | 0.9247 | 0.2594 | 0.8295 | 0.7868 |
| test | benchmark_plus_positives | rrf_full_pool | 0.8095 | 0.7441 | 0.8636 | 0.3613 | 0.9085 | 0.7312 |
| test | benchmark_plus_positives | rrf_geo16 | 0.7644 | 0.7562 | 0.8627 | 0.3216 | 0.8321 | 0.7400 |
| test | benchmark_plus_positives | rrf_history_half | 0.8175 | 0.7397 | 0.8621 | 0.3679 | 0.9156 | 0.7293 |
| test | benchmark_plus_positives | rrf_text_pool | 0.7808 | 0.7552 | 0.8658 | 0.3362 | 0.8494 | 0.7422 |
| test | union | catboost_full_no_history_features | 0.7231 | 0.7290 | 0.8716 | 0.1318 | 0.7686 | 0.7191 |
| test | union | catboost_full_with_history_features | 0.6936 | 0.7582 | 0.8746 | 0.1584 | 0.7364 | 0.7334 |
| test | union | catboost_text_pool | 0.7223 | 0.7290 | 0.8712 | 0.1319 | 0.7665 | 0.7191 |
| test | union | rrf_full_pool | 0.6944 | 0.6617 | 0.7937 | 0.1714 | 0.8430 | 0.6447 |
| test | union | rrf_geo16 | 0.6670 | 0.6881 | 0.8016 | 0.1751 | 0.7594 | 0.6663 |
| test | union | rrf_history_half | 0.6984 | 0.6485 | 0.7870 | 0.1636 | 0.8495 | 0.6355 |
| test | union | rrf_text_pool | 0.6784 | 0.6810 | 0.8023 | 0.1702 | 0.7762 | 0.6632 |
На test/union разница выбранной модели с `rrf_geo16` составляет +0.0700 при совпадающей локации и -0.0434 при различающейся. Последний сегмент остаётся слабым местом; правило для выездных услуг требует нового dev-эксперимента.
На test/benchmark_plus_positives разница выбранной модели с `rrf_geo16` составляет +0.0625 при совпадающей локации и -0.0620 при различающейся. Последний сегмент остаётся слабым местом; правило для выездных услуг требует нового dev-эксперимента.

Счётчики ответов, которые попали в пул только через историю, находятся в `slices.csv` (`history_only_positives` и `history_only_found`). Они показывают дополнительную полноту этого источника отдельно от исторических признаков.

## Утечки и воспроизводимость

Для каждой из трёх внутренних частей история и популярность вычисляются по двум другим. Обучающая положительная пара не участвует в собственных статистиках. Описание и заголовок объявления индексируются как доступный корпус без использования меток. Признаков принадлежности benchmark или добавления как положительного ответа в модель нет. В сценарии benchmark + ответы дополняется только допустимый корпус поиска; эта информация остаётся в диагностике и не поступает в таблицу признаков.

Признаков: текстовая модель 41, полный набор 56. В `ranker.py` зафиксированы их имена и порядок, включая исходные оценки BM25 до множителя географии и отдельные признаки отсутствия в источниках.

Артефакты: `artifacts/ranker/full/c392362a99e2c4d1`. `selection.json` содержит первичный максимум union/dev; `dev_decision.json` фиксирует переносоустойчивый выбор по обоим сценариям dev до чтения test. `models.json` содержит параметры и SHA256 весов; `oof_*.json` — состав обучающих примеров и полноту пула. Отдельные файлы `dev/per_query.parquet`, `test/per_query.parquet`, `metrics.csv`, `slices.csv` и `paired_comparisons.json` позволяют проверить выводы. `feature_importance.csv` содержит PredictionValuesChange для двух моделей; это описание использованных моделью разбиений, не причинный эффект признака.

## Вычислительные затраты

- dev: 1792.6 с для 35652 запросов и двух схем; пик рабочего набора процесса-работника 2.96 GiB; работников 4.
- test: 1745.1 с для 35376 запросов и двух схем; пик рабочего набора процесса-работника 2.94 GiB; работников 4.
- OOF-признаки: 147.5 с; обучение моделей: 142.7 с. Время подготовки общего поискового индекса показано в расширенном отчёте.

- Предсказание benchmark: 220.7 с для 2452 запросов после загрузки индекса; полное время команды 230.7 с; пик рабочего набора 3.91 GiB.
- Итоговый CSV: `answer_ranker.csv`, SHA256 `1844e8dc91fd5d562aed1ca79f34598105bc3d6e3c8497a12ebb3c0dcd2be44e`; проверка формата: 50–50 кандидатов на запрос.

Повторный запуск дал побайтно идентичный CSV: SHA256 совпал.

Команда запуска:

```powershell
uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli ranker --workers 4
uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli predict-ranker --ranker-answer answer_ranker.csv
```
