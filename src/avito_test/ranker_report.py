"""Results of one fixed CatBoostRanker experiment; no test-driven selection."""
# Summaries are computed from saved per-query results, including missed queries.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRanker

from .data import SEED, sha256_file, write_json
from .ranker import FEATURES, HISTORY_FEATURES, TEXT_FEATURES


def comparisons(frame: pl.DataFrame, selected: str) -> list[dict]:
    keys = ["stage", "scheme", "group_id"]
    pairs = [("catboost_full_no_history_features", "catboost_text_pool"),
             ("catboost_full_with_history_features", "catboost_full_no_history_features"),
             (selected, "rrf_history_half"), (selected, "rrf_geo16"),
             (selected, "rrf_full_pool")]
    result = []
    for winner, reference in dict.fromkeys(pairs):
        a = frame.filter(pl.col("variant") == winner).select(*keys, "text", "recall@50")
        b = frame.filter(pl.col("variant") == reference).select(*keys, pl.col("recall@50").alias("baseline"))
        paired = a.join(b, on=keys, validate="1:1").with_columns(
            (pl.col("recall@50") - pl.col("baseline")).alias("delta"))
        for (stage, scheme), part in paired.group_by("stage", "scheme"):
            clusters = part.group_by("text").agg(pl.col("delta").sum().alias("sum"), pl.len().alias("count")).sort("text")
            values, sizes = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
            rng = np.random.default_rng(SEED)
            samples = []
            for _ in range(1000):
                draw = rng.integers(0, len(values), len(values))
                samples.append(values[draw].sum() / sizes[draw].sum())
            result.append({"stage": stage, "scheme": scheme, "variant": winner,
                           "control": reference, "queries": part.height,
                           "delta_recall50": part["delta"].mean(),
                           "ci95": np.quantile(samples, [.025, .975]).tolist(),
                           "improved_queries": (part["delta"] > 0).sum(),
                           "worsened_queries": (part["delta"] < 0).sum()})
    return sorted(result, key=lambda row: (row["stage"], row["scheme"], row["variant"], row["control"]))


def write_ranker_report(run: Path, destination: Path, frame: pl.DataFrame) -> None:
    result = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    oof = json.loads(next(run.glob("oof_*.json")).read_text(encoding="utf-8"))
    models = json.loads((run / "models.json").read_text(encoding="utf-8"))
    importances = []
    for name in ("text", "history"):
        model = CatBoostRanker()
        model.load_model(str(run / f"{name}_model.cbm"))
        importances.extend({"model": name, "feature": feature, "importance": float(importance)}
                           for feature, importance in zip(
                               model.feature_names_, model.get_feature_importance(type="PredictionValuesChange")))
    pl.DataFrame(importances).sort("model", "importance", descending=[False, True]).write_csv(
        run / "feature_importance.csv")
    decision_path = run / "dev_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else None
    if decision and sha256_file(run / "dev" / "per_query.parquet") != decision["dev_metrics_sha256"]:
        raise ValueError("Dev decision does not match saved metrics")
    selected = decision["selected_variant"] if decision else result["config"]["variant"]
    selected_union = (decision["selected_union_dev_recall50"] if decision else
                      result["config"]["union_dev_recall50"])
    selected_augmented = (decision["selected_augmented_dev_recall50"] if decision else
                          result["config"]["augmented_dev_recall50"])
    table = pl.read_csv(run / "metrics.csv")
    paired = comparisons(frame, selected)
    write_json(run / "paired_comparisons.json", paired)
    expanded_path = run.parents[2] / "expanded" / "dev_test" / "per_query.parquet"
    control_agreement = None
    if expanded_path.exists():
        keys = ["group_id", "stage", "scheme", "experiment"]
        current = frame.filter(pl.col("variant").is_in(["rrf_history_half", "rrf_geo16"])).select(
            "group_id", "stage", "scheme", pl.col("variant").alias("experiment"),
            pl.col("recall@50").alias("new_recall"))
        previous = pl.read_parquet(expanded_path).filter(pl.col("experiment").is_in(
            ["rrf_history_half", "rrf_geo16"])).select(*keys, "recall@50")
        joined = current.join(previous, on=keys, validate="1:1")
        disagreement = joined.filter((pl.col("new_recall") - pl.col("recall@50")).abs() > 1e-8).height
        if joined.height != current.height or disagreement:
            raise ValueError("Frozen RRF controls differ from expanded validation")
        control_agreement = {"compared_rows": joined.height, "different_recall_rows": disagreement}
        write_json(run / "control_agreement.json", control_agreement)
    segments = ["seen_item", "new_item", "same_location", "other_location",
                "history_source", "outside_history_source"]
    slices = frame.group_by("stage", "scheme", "variant").agg(
        *[pl.col(name + "_recall@50").mean() for name in segments],
        *[pl.col(name + "_positives").sum() for name in segments],
        *[pl.col(name + "_found").sum() for name in segments],
        pl.col("history_only_positives").sum(),
        pl.col("history_only_found").sum(),
    ).sort("stage", "scheme", "variant")
    slices.write_csv(run / "slices.csv")
    lines = ["# Первый эксперимент с CatBoostRanker", "",
             "Цель — проверить, улучшает ли обучаемый отбор Recall@50 при том же поисковом индексе. "
             "Модели обучены на out-of-fold примерах внутри исходного train; dev содержит полный пул. "
             "Test рассчитан после решения по двум сценариям dev и служит диагностикой.", "",
             "## Исходные условия", "",
             "- Контрольные варианты: отправленный `rrf_history_half` и текстовый `rrf_geo16`.",
             "- Текстовый пул: топ-500 BM25 и символьного поиска с географией, а также их вариантов "
             "со статистиками. Общий пул добавляет топ-500 исторического канала. Повторы item_id удалены.",
             "- Для каждого пула показан собственный RRF с равными весами его источников. "
             "Отдельно сравниваются два CatBoostRanker: текстовый набор признаков и полный набор "
             "с историей. Текстовая модель применяется к двум пулам без переобучения.",
             f"- Обучение: {oof['requested_queries']} OOF-запросов, "
             f"{oof['queries_used_for_training']} групп с найденным ответом; "
             f"{oof['groups_with_no_found_positive']} групп без него исключены только из PairLogit. "
             f"Macro полнота обучающего пула по всем выбранным OOF-запросам: {oof['macro_pool_recall']:.4f}.",
             f"- На запрос оставлены все найденные положительные ответы и не более "
             f"{oof['negative_limit_per_query']} альтернатив. Нули — неразмеченные объявления, "
             "часть из них может быть подходящей.",
             f"- CatBoostRanker 1.2.10, CPU, PairLogit: {models['params']['iterations']} деревьев, "
             f"глубина {models['params']['depth']}, learning_rate {models['params']['learning_rate']}, "
             f"seed {SEED}. [Документация PairLogit](https://catboost.ai/docs/en/concepts/loss-functions-ranking).", "",
             "## Метрики", "",
             "C@pool — macro полнота объединения выбранных каналов до сокращения до 50. "
             "Потери отбора — положительные объявления, которые есть в пуле, но не попали в финальные 50.", "",
             "| Часть | Корпус | Вариант | Recall@50 | C@pool | Потери отбора | Пропуски поиска |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in table.iter_rows(named=True):
        lines.append(f"| {row['stage']} | {row['scheme']} | {row['variant']} | "
                     f"{row['recall@50']:.4f} | {row['candidate_recall']:.4f} | "
                     f"{row['lost_at_selection']} | {row['missed_by_sources']} |")
    if decision:
        lines += ["", "Первичный максимум union/dev был у "
                  f"`{decision['union_only_variant']}` ({decision['union_only_union_dev_recall50']:.4f}), "
                  f"но на benchmark + ответы он дал {decision['union_only_augmented_dev_recall50']:.4f} "
                  f"против {decision['selected_augmented_dev_recall50']:.4f} у выбранной модели. "
                  "Для переноса зафиксировано правило: среди моделей в пределах 0.01 Recall@50 "
                  "от лучшей во втором сценарии dev выбрать максимум union/dev. "
                  "Это решение сохранено в `dev_decision.json` до просмотра test."]
    lines += ["", "Зафиксированный обученный вариант выбран по dev:", "",
              f"`{selected}` — union/dev {selected_union:.4f}; "
              f"лучший контроль `{result['config']['best_control']}` — "
              f"{result['config']['best_control_union_dev_recall50']:.4f}. "
              f"Разница во втором сценарии dev относительно лучшего контроля: "
              f"{selected_augmented - result['config']['best_control_augmented_dev_recall50']:+.4f}.", "",
              "## Парные сравнения", "",
              "Разница считается по одним и тем же запросам. Интервал — 1000 bootstrap-повторов "
              "с выборкой целых кластеров нормализованного текста, без поправки на несколько сравнений.", "",
              "| Часть | Корпус | Вариант − контроль | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
              "| --- | --- | --- | ---: | --- | ---: |"]
    for row in paired:
        lines.append(f"| {row['stage']} | {row['scheme']} | {row['variant']} − {row['control']} | "
                     f"{row['delta_recall50']:+.4f} | "
                     f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | "
                     f"{row['improved_queries']} / {row['worsened_queries']} |")
    lines += ["", "## Срезы правильных объявлений", "",
              "Для каждого сегмента ниже приведён macro Recall@50 по запросам, в которых есть хотя бы "
              "один ответ этого типа. `seen_item` означает выбор этого item_id в истории обучающей части; "
              "`history_source` означает попадание в топ-500 исторического канала. "
              "Это диагностические разбиения ответов, не признаки доступности в benchmark.", "",
              "| Часть | Корпус | Вариант | С историей item | Без истории item | Локация совпадает | Локация различается | Исторический канал | Вне канала |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in slices.iter_rows(named=True):
        def f(name):
            value = row[name + "_recall@50"]
            return "—" if value is None else f"{value:.4f}"
        lines.append(f"| {row['stage']} | {row['scheme']} | {row['variant']} | "
                     f"{f('seen_item')} | {f('new_item')} | {f('same_location')} | "
                     f"{f('other_location')} | {f('history_source')} | "
                     f"{f('outside_history_source')} |")
    if "test" in slices["stage"].to_list():
        for scheme in ("union", "benchmark_plus_positives"):
            selected_row = slices.filter((pl.col("stage") == "test") &
                                         (pl.col("scheme") == scheme) &
                                         (pl.col("variant") == selected)).row(0, named=True)
            geo_row = slices.filter((pl.col("stage") == "test") &
                                    (pl.col("scheme") == scheme) &
                                    (pl.col("variant") == "rrf_geo16")).row(0, named=True)
            lines.append(f"На test/{scheme} разница выбранной модели с `rrf_geo16` составляет "
                         f"{selected_row['same_location_recall@50'] - geo_row['same_location_recall@50']:+.4f} "
                         f"при совпадающей локации и "
                         f"{selected_row['other_location_recall@50'] - geo_row['other_location_recall@50']:+.4f} "
                         "при различающейся. Последний сегмент остаётся слабым местом; "
                         "правило для выездных услуг требует нового dev-эксперимента.")
    lines += ["", "Счётчики ответов, которые попали в пул только через историю, находятся в "
              "`slices.csv` (`history_only_positives` и `history_only_found`). Они показывают "
              "дополнительную полноту этого источника отдельно от исторических признаков.", "",
              "## Утечки и воспроизводимость", "",
              "Для каждой из трёх внутренних частей история и популярность вычисляются по двум другим. "
              "Обучающая положительная пара не участвует в собственных статистиках. "
              "Описание и заголовок объявления индексируются как доступный корпус без использования меток. "
              "Признаков принадлежности benchmark или добавления как положительного ответа в модель нет. "
              "В сценарии benchmark + ответы дополняется только допустимый корпус поиска; эта "
              "информация остаётся в диагностике и не поступает в таблицу признаков.", "",
              f"Признаков: текстовая модель {len(TEXT_FEATURES)}, полный набор {len(FEATURES)}. "
              "В `ranker.py` зафиксированы их имена и порядок, включая исходные оценки BM25 до "
              "множителя географии и отдельные признаки отсутствия в источниках.", "",
              f"Артефакты: `{run.as_posix()}`. "
              "`selection.json` содержит первичный максимум union/dev; `dev_decision.json` фиксирует "
              "переносоустойчивый выбор по обоим сценариям dev до чтения test. `models.json` содержит параметры "
              "и SHA256 весов; `oof_*.json` — состав обучающих примеров и полноту пула. "
              "Отдельные файлы `dev/per_query.parquet`, `test/per_query.parquet`, `metrics.csv`, "
              "`slices.csv` и `paired_comparisons.json` позволяют проверить выводы. "
              "`feature_importance.csv` содержит PredictionValuesChange для двух моделей; "
              "это описание использованных моделью разбиений, не причинный эффект признака.", "",
              "## Вычислительные затраты", ""]
    for stage in ["dev", "test"]:
        timing = result["timing"].get(stage)
        if timing:
            lines.append(f"- {stage}: {timing['elapsed_seconds']:.1f} с для {timing['queries']} запросов "
                         f"и двух схем; пик рабочего набора процесса-работника "
                         f"{timing['worker_peak_rss_bytes']/2**30:.2f} GiB; работников {timing['workers']}.")
    lines += [f"- OOF-признаки: {oof['elapsed_seconds']:.1f} с; обучение моделей: "
              f"{models['elapsed_seconds']:.1f} с. Время подготовки общего поискового индекса "
              "показано в расширенном отчёте.", ""]
    primary_manifest = run / "answer_ranker_manifest.json"
    prediction_manifests = ([primary_manifest] if primary_manifest.exists() else
                            sorted(run.glob("answer*_manifest.json")))
    if prediction_manifests:
        prediction = json.loads(prediction_manifests[-1].read_text(encoding="utf-8"))
        lines += [f"- Предсказание benchmark: {prediction['prediction_seconds']:.1f} с для "
                  f"{prediction['validation']['queries']} запросов после загрузки индекса; "
                  f"полное время команды {prediction['total_seconds']:.1f} с; "
                  f"пик рабочего набора {prediction['peak_rss_bytes']/2**30:.2f} GiB.",
                  f"- Итоговый CSV: `{prediction_manifests[-1].stem.removesuffix('_manifest')}.csv`, "
                  f"SHA256 `{prediction['answer_sha256']}`; проверка формата: "
                  f"{prediction['validation']['min_candidates']}–"
                  f"{prediction['validation']['max_candidates']} кандидатов на запрос.", ""]
        repeat_path = run / "answer_ranker_repeat_manifest.json"
        if repeat_path.exists() and primary_manifest.exists():
            repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
            if repeat["answer_sha256"] != prediction["answer_sha256"]:
                raise ValueError("Repeated benchmark prediction differs from the primary CSV")
            lines.append("Повторный запуск дал побайтно идентичный CSV: SHA256 совпал.\n")
    lines += ["Команда запуска:", "",
              "```powershell",
              "uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli ranker --workers 4",
              "uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli predict-ranker --ranker-answer answer_ranker.csv",
              "```", ""]
    if "smoke" in run.parts:
        lines.insert(2, "**Технический прогон на части запросов; не использовать как итоговую оценку.**\n")
    if control_agreement:
        lines.insert(2, f"Контрольные RRF совпали с расширенной проверкой: "
                        f"{control_agreement['compared_rows']} парных строк, расхождений 0.\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"CatBoost report: {destination}", flush=True)
