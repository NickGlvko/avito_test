"""Summaries of expanded validation. No parameter tuning or prediction changes."""
# Render stored metrics only; reporting must not change the selected configuration.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import SEED, sha256_file, write_json


def aggregate(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    means = ["recall@50", "recall@100", "recall@200", "recall@500", "candidate_recall",
             "candidate_oracle@50", "ceiling@50", "hit@50", "candidate_count",
             "outside_recall@50", "outside_candidate_recall", "inside_recall@50", "inside_candidate_recall"]
    sums = ["positives", "retrieved", "candidate_found", "lost_at_selection", "missed_by_sources",
            "unavoidable_candidate_cap_loss", "avoidable_selection_loss", "outside_positives",
            "outside_retrieved", "outside_candidate_found", "inside_positives", "inside_retrieved", "inside_candidate_found"]
    return frame.group_by(by).agg(pl.len().alias("queries"), *[pl.col(c).mean() for c in means],
                                 *[pl.col(c).sum() for c in sums],
                                 (pl.col("outside_positives") > 0).sum().alias("queries_with_outside"),
                                 (pl.col("inside_positives") > 0).sum().alias("queries_with_inside"),
                                 pl.col("corpus_size").min().alias("corpus_min"),
                                 pl.col("corpus_size").max().alias("corpus_max")).sort(by)


def history_comparisons(frame: pl.DataFrame) -> list[dict]:
    keys = ["stage", "scheme", "group_id"]
    current = frame.filter(pl.col("experiment") == "rrf_history_half")
    baseline = frame.filter(pl.col("experiment") == "rrf_stats").select(*keys,
        pl.col("recall@50").alias("without_history"), pl.col("outside_recall@50").alias("outside_without_history"))
    paired = current.join(baseline, on=keys, validate="1:1").with_columns(
        (pl.col("recall@50") - pl.col("without_history")).alias("delta"),
        (pl.col("outside_recall@50") - pl.col("outside_without_history")).alias("outside_delta"))
    results = []
    for (stage, scheme), group in paired.group_by("stage", "scheme", maintain_order=True):
        for cohort, subset, column in [
            ("all", group, "delta"),
            ("all_positives_in_benchmark", group.filter(pl.col("all_in_benchmark")), "delta"),
            ("has_positives_outside", group.filter(~pl.col("all_in_benchmark")), "delta"),
            ("outside_positives_only", group.filter(pl.col("outside_positives") > 0), "outside_delta"),
        ]:
            if not subset.height:
                continue
            # Cluster bootstrap by normalized text, preserving the macro-query estimand.
            clusters = subset.group_by("text").agg(pl.col(column).sum().alias("sum"), pl.len().alias("count")).sort("text")
            sums, sizes = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
            rng = np.random.default_rng(SEED)
            samples = []
            for _ in range(1000):
                indices = rng.integers(0, len(sums), len(sums))
                samples.append(sums[indices].sum() / sizes[indices].sum())
            delta = subset[column]
            results.append({"stage": stage, "scheme": scheme, "cohort": cohort,
                            "queries": subset.height, "text_clusters": clusters.height,
                            "delta_recall50": delta.mean(), "ci95": np.quantile(samples, [.025, .975]).tolist(),
                            "improved_queries": (delta > 0).sum(), "worse_queries": (delta < 0).sum()})
    return results


def write_extended_report(run_dir: Path, destination: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise ValueError("Cannot report an incomplete run")
    frame = pl.read_parquet(run_dir / "per_query.parquet")
    summary = aggregate(frame, ["stage", "scheme", "experiment"])
    summary.write_csv(run_dir / "metrics.csv")
    cohorts = aggregate(frame, ["stage", "scheme", "experiment", "all_in_benchmark"])
    cohorts.write_csv(run_dir / "cohorts.csv")
    for feature in ["known_text", "all_items_seen", "has_filters"]:
        aggregate(frame, ["stage", "scheme", "experiment", feature]).write_csv(run_dir / f"slice_{feature}.csv")
    positive_rows = pl.read_parquet(run_dir / "positive_ranks.parquet")
    positive_rows.group_by("stage", "scheme", "in_benchmark", "seen_item", "failure_stage").len().sort(
        "stage", "scheme", "in_benchmark", "seen_item", "failure_stage").write_csv(run_dir / "positive_slices.csv")
    comparisons = history_comparisons(frame)
    write_json(run_dir / "history_comparisons.json", comparisons)
    write_json(run_dir / "summary.json", {"manifest": manifest, "metrics": summary.to_dicts(), "history": comparisons,
                                          "report_code_sha256": sha256_file(Path(__file__)),
                                          "bootstrap": {"unit": "normalized text", "replicates": 1000, "seed": SEED}})
    corpus = manifest["corpus"]
    reference_stage = "test" if "test" in frame["stage"].unique().to_list() else "dev"
    findings = [f"## Основные наблюдения ({reference_stage})", ""]
    chosen = summary.filter((pl.col("stage") == reference_stage) &
                            (pl.col("experiment") == "rrf_history_half"))
    for row in chosen.iter_rows(named=True):
        misses = row["lost_at_selection"] + row["missed_by_sources"]
        selection_share = row["lost_at_selection"] / misses if misses else 0.
        comparison = next((c for c in comparisons if c["stage"] == reference_stage and
                           c["scheme"] == row["scheme"] and c["cohort"] == "outside_positives_only"), None)
        history_finding = "В этой выборке нет положительных ответов вне benchmark."
        if comparison is not None:
            interval = comparison["ci95"]
            conclusion = ("выигрыш истории сохраняется" if interval[0] > 0 else
                          "история ухудшает результат" if interval[1] < 0 else
                          "устойчивый выигрыш истории не подтверждён")
            history_finding = (f"Для ответов вне benchmark {conclusion}: "
                               f"Δ Recall@50 {comparison['delta_recall50']:+.4f}, "
                               f"95% интервал [{interval[0]:+.4f}, {interval[1]:+.4f}].")
        findings.append(f"- **{row['scheme']}**: Recall@50 {row['recall@50']:.4f}, "
                        f"полнота объединения {row['candidate_recall']:.4f}. "
                        f"При отборе потеряно {row['lost_at_selection']} положительных ответов "
                        f"({row['lost_at_selection']/row['positives']:.1%} всех ответов, "
                        f"{selection_share:.1%} всех пропусков). {history_finding}")
    union_rows = summary.filter((pl.col("stage") == reference_stage) & (pl.col("scheme") == "union"))
    geography = union_rows.filter(pl.col("experiment") == "rrf_geo16")
    if geography.height:
        findings.append(f"- Текстовый RRF с географией в union даёт Recall@50 "
                        f"{geography['recall@50'][0]:.4f}; RRF со статистиками — "
                        f"{union_rows.filter(pl.col('experiment') == 'rrf_stats')['recall@50'][0]:.4f}. "
                        "В этой схеме популярность и история не улучшили отбор первых 50.")
    same_groups = cohorts.filter((pl.col("stage") == reference_stage) &
                                 (pl.col("scheme") == "union") &
                                 (pl.col("experiment") == "rrf_history_half") &
                                 pl.col("all_in_benchmark"))
    if same_groups.height:
        matched = same_groups.row(0, named=True)
        findings.append(f"- На прежней узкой когорте при поиске в union потерь отбора "
                        f"{matched['lost_at_selection']} из {matched['positives']} ответов "
                        f"({matched['lost_at_selection']/matched['positives']:.1%}); "
                        "старое значение — 314 из 1875 (16.7%). Дополнительные конкуренты "
                        "усиливают проблему даже на одних и тех же запросах.")
        matched_augmented = cohorts.filter((pl.col("stage") == reference_stage) &
                                            (pl.col("scheme") == "benchmark_plus_positives") &
                                            (pl.col("experiment") == "rrf_history_half") &
                                            pl.col("all_in_benchmark"))
        if matched_augmented.height:
            augmented = matched_augmented.row(0, named=True)
            findings.append(f"- На этих же {matched['queries']} запросах при общем текстовом индексе "
                            f"Recall@50 падает с {augmented['recall@50']:.4f} "
                            f"(benchmark + ответы) до {matched['recall@50']:.4f} (union). "
                            "Это прямое сравнение влияния доступного корпуса при фиксированных "
                            "словарях, IDF и обучающих взаимодействиях.")
    findings += ["", "В исходном узком test потери отбора составляли 314 / 1875 = 16.7% "
                 "всех положительных ответов и 314 / (314 + 135) = 69.9% пропусков. "
                 "Сравнивать только абсолютные количества при разных размерах выборок нельзя.", "",
                 "Численная близость union и результата платформы сама по себе не "
                 "подтверждает точность такой оценки: группы train могут отличаться от "
                 "событий бенчмарка и по формулировкам, и по поведению пользователей.", ""]
    lines = ["# Расширенная валидация фиксированного CPU baseline", "",
             "Проверка на платформе: **Recall@50 = 0.666981** (сообщено пользователем). "
             "Ранее на узкой локальной test-выборке получено 0.7552. "
             "Ниже сравниваются условия проверки; параметры моделей и answer.csv не изменяются.", "",
             *findings,
             "## Протокол", "",
             f"- Сохранены исходные group_id и split с seed {SEED}: train < 8, dev = 8, test = 9.",
             f"- Обучающие взаимодействия: {manifest['training_rows']:,}. "
             "История, популярность и соответствия категорий построены только по этой части.",
             f"- Корпус union: **{corpus['union_items']:,}** объявлений = "
             f"{corpus['benchmark_items']:,} benchmark + {corpus['extra_items']:,} дополнительных train item_id. "
             "Все конкуренты участвуют в поиске; одинаковые заголовки не схлопываются.",
             "- Корпус benchmark_plus_positives: весь benchmark плюс недостающие положительные "
             "объявления только текущего запроса. Объявления, добавленные для других запросов, недоступны.",
             "- Маска корпуса применяется до топ-500 каждого источника и до поиска 20 соседних "
             "исторических текстов. Недоступные объявления не занимают места кандидатов.",
             "- Словари, IDF, средние длины BM25 рассчитаны один раз по union; в обоих сценариях "
             "они одинаковы. TF-IDF истории также фиксирован и обучен только на train-взаимодействиях "
             "с объявлениями union. Переобучения словарей под каждый запрос нет.",
             "- Это приближение отличается от полной перестройки индекса по benchmark + ответы "
             "для каждого запроса. Оно позволяет отделить влияние доступных конкурентов от смены IDF. "
             "Поэтому даже прежняя узкая когорта во второй схеме может отличаться от старого baseline.",
             "- Параметры: прежние веса полей, множитель географии 16, RRF k=60, глубина 500, "
             "вес истории 0.5. Новые варианты не подбирались. Test уже просмотрен ранее и является диагностическим.",
             "- Наличие добавляемого объявления известно из разметки только при формировании "
             "диагностического корпуса; эта метка не передаётся в score или обучающие статистики.", "",
             "Оба сценария приближенные: union увеличивает конкуренцию и меняет состав корпуса; "
             "benchmark_plus_positives использует знание правильных ответов и не добавляет всех "
             "их естественных конкурентов. Ни одно число не является точным прогнозом платформы.", "",
             "## Полнота покрытия", "",
             "| Часть | Запросов | Положительных | Ответов вне benchmark | Групп полностью в benchmark | Групп с ответами вне benchmark | Исключено |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    current = frame.filter((pl.col("experiment") == "rrf_history_half") & (pl.col("scheme") == "union"))
    for (stage,), part in current.group_by("stage", maintain_order=True):
        lines.append(f"| {stage} | {part.height} | {part['positives'].sum()} | {part['outside_positives'].sum()} | "
                     f"{part['all_in_benchmark'].sum()} | {(~part['all_in_benchmark']).sum()} | 0 |")
    lines += ["", "Знаменатель метрики содержит все известные положительные объявления группы. "
              "В сценариях сравнивается одно и то же множество запросов. "
              "Macro Recall — среднее долей по запросам, а не по положительным строкам.", "",
              "## Сравнение вариантов", "",
              "C@500 — macro полнота объединения до 500 кандидатов каждого выбранного источника. "
              "Размер объединения может превышать 500. Recall@500 относится к результату RRF после объединения.", "",
              "| Часть | Корпус | Вариант | Recall@50 | Recall@500 | C@500 |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in summary.iter_rows(named=True):
        lines.append(f"| {row['stage']} | {row['scheme']} | {row['experiment']} | "
                     f"{row['recall@50']:.4f} | {row['recall@500']:.4f} | {row['candidate_recall']:.4f} |")
    lines += ["", "## Сохраняется ли польза истории", "",
              "Сравнение rrf_history_half с rrf_stats: отличаются только наличием канала истории. "
              "95% интервалы — парный bootstrap по кластерам нормализованных текстов, 1000 повторов. "
              "При ресемплинге сохраняется вес полного запроса в macro-метрике.", "",
              "| Часть | Корпус | Когорта | Запросов | Δ Recall@50 | 95% интервал | Улучшено / ухудшено |",
              "| --- | --- | --- | ---: | ---: | --- | ---: |"]
    for c in comparisons:
        lines.append(f"| {c['stage']} | {c['scheme']} | {c['cohort']} | {c['queries']} | "
                     f"{c['delta_recall50']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                     f"{c['improved_queries']} / {c['worse_queries']} |")
    lines += ["", "`has_positives_outside` оценивает полный набор ответов запросов с хотя бы одним "
              "ответом вне benchmark. `outside_positives_only` — отдельная диагностика только "
              "таких ответов: доля по запросу считается внутри этого подмножества. Основная метрика "
              "по полным наборам ответов при этом не меняется.", "",
              "## Потери текущего решения", "",
              "Счётчики ниже относятся к положительным парам. Потеря отбора: ответ есть в объединении "
              "трёх выбранных каналов, но отсутствует в финальных 50. Отдельно показана часть "
              "потерь, неизбежная при более чем 50 найденных положительных ответах.", "",
              "| Часть | Корпус | Recall@50 | C@500 | Найдено в 50 | Потери отбора | Из них из-за лимита 50 | Не найдено источниками |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.filter(pl.col("experiment") == "rrf_history_half").iter_rows(named=True):
        lines.append(f"| {row['stage']} | {row['scheme']} | {row['recall@50']:.4f} | {row['candidate_recall']:.4f} | "
                     f"{row['retrieved']} | {row['lost_at_selection']} | {row['unavoidable_candidate_cap_loss']} | {row['missed_by_sources']} |")
    lines += ["", "## Контроль на прежней узкой когорте", "",
              "Позволяет сравнивать один и тот же набор запросов при изменении числа конкурентов. "
              "Старый test: 1820 запросов, 1875 ответов, 314 потерь отбора, Recall@50 0.7552. "
              "Новый общий индекс меняет также текстовые и исторические IDF; сравнение с исходным "
              "числом 314 не является чистым эффектом только числа объявлений.", "",
              "| Часть | Корпус | Запросов | Recall@50 | C@500 | Потери отбора | Пропуски поиска |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in cohorts.filter((pl.col("experiment") == "rrf_history_half") & pl.col("all_in_benchmark")).iter_rows(named=True):
        lines.append(f"| {row['stage']} | {row['scheme']} | {row['queries']} | {row['recall@50']:.4f} | "
                     f"{row['candidate_recall']:.4f} | {row['lost_at_selection']} | {row['missed_by_sources']} |")
    lines += ["", "## Артефакты и запуск", "",
              "```powershell",
              "uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli evaluate-expanded --workers 4",
              "```", "",
              f"Результаты: `{run_dir.as_posix()}`. `metrics.csv` — общие результаты; "
              "`cohorts.csv` — запросы с ответами внутри/вне benchmark; `history_comparisons.json` — "
              "парные сравнения; `positive_slices.csv` — ошибки по доступности объявления и знакомству "
              "с ним в train; `per_query.parquet` и `positive_ranks.parquet` — детальные результаты.", "",
              f"Индекс union: {corpus['build_seconds']:.1f} с, {corpus['index_bytes']/2**20:.1f} MiB. "
              f"Время этой команды оценки: {manifest['evaluation_seconds_this_invocation']:.1f} с; "
              f"восстановлено из завершённых пакетов: {manifest['resumed_groups']} запросов. "
              "После записи двух файлов пакета создаётся отметка завершения; повторный запуск "
              "продолжает расчёт, если сигнатуры данных, кода и параметров совпадают.", ""]
    if manifest["limit"]:
        lines.insert(2, f"**Технический прогон: только {manifest['limit']} групп. Не использовать как итоговую оценку.**\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Expanded report: {destination} ({frame.height} query-variant rows)", flush=True)
