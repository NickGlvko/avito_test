"""Reproducible geography slices, paired comparisons, and dev-only decision."""
# Compare geography rules by overall recall before inspecting location slices.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import SEED, sha256_file, write_json

BASE = "catboost_submitted"


def _summary(frame: pl.DataFrame, variant: str) -> dict:
    def ratio(kind: str, field: str) -> pl.Expr:
        return pl.when(pl.col(f"{kind}_positives") > 0).then(
            pl.col(f"{kind}_{field}") / pl.col(f"{kind}_positives")).otherwise(None).mean()

    row = frame.select(
        pl.col("recall@50").mean().alias("recall50"),
        pl.col("candidate_recall").mean().alias("pool_recall"),
        ratio("same", "found").alias("same_recall50"),
        ratio("other", "found").alias("other_recall50"),
        ratio("same", "candidate_found").alias("same_pool_recall"),
        ratio("other", "candidate_found").alias("other_pool_recall"),
        pl.col("candidate_found").sum().alias("candidate_found"),
        pl.col("found").sum().alias("found"),
        pl.col("other_positives").sum().alias("other_positives"),
        pl.col("other_candidate_found").sum().alias("other_candidate_found"),
        pl.col("other_found").sum().alias("other_found"),
        pl.col("selected_other").mean().alias("selected_other_mean"),
        pl.col("selected_other").median().alias("selected_other_median"),
        pl.col("candidate_count").mean().alias("pool_size_mean"),
    ).row(0, named=True)
    row["variant"] = variant
    row["queries"] = frame.height
    row["selection_losses"] = row["candidate_found"] - row["found"]
    row["other_selection_losses"] = row["other_candidate_found"] - row["other_found"]
    return row


def _comparison(candidate: pl.DataFrame, baseline: pl.DataFrame) -> dict:
    paired = candidate.select("group_id", "text", "recall@50").join(
        baseline.select("group_id", pl.col("recall@50").alias("baseline")),
        on="group_id", validate="1:1").with_columns(
            (pl.col("recall@50") - pl.col("baseline")).alias("delta"))
    clusters = paired.group_by("text").agg(
        pl.col("delta").sum().alias("total"), pl.len().alias("size")).sort("text")
    totals = clusters["total"].to_numpy()
    sizes = clusters["size"].to_numpy()
    rng = np.random.default_rng(SEED)
    draws = np.array([totals[ix].sum() / sizes[ix].sum()
                      for ix in (rng.integers(0, len(totals), len(totals)) for _ in range(1000))])
    return {"delta": float(paired["delta"].mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "improved_queries": int((paired["delta"] > 0).sum()),
            "worsened_queries": int((paired["delta"] < 0).sum())}


def analyze(diagnostics: Path, ranker_run: Path, model_evaluation: Path | None,
            output: Path, stage: str = "dev") -> dict:
    quota_frame = pl.read_parquet(diagnostics / "metrics.parquet")
    positive_frame = pl.read_parquet(diagnostics / "positives.parquet")
    probes = pl.read_parquet(diagnostics / "probes.parquet")
    old_full = pl.read_parquet(ranker_run / stage / "per_query.parquet")
    expected_queries = old_full["group_id"].n_unique()
    old = old_full.join(
        quota_frame.select("group_id").unique(), on="group_id", how="semi")
    submitted = old.filter(pl.col("variant") == "catboost_full_no_history_features")
    agreement = quota_frame.filter(pl.col("quota") == 0).join(
        submitted.select("group_id", "scheme",
                         pl.col("recall@50").alias("old_recall"),
                         pl.col("candidate_recall").alias("old_pool")),
        on=["group_id", "scheme"], validate="1:1")
    if agreement.height != quota_frame.filter(pl.col("quota") == 0).height:
        raise ValueError("Submitted control lacks query/scope rows")
    if (agreement.select((pl.col("recall@50") - pl.col("old_recall")).abs().max()).item() > 1e-8 or
            agreement.select((pl.col("candidate_recall") - pl.col("old_pool")).abs().max()).item() > 1e-8):
        raise ValueError("Quota 0 does not reproduce the submitted model")
    expanded = pl.read_parquet(model_evaluation / "metrics.parquet") if model_evaluation else None
    result = {"stage": stage, "queries": quota_frame.filter(pl.col("quota") == 0).height // 2,
              "expected_queries": expected_queries,
              "controls_match": True, "diagnostics": str(diagnostics),
              "ranker_run": str(ranker_run), "model_evaluation": str(model_evaluation) if model_evaluation else None,
              "diagnostic_timing": json.loads((diagnostics / "run.json").read_text(encoding="utf-8")),
              "expanded_timing": (json.loads((model_evaluation / "run.json").read_text(encoding="utf-8"))
                                  if model_evaluation else None),
              "schemes": {}}
    if result["expanded_timing"]:
        model_run = Path(result["expanded_timing"]["model_run"])
        result["training_audit"] = json.loads((model_run / "training.json").read_text(encoding="utf-8"))
    for scheme in ("union", "benchmark_plus_positives"):
        current = quota_frame.filter(pl.col("scheme") == scheme)
        baseline = current.filter(pl.col("quota") == 0)
        variants = {}
        for quota in (0, 2, 5, 10):
            name = BASE if quota == 0 else f"reserve_{quota}"
            part = current.filter(pl.col("quota") == quota)
            variants[name] = {**_summary(part, name),
                              "vs_submitted": _comparison(part, baseline) if quota else None}
        if expanded is not None:
            part = expanded.filter(pl.col("scheme") == scheme)
            variants["global_retrained"] = {
                **_summary(part, "global_retrained"),
                "vs_submitted": _comparison(part, baseline)}
        for control in ("rrf_geo16", "rrf_history_half"):
            part = old.filter((pl.col("scheme") == scheme) & (pl.col("variant") == control))
            variants[control] = {"variant": control, "queries": part.height,
                                 "recall50": float(part["recall@50"].mean()),
                                 "pool_recall": float(part["candidate_recall"].mean()),
                                 "other_recall50": float(part["other_location_recall@50"].mean()),
                                 "same_recall50": float(part["same_location_recall@50"].mean())}
        ranks = {}
        for kind in ("same", "other", "unknown"):
            part = positive_frame.filter((pl.col("scheme") == scheme) &
                                         (pl.col("location_kind") == kind))
            values = part["catboost_rank"].to_numpy()
            ranks[kind] = {"positives": len(values), "missing_pool": int((values == 0).sum()),
                           "top50": int(((values >= 1) & (values <= 50)).sum()),
                           "rank51_100": int(((values >= 51) & (values <= 100)).sum()),
                           "rank101_200": int(((values >= 101) & (values <= 200)).sum()),
                           "rank201_plus": int((values >= 201).sum())}
        probe = probes.filter(pl.col("scheme") == scheme)
        p = probe.select(
            pl.col("same_positives").sum().alias("same_positives"),
            pl.col("other_positives").sum().alias("other_positives"),
            *[(pl.col(f"{kind}_extended_found") - pl.col(f"{kind}_pool_found")).sum().alias(
                f"{kind}_new_positives") for kind in ("same", "other", "unknown")],
            pl.col("pool_count").mean().alias("base_pool_size"),
            pl.col("extended_count").mean().alias("extended_pool_size"),
        ).row(0, named=True)
        result["schemes"][scheme] = {"variants": variants, "ranks": ranks,
                                      "global_probe": p}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def freeze_dev_decision(summary: dict, output: Path) -> dict:
    if summary["stage"] != "dev":
        raise ValueError("Selection can only use dev")
    if summary["queries"] != summary["expected_queries"]:
        raise ValueError("Cannot freeze a decision from a partial dev sample")
    union = summary["schemes"]["union"]["variants"]
    augmented = summary["schemes"]["benchmark_plus_positives"]["variants"]
    base_u, base_a = union[BASE]["recall50"], augmented[BASE]["recall50"]
    candidates = [name for name in ("reserve_2", "reserve_5", "reserve_10", "global_retrained")
                  if name in union and union[name]["recall50"] > base_u and
                  augmented[name]["recall50"] >= base_a and
                  union[name]["vs_submitted"]["ci95"][0] > 0]
    selected = max(candidates, key=lambda name: (union[name]["recall50"], name)) if candidates else BASE
    decision = {"selected": selected, "criterion": "union/dev gain with paired text-cluster bootstrap lower bound > 0 and no mean loss on benchmark_plus_positives/dev",
                "dev_union_recall50": union[selected]["recall50"],
                "dev_augmented_recall50": augmented[selected]["recall50"],
                "submitted_union_recall50": base_u,
                "submitted_augmented_recall50": base_a,
                "dev_summary_sha256": sha256_file(output.with_name("dev_summary.json"))}
    write_json(output, decision)
    return decision


def markdown(summary: dict, decision: dict | None, destination: Path) -> None:
    introduction = ("Контроль — отправленный CatBoost. Оценка на union/dev служит основным ориентиром; "
                    "benchmark + ответы проверяет перенос. test используется только после фиксации решения на dev."
                    if summary["stage"] == "dev" else
                    "Диагностический test рассчитан после фиксации решения на dev и не используется для выбора модели.")
    lines = ["# Ограниченный эксперимент с географией", "",
             introduction, "",
             "Корпус и разбиение сохранены из предыдущей проверки: 515 895 объявлений в union, "
             "история для dev/test построена только по split < 8. Квоты применяются к прежнему пулу "
             "после CatBoost. Отсутствующая локация помечается отдельно и не считается другой. "
             "Глобальные каналы дают до 100 кандидатов BM25 и до 100 символьного поиска; "
             "повторы item_id удаляются. Расширенная модель обучена на трёх out-of-fold частях "
             "исходного train, на всех найденных положительных и до 150 неразмеченных конкурентов "
             "на запрос. Параметры PairLogit и 240 деревьев не менялись. "
             "Признаков принадлежности benchmark и добавления ответа в корпус у модели нет; "
             "доступность применяется до топ-100 каждого глобального канала. "
             "В текущих данных пропусков локации запроса и объявления нет.", "",
             "## Основное сравнение", "",
             "| Вариант | Union Recall@50 | Benchmark + ответы Recall@50 | Recall другой локации, union | C@pool, union |",
             "| --- | ---: | ---: | ---: | ---: |"]
    union_variants = summary["schemes"]["union"]["variants"]
    augmented_variants = summary["schemes"]["benchmark_plus_positives"]["variants"]
    for name, row in union_variants.items():
        lines.append(f"| {name} | {row['recall50']:.4f} | "
                     f"{augmented_variants[name]['recall50']:.4f} | "
                     f"{row['other_recall50']:.4f} | {row['pool_recall']:.4f} |")
    lines += ["", "## Recall@50 и полнота пула", "",
             "| Сценарий | Вариант | Recall@50 | Recall другой локации | C@pool | Потери отбора |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for scheme, content in summary["schemes"].items():
        for name, row in content["variants"].items():
            losses = row.get("selection_losses")
            lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | "
                         f"{row['other_recall50']:.4f} | {row['pool_recall']:.4f} | "
                         f"{losses if losses is not None else '—'} |")
    lines += ["", "## Срез по локации исходной модели", "",
              "Macro-полнота считается только по запросам с ответами данного типа. "
              "Доля потерь отбора ниже относится к ответам, которые уже были в пуле.", "",
              "| Сценарий | Тип | C@pool | Recall@50 | Ответов в пуле | Потеряно отбором | Доля потерь | Других локаций в топ-50, среднее / медиана |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scheme, content in summary["schemes"].items():
        row = content["variants"][BASE]
        for kind in ("same", "other"):
            found = row[f"{kind}_candidate_found"] if kind == "other" else None
            if kind == "same":
                ranks = content["ranks"]["same"]
                found = ranks["positives"] - ranks["missing_pool"]
                selected = ranks["top50"]
            else:
                selected = row["other_found"]
            lost = found - selected
            selected_other = (f"{row['selected_other_mean']:.1f} / {row['selected_other_median']:.0f}"
                              if kind == "other" else "—")
            loss_share = f"{lost/found:.1%}" if found else "—"
            lines.append(f"| {scheme} | {kind} | {row[f'{kind}_pool_recall']:.4f} | "
                         f"{row[f'{kind}_recall50']:.4f} | {found} | {lost} | "
                         f"{loss_share} | {selected_other} |")
    lines += ["", "## Где теряются ответы", "",
              "Позиции относятся к исходной модели CatBoost. Ноль означает отсутствие в пуле. "
              "Потери отбора — найденные источниками ответы за пределами топ-50.", "",
              "| Сценарий | Локация | Ответов | Вне пула | В 50 | 51–100 | 101–200 | Дальше |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scheme, content in summary["schemes"].items():
        for kind, row in content["ranks"].items():
            lines.append(f"| {scheme} | {kind} | {row['positives']} | {row['missing_pool']} | "
                         f"{row['top50']} | {row['rank51_100']} | {row['rank101_200']} | {row['rank201_plus']} |")
    lines += ["", "## Два глобальных текстовых канала", "",
              "Добавлены топ-100 BM25 и символьного поиска без множителя географии. "
              "Обученная модель заново видит расширенный пул в OOF-примерах; прежние веса CatBoost не переиспользуются.", "",
              "| Сценарий | Новых ответов той же локации | Другой локации | Средний размер пула → расширенного |",
              "| --- | ---: | ---: | ---: |"]
    for scheme, content in summary["schemes"].items():
        probe = content["global_probe"]
        lines.append(f"| {scheme} | {probe['same_new_positives']} | {probe['other_new_positives']} | "
                     f"{probe['base_pool_size']:.0f} → {probe['extended_pool_size']:.0f} |")
    if summary.get("training_audit"):
        training = summary["training_audit"]
        lines += ["", f"В OOF-обучении расширенный пул добавил "
                  f"{training['expanded_found']-training['base_found']} известных положительных пар "
                  f"из {training['known_positives']}; все {training['global_new_other']} — другой локации. "
                  f"Обучено на {training['sampled_rows']} кандидатах из {training['used_queries']} запросов "
                  f"за {training['elapsed_seconds']:.1f} с."]
        base = summary["schemes"]["union"]["variants"][BASE]
        expanded = summary["schemes"]["union"]["variants"]["global_retrained"]
        if expanded["pool_recall"] > base["pool_recall"] and expanded["recall50"] < base["recall50"]:
            lines.append("Пул стал полнее, но переобученный отбор хуже использовал его. "
                         "Малое число новых положительных примеров в OOF на фоне новых "
                         "неразмеченных конкурентов — возможная причина, не доказанная этим экспериментом.")
    lines += ["", "## Парное сравнение с отправленным CatBoost", "",
              "Интервалы — 1000 bootstrap-повторов по кластерам нормализованного текста. "
              "Они диагностические и не учитывают множественный выбор вариантов.", "",
              "| Сценарий | Вариант | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
              "| --- | --- | ---: | --- | ---: |"]
    for scheme, content in summary["schemes"].items():
        for name, row in content["variants"].items():
            comparison = row.get("vs_submitted")
            if comparison is None:
                continue
            lines.append(f"| {scheme} | {name} | {comparison['delta']:+.4f} | "
                         f"[{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}] | "
                         f"{comparison['improved_queries']} / {comparison['worsened_queries']} |")
    if decision:
        lines += ["", "## Зафиксированное решение dev", "",
                  f"Выбран `{decision['selected']}`. Правило: {decision['criterion']}. "
                  f"Union/dev {decision['dev_union_recall50']:.4f}, "
                  f"benchmark + ответы/dev {decision['dev_augmented_recall50']:.4f}."]
        if decision["selected"] == BASE:
            lines.append("Ни один проверенный географический вариант не дал устойчивого общего выигрыша на dev. "
                         "Новый файл ответа не создаётся; остаётся `answer_ranker.csv`.")
        if summary["stage"] == "dev":
            test_path = Path(summary["diagnostics"]).parents[2] / "test_summary.json"
            if test_path.exists():
                test = json.loads(test_path.read_text(encoding="utf-8"))
                lines += ["", "## Диагностический test после фиксации решения", "",
                          "Этот test уже просматривался в предыдущем этапе и не используется для подбора. "
                          "Расширенная модель не запускалась на test, поскольку проиграла на обоих dev-сценариях.", "",
                          "| Вариант | Union Recall@50 | Benchmark + ответы Recall@50 | Recall другой локации, union |",
                          "| --- | ---: | ---: | ---: |"]
                for name, row in test["schemes"]["union"]["variants"].items():
                    lines.append(f"| {name} | {row['recall50']:.4f} | "
                                 f"{test['schemes']['benchmark_plus_positives']['variants'][name]['recall50']:.4f} | "
                                 f"{row['other_recall50']:.4f} |")
    diagnostics = summary["diagnostic_timing"]
    lines += ["", "## Вычислительные затраты", "",
              f"Диагностика: {diagnostics['elapsed_seconds']:.1f} с, "
              f"амортизировано около {1000*diagnostics['elapsed_seconds']/(2*summary['queries']):.1f} мс на запрос/сценарий, "
              f"пик одного процесса {diagnostics['peak_worker_rss_bytes']/2**30:.2f} GiB, "
              f"{diagnostics['workers']} процесса."]
    if summary["expanded_timing"]:
        expanded = summary["expanded_timing"]
        lines.append(f"Оценка расширенной модели: {expanded['elapsed_seconds']:.1f} с, "
                     f"амортизировано около {1000*expanded['elapsed_seconds']/(2*summary['queries']):.1f} мс на запрос/сценарий, "
                     f"пик одного процесса {expanded['peak_worker_rss_bytes']/2**30:.2f} GiB, "
                     f"{expanded['workers']} процесса.")
    report_command = ("uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli geo-report "
                      f"--stage {summary['stage']} --geo-diagnostics {summary['diagnostics']}")
    if summary["model_evaluation"]:
        report_command += f" --geo-evaluation {summary['model_evaluation']}"
    commands = [f"uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli geo-diagnose --stage {summary['stage']} --workers 4"]
    if summary["stage"] == "dev":
        commands.extend([
            "uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli geo-train --train-queries 1000",
            "uv --cache-dir .uv-cache run --offline --frozen python -X utf8 -m avito_test.cli geo-evaluate --stage dev --workers 4",
        ])
    lines += ["", "## Воспроизведение", "", "```powershell", *commands,
              report_command, "```", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
