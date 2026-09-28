"""Dev-only selection and diagnostics for the 15,000-query B/D experiment."""
# Show candidate coverage alongside recall to separate search and selection.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .data import sha256_file
from .semantic_report import compare
from .semantic_scale_select import OUT


REPORT = Path("reports/semantic_scale_15000.md")
PAIRS = (("B15000", "B3000"), ("D15000", "D3000"),
         ("D15000", "B15000"), ("D15000", "B3000"))


def summarize(frame: pl.DataFrame) -> dict:
    result = {}
    for scope in ("union", "benchmark_plus_positives"):
        result[scope] = {}
        for variant in frame["variant"].unique().sort().to_list():
            part = frame.filter((pl.col("scheme") == scope) & (pl.col("variant") == variant))
            if not part.height:
                continue
            result[scope][variant] = part.select(
                pl.len().alias("queries"),
                pl.col("recall@50").mean().alias("recall50"),
                pl.col("candidate_recall").mean().alias("pool_recall"),
                pl.col("lost_at_selection").sum().alias("selection_losses"),
                pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
                pl.col("other_location_recall@50").mean().alias("other_location_recall50"),
                pl.col("semantic_only_positives").sum().alias("semantic_only_positives"),
                pl.col("semantic_only_in_top50").sum().alias("semantic_only_in_top50"),
                pl.col("old_positive_lost_vs_B15000").sum().alias("old_positive_lost_vs_B15000"),
                pl.col("old_positive_gained_vs_B15000").sum().alias("old_positive_gained_vs_B15000")
            ).row(0, named=True)
    return result


def analyze(dev_path: Path, test_path: Path | None = None, select: bool = False) -> dict:
    dev_file = dev_path / "per_query.parquet"
    dev = pl.read_parquet(dev_file)
    if dev["stage"].unique().to_list() != ["dev"]:
        raise ValueError("Model selection must use dev only")
    dev_summary = summarize(dev)
    comparisons = {scope: {f"{a}-{b}": compare(dev, scope, a, b) for a, b in PAIRS}
                   for scope in dev_summary}
    result = {"dev_path": str(dev_path), "dev_sha256": sha256_file(dev_file),
              "dev_summary": dev_summary, "dev_comparisons": comparisons}
    result["dev_timing"] = json.loads((dev_path / "timing.json").read_text(encoding="utf-8"))
    if select:
        if dev.filter(pl.col("variant") == "B3000")["group_id"].n_unique() != 35652:
            raise ValueError("Decision requires full dev, not a slice")
        union = dev_summary["union"]
        augmented = dev_summary["benchmark_plus_positives"]
        eligible_b = (comparisons["union"]["B15000-B3000"]["ci95"][0] > 0 and
                      augmented["B15000"]["recall50"] >= augmented["B3000"]["recall50"])
        eligible_d = (comparisons["union"]["D15000-B3000"]["ci95"][0] > 0 and
                      augmented["D15000"]["recall50"] >= augmented["B3000"]["recall50"] and
                      comparisons["union"]["D15000-B15000"]["ci95"][0] > 0 and
                      augmented["D15000"]["recall50"] >= augmented["B15000"]["recall50"])
        selected = "D15000" if eligible_d else ("B15000" if eligible_b else "B3000")
        decision = {"selected": selected, "eligible_B15000": eligible_b,
                    "eligible_D15000": eligible_d,
                    "rule": "positive text-cluster paired CI versus submitted B on union/dev; no augmented/dev mean loss; D additionally needs positive CI versus B15000 and no augmented loss versus B15000",
                    "dev_per_query_sha256": result["dev_sha256"],
                    "dev_union_recall50": union[selected]["recall50"],
                    "dev_augmented_recall50": augmented[selected]["recall50"],
                    "submitted_B_union_recall50": union["B3000"]["recall50"],
                    "submitted_B_augmented_recall50": augmented["B3000"]["recall50"]}
        (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        result["decision"] = decision
    elif (OUT / "decision.json").exists():
        result["decision"] = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
        if result["decision"]["dev_per_query_sha256"] != result["dev_sha256"]:
            raise ValueError("Frozen dev decision does not match these dev predictions")
    if test_path is not None:
        if "decision" not in result:
            raise ValueError("Test requires a frozen dev decision")
        test = pl.read_parquet(test_path / "per_query.parquet")
        result["test_path"] = str(test_path)
        result["test_summary"] = summarize(test)
        result["test_timing"] = json.loads((test_path / "timing.json").read_text(encoding="utf-8"))
        selected = result["decision"]["selected"]
        if selected != "B3000":
            result["test_comparison"] = {scope: compare(test, scope, selected, "B3000")
                                         for scope in result["test_summary"]}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    training = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    audit = training["audit"]
    lines = ["# Увеличение OOF-обучения CatBoost до 15 000 запросов", "",
             "Меняется только количество train-групп. E5, тексты, источники, глубина кандидатов, 150 альтернатив на запрос и параметры PairLogit сохранены. Отправленный B на 3000 запросах имел платформенный Recall@50 **0.732865** (сообщено пользователем); новая оценка платформы пока неизвестна.", "",
             f"Включены прежние 3000 групп и 12 000 новых случайных групп; seed `{selection['selection_seed']}`, SHA256 списка `{selection['selected_sha256']}`. Распределение по OOF-частям: {selection['by_fold']}. Внутренние статистики каждой части рассчитаны только на двух других частях исходного train; dev/test не обучают модель.", "",
             "## Обучение", "",
             f"B: {audit['b_used_groups']} групп из 15 000, {audit['b_rows']} строк, {audit['b_positives']} известных положительных и {audit['b_rows']-audit['b_positives']} неразмеченных альтернатив; полнота OOF-пула {audit['b_pool_macro_recall']:.4f}.",
             f"D: {audit['d_used_groups']} групп из 15 000, {audit['d_rows']} строк, {audit['d_positives']} известных положительных и {audit['d_rows']-audit['d_positives']} неразмеченных альтернатив; полнота OOF-пула {audit['d_pool_macro_recall']:.4f}. Смысловые каналы добавили {audit['semantic_only_found']} известных положительных пар, отсутствовавших в прежнем пуле.",
             f"Подготовка новых OOF-признаков: {audit['elapsed_seconds']:.1f} с, пик процесса {audit['peak_rss_bytes']/2**30:.2f} GiB. Обучение B/D: {training['B']['train_seconds']:.1f}/{training['D']['train_seconds']:.1f} с; пик процесса при обучении {max(training['B']['process_peak_rss_bytes'], training['D']['process_peak_rss_bytes'])/2**30:.2f} GiB. Параметры одинаковы с 3000-групповым контролем.", "",
             "## Полный dev", "",
             "| Корпус | Модель | Recall@50 | C@pool | Потери отбора | Совпадающая локация | Другая локация | Только E5 в 50 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope, variants in dev_summary.items():
        for variant, row in variants.items():
            lines.append(f"| {scope} | {variant} | {row['recall50']:.4f} | {row['pool_recall']:.4f} | {row['selection_losses']} | {row['same_location_recall50']:.4f} | {row['other_location_recall50']:.4f} | {row['semantic_only_in_top50']} |")
    dev_timing = result["dev_timing"]
    lines += ["", f"Полная оценка dev: {dev_timing['seconds']:.1f} с, пик рабочего набора одного процесса {dev_timing['worker_peak_rss_bytes']/2**30:.2f} GiB при двух рабочих процессах."]
    lines += ["", "## Парные сравнения dev", "",
              "Bootstrap по кластерам нормализованного текста, 1000 повторов, seed 20260926. Интервалы диагностические, без поправки на выбор модели.", "",
              "| Корпус | Сравнение | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
              "| --- | --- | ---: | --- | ---: |"]
    for scope, entries in comparisons.items():
        for pair, row in entries.items():
            lo, hi = row["ci95"]
            lines.append(f"| {scope} | {pair} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['improved']} / {row['worsened']} |")
    d = dev_summary["union"]["D15000"]
    lines += ["", f"На union/dev D15000 добавил в финальные 50 {d['semantic_only_in_top50']} положительных объявлений, найденных только E5. По сравнению с B15000 он потерял {d['old_positive_lost_vs_B15000']} и вернул {d['old_positive_gained_vs_B15000']} положительных объявлений прежнего пула."]
    if "decision" in result:
        lines += ["", f"Зафиксированный по dev выбор: **{result['decision']['selected']}**. Правило: {result['decision']['rule']}."]
    if "test_summary" in result:
        test_timing = result["test_timing"]
        lines += ["", f"Оценка test: {test_timing['seconds']:.1f} с, пик рабочего набора одного процесса {test_timing['worker_peak_rss_bytes']/2**30:.2f} GiB при двух рабочих процессах."]
        lines += ["", "## Диагностический test", "",
                  "| Корпус | Модель | Recall@50 | C@pool | Потери отбора | Совпадающая локация | Другая локация |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for scope, variants in result["test_summary"].items():
            for variant, row in variants.items():
                lines.append(f"| {scope} | {variant} | {row['recall50']:.4f} | {row['pool_recall']:.4f} | {row['selection_losses']} | {row['same_location_recall50']:.4f} | {row['other_location_recall50']:.4f} |")
        if "test_comparison" in result:
            lines += ["", "| Корпус | Δ выбранной модели к отправленному B | 95% интервал | Улучшено / ухудшено |",
                      "| --- | ---: | --- | ---: |"]
            for scope, row in result["test_comparison"].items():
                lo, hi = row["ci95"]
                lines.append(f"| {scope} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['improved']} / {row['worsened']} |")
    package = Path("models/semantic_15000_v1/config.json")
    if package.exists():
        config = json.loads(package.read_text(encoding="utf-8"))
        lines += ["", f"Новый ответ `{config['answer']}`: SHA256 `{config['answer_sha256']}`, {config['validation']['queries']} запросов по {config['validation']['min_candidates']}–{config['validation']['max_candidates']} кандидатов. Два офлайн-запуска совпали побайтно; веса и отображение item_id находятся в `models/semantic_15000_v1/`. Предсказание заняло {config['seconds']:.1f} с; пик рабочего набора {config['peak_rss_bytes']/2**30:.2f} GiB."]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "selected": result.get("decision", {}).get("selected")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dev", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    analyze(args.dev, args.test, args.select)
