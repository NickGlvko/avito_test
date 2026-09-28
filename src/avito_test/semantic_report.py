"""Dev-only selection and paired diagnostics for frozen E5 ablations."""
# Select on dev and treat repeated test results as diagnostic.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import SEED, sha256_file

OUT = Path("artifacts/semantic")
REPORT = Path("reports/semantic.md")


def compare(frame: pl.DataFrame, scheme: str, candidate: str, baseline: str) -> dict:
    left = frame.filter((pl.col("scheme") == scheme) & (pl.col("variant") == candidate)).select(
        "group_id", "text", pl.col("recall@50").alias("candidate"))
    right = frame.filter((pl.col("scheme") == scheme) & (pl.col("variant") == baseline)).select(
        "group_id", pl.col("recall@50").alias("baseline"))
    paired = left.join(right, on="group_id", validate="1:1").with_columns(
        (pl.col("candidate") - pl.col("baseline")).alias("delta"))
    clusters = paired.group_by("text").agg(pl.col("delta").sum().alias("total"),
                                            pl.len().alias("size")).sort("text")
    totals, sizes = clusters["total"].to_numpy(), clusters["size"].to_numpy()
    rng = np.random.default_rng(SEED)
    draws = np.array([totals[indices].sum() / sizes[indices].sum()
                      for indices in (rng.integers(0, len(totals), len(totals)) for _ in range(1000))])
    return {"delta": float(paired["delta"].mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "improved": int((paired["delta"] > 0).sum()),
            "worsened": int((paired["delta"] < 0).sum())}


def summarize(frame: pl.DataFrame) -> dict:
    result = {}
    for scheme in ("union", "benchmark_plus_positives"):
        result[scheme] = {}
        for variant in frame["variant"].unique().sort().to_list():
            part = frame.filter((pl.col("scheme") == scheme) & (pl.col("variant") == variant))
            if not part.height:
                continue
            result[scheme][variant] = part.select(
                pl.len().alias("queries"), pl.col("recall@50").mean().alias("recall50"),
                pl.col("candidate_recall").mean().alias("pool_recall"),
                pl.col("candidate_count").mean().alias("mean_pool_size"),
                pl.col("lost_at_selection").sum().alias("selection_losses"),
                pl.col("missed_by_sources").sum().alias("retrieval_losses"),
                pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
                pl.col("other_location_recall@50").mean().alias("other_location_recall50"),
                pl.col("new_semantic_positives").sum().alias("new_semantic_positives"),
                pl.col("new_semantic_positives_in_top50").sum().alias("new_semantic_positives_in_top50"),
                pl.col("semantic_global_recall50").mean().alias("semantic_global_recall50"),
                pl.col("semantic_global_recall200").mean().alias("semantic_global_recall200"),
                pl.col("semantic_local_recall50").mean().alias("semantic_local_recall50"),
                pl.col("semantic_local_recall200").mean().alias("semantic_local_recall200"),
                pl.col("semantic_union_recall200").mean().alias("semantic_union_recall200"),
            ).row(0, named=True)
    return result


def analyze(dev_path: Path, test_path: Path | None = None,
            select: bool = False) -> dict:
    dev = pl.read_parquet(dev_path / "per_query.parquet")
    if dev["stage"].unique().to_list() != ["dev"]:
        raise ValueError("Selection must use dev only")
    summary = summarize(dev)
    comparisons = {scheme: {f"{a}-{b}": compare(dev, scheme, a, b)
                 for a, b in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"))}
                 for scheme in summary}
    report = {"dev_path": str(dev_path), "dev_sha256": sha256_file(dev_path / "per_query.parquet"),
              "dev_summary": summary, "dev_comparisons": comparisons}
    if select:
        if dev.filter(pl.col("variant") == "A")["group_id"].n_unique() != 35652:
            raise ValueError("Freeze a decision only after full dev evaluation")
        union, augmented = summary["union"], summary["benchmark_plus_positives"]
        eligible = [name for name in ("B", "C", "D")
                    if union[name]["recall50"] > union["A"]["recall50"]
                    and comparisons["union"][f"{name}-A"]["ci95"][0] > 0
                    and augmented[name]["recall50"] >= augmented["A"]["recall50"]]
        selected = max(eligible, key=lambda name: (union[name]["recall50"], name)) if eligible else "A"
        decision = {"selected": selected,
                    "rule": "positive paired text-cluster CI on union/dev and no mean loss on benchmark_plus_positives/dev",
                    "dev_per_query_sha256": report["dev_sha256"],
                    "dev_union_recall50": union[selected]["recall50"],
                    "dev_augmented_recall50": augmented[selected]["recall50"],
                    "control_union_recall50": union["A"]["recall50"],
                    "control_augmented_recall50": augmented["A"]["recall50"]}
        (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        report["decision"] = decision
    elif (OUT / "decision.json").exists():
        report["decision"] = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if test_path is not None:
        if "decision" not in report:
            raise ValueError("Diagnostic test requires a frozen dev decision")
        test = pl.read_parquet(test_path / "per_query.parquet")
        report["test_path"] = str(test_path)
        report["test_summary"] = summarize(test)
        selected = report["decision"]["selected"]
        if selected != "A":
            report["test_comparison"] = {scheme: compare(test, scheme, selected, "A")
                                         for scheme in report["test_summary"]}
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Проверка смыслового поиска E5", "",
             "Контроль A — отправленный CatBoost; B добавляет сходство к прежнему пулу; C расширяет пул без E5-признаков; D расширяет пул и добавляет E5-признаки. Решение выбирается только на полном dev.", "",
             "Заморожена open-source модель [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), ревизия `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Кодирование: `query:`/`passage:`, естественный очищенный текст, максимум 256 токенов, L2-нормализация, размерность 384. ONNX float32 совпал с исходной моделью на контрольных 192 входах; FAISS IndexFlatIP выполняет точный поиск. История и популярность в OOF остаются рассчитанными только вне проверочной части.", "",
             "CatBoostRanker использует прежние параметры PairLogit. Известные выбранные объявления имеют метку 1, остальные кандидаты — 0 как неразмеченные альтернативы; среди них могут быть подходящие услуги. На dev оценивается полный пул, без сокращения альтернатив.", "",
             "## Recall@50 и полнота пула", "",
             "| Сценарий | Вариант | Recall@50 | C@pool | Средний пул | Потери отбора | Совпадающая локация | Другая локация |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope, variants in summary.items():
        for name, row in variants.items():
            lines.append(f"| {scope} | {name} | {row['recall50']:.4f} | {row['pool_recall']:.4f} | {row['mean_pool_size']:.0f} | {row['selection_losses']} | {row['same_location_recall50']:.4f} | {row['other_location_recall50']:.4f} |")
    lines += ["", "## Самостоятельный смысловой поиск", "",
              "| Сценарий | Глобальный @50 | Глобальный @200 | Локальный @50 | Локальный @200 | Объединение @200 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for scope, variants in summary.items():
        row = variants["A"]
        lines.append(f"| {scope} | {row['semantic_global_recall50']:.4f} | {row['semantic_global_recall200']:.4f} | {row['semantic_local_recall50']:.4f} | {row['semantic_local_recall200']:.4f} | {row['semantic_union_recall200']:.4f} |")
    lines += ["", "## Парные различия на dev", "",
              "Bootstrap по кластерам нормализованного текста: 1000 повторов, seed 20260926.", "",
              "| Сценарий | Сравнение | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
              "| --- | --- | ---: | --- | ---: |"]
    for scope, variants in comparisons.items():
        for name, row in variants.items():
            lo, hi = row["ci95"]
            lines.append(f"| {scope} | {name} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['improved']} / {row['worsened']} |")
    if "D" in summary["union"]:
        row = summary["union"]["D"]
        lines += ["", f"На union/dev семантические каналы добавили к прежнему пулу {row['new_semantic_positives']} положительных пар «запрос — объявление»; D включил в топ-50 {row['new_semantic_positives_in_top50']} из них."]
    training_audit = OUT / "models/expanded_oof.json"
    if training_audit.exists():
        audit = json.loads(training_audit.read_text(encoding="utf-8"))
        lines += ["", "## Обучение", "",
                  f"Расширенный OOF использовал {audit['queries']} исходных запросов; в пуле найдено {audit['expanded_found']} из {audit['known_positives']} известных положительных пар против {audit['base_found']} у прежнего пула. Для парного обучения пригодны {audit['groups_with_positive_in_pool']} групп; остальные остаются в знаменателе оценки."]
    if "decision" in report:
        lines += ["", f"Зафиксированный на dev вариант: **{report['decision']['selected']}**. Правило: {report['decision']['rule']}."]
    if "test_summary" in report:
        lines += ["", "## Диагностический test", "",
                  "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | Совпадающая локация | Другая локация |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for scope, variants in report["test_summary"].items():
            for name, row in variants.items():
                lines.append(f"| {scope} | {name} | {row['recall50']:.4f} | {row['pool_recall']:.4f} | {row['selection_losses']} | {row['same_location_recall50']:.4f} | {row['other_location_recall50']:.4f} |")
        if "test_comparison" in report:
            lines += ["", "Парное сравнение выбранного на dev варианта с A на test:", "",
                      "| Сценарий | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
                      "| --- | ---: | --- | ---: |"]
            for scope, comparison in report["test_comparison"].items():
                lo, hi = comparison["ci95"]
                lines.append(f"| {scope} | {comparison['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {comparison['improved']} / {comparison['worsened']} |")
    encoder_state = OUT / "embeddings/encode_state.json"
    if encoder_state.exists():
        state = json.loads(encoder_state.read_text(encoding="utf-8"))
        if state.get("complete"):
            sequential = state["elapsed_seconds"]
            segments = state.get("parallel_segments", [])
            segment_seconds = sum(segment["elapsed_seconds"] for segment in segments)
            lines += ["", "## Вычислительные затраты", "",
                      f"Кодирование корпуса: {state['unique_texts']} уникальных текстов; {sequential:.1f} с до разделения на диапазоны и суммарно {segment_seconds:.1f} с обработки диапазонов. Часть диапазонов короткое время обрабатывалась одновременно, поэтому сумма не является временем на часах. Подготовка текстов и загрузка модели сюда не включены. Векторы сохранены с SHA256 `{state['vector_sha256']}`."]
    timing = dev_path / "timing.json"
    if timing.exists():
        used = json.loads(timing.read_text(encoding="utf-8"))
        lines += [f"Оценка dev: {used['seconds']:.1f} с, {used['workers']} процесса, пик рабочего набора процесса {used['worker_peak_rss_bytes'] / 2**30:.2f} GiB; суммарное время точного векторного поиска по процессам {used['dense_search_seconds_sum']:.1f} с."]
    if test_path is not None and (test_path / "timing.json").exists():
        used = json.loads((test_path / "timing.json").read_text(encoding="utf-8"))
        lines += [f"Диагностический test: {used['seconds']:.1f} с; пик рабочего набора процесса {used['worker_peak_rss_bytes'] / 2**30:.2f} GiB."]
    package = Path("models/semantic_v1/config.json")
    if package.exists():
        prediction = json.loads(package.read_text(encoding="utf-8"))
        lines += [f"Предсказание benchmark: {prediction['seconds']:.1f} с; пик {prediction['peak_rss_bytes'] / 2**30:.2f} GiB. SHA256 `answer_semantic.csv`: `{prediction['answer_sha256']}`.",
                  f"В `models/semantic_v1/` сохранены веса B (SHA256 `{prediction['model_sha256']}`), конфигурация обучения и соответствие {prediction['item_vector_rows']} item_id строкам векторов (SHA256 `{prediction['item_id_mapping_sha256']}`). Ревизия E5: `{prediction['e5_revision']}`. Локальную проверку комплекта выполняет `python -m avito_test.semantic_verify`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "decision": report.get("decision")}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dev", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    analyze(args.dev, args.test, args.select)
