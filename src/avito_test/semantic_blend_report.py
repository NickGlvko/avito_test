"""Select a fixed score blend on dev and report the diagnostic test."""
# Diagnostic test is reported after the blend has been chosen on dev.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .data import sha256_file
from .semantic_blend import ALPHAS, CONTROL, variant
from .semantic_blend_control import OUT, freeze
from .semantic_report import compare


REPORT = Path("reports/semantic_blend.md")


def summaries(frame: pl.DataFrame) -> dict:
    result = {}
    for scheme in ("union", "benchmark_plus_positives"):
        result[scheme] = {}
        for name in frame["variant"].unique().sort().to_list():
            part = frame.filter((pl.col("scheme") == scheme) &
                                (pl.col("variant") == name))
            if part.height:
                result[scheme][name] = part.select(
                    pl.len().alias("queries"),
                    pl.col("recall@50").mean().alias("recall50"),
                    pl.col("candidate_recall").mean().alias("pool_recall"),
                    pl.col("lost_at_selection").sum().alias("selection_losses"),
                    pl.col("same_location_recall@50").mean().alias("same_location_recall50"),
                    pl.col("other_location_recall@50").mean().alias("other_location_recall50")
                ).row(0, named=True)
    return result


def analyze(dev_path: Path, test_path: Path | None = None, select: bool = False) -> dict:
    control = freeze()
    dev_file = dev_path / "per_query.parquet"
    dev = pl.read_parquet(dev_file)
    if dev["stage"].unique().to_list() != ["dev"]:
        raise ValueError("Blends may only be selected on dev")
    dev_summary = summaries(dev)
    pairs = {scheme: {name: compare(dev, scheme, name, CONTROL)
                      for name in (variant(alpha) for alpha in ALPHAS)}
             for scheme in dev_summary}
    result = {"dev_path": str(dev_path), "dev_sha256": sha256_file(dev_file),
              "dev_summary": dev_summary, "dev_comparisons": pairs,
              "control": control,
              "dev_timing": json.loads((dev_path / "timing.json").read_text(encoding="utf-8"))}
    if select:
        if dev.filter(pl.col("variant") == CONTROL)["group_id"].n_unique() != 35652:
            raise ValueError("Selecting from a dev slice is forbidden")
        eligible = [variant(alpha) for alpha in ALPHAS if
                    pairs["union"][variant(alpha)]["ci95"][0] > 0 and
                    dev_summary["benchmark_plus_positives"][variant(alpha)]["recall50"] >=
                    dev_summary["benchmark_plus_positives"][CONTROL]["recall50"]]
        selected = max(eligible, key=lambda name: (
            dev_summary["union"][name]["recall50"],
            dev_summary["benchmark_plus_positives"][name]["recall50"])) if eligible else CONTROL
        decision = {"selected": selected,
                    "alpha": float(selected.removeprefix("blend_")) if selected != CONTROL else None,
                    "eligible_blends": eligible,
                    "rule": "positive lower bound of paired text-cluster bootstrap on union/dev versus B15000, no mean loss on benchmark_plus_positives/dev; among eligible choose highest union/dev Recall@50",
                    "dev_per_query": str(dev_file), "dev_sha256": result["dev_sha256"],
                    "control_answer_sha256": control["answer_sha256"],
                    "B3000_model_sha256": control["B3000_model_sha256"],
                    "B15000_model_sha256": control["B15000_model_sha256"]}
        (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        result["decision"] = decision
    elif (OUT / "decision.json").exists():
        decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
        if decision["dev_sha256"] != result["dev_sha256"]:
            raise ValueError("Frozen dev decision and analysis differ")
        result["decision"] = decision
    if test_path:
        if "decision" not in result or result["decision"]["selected"] == CONTROL:
            raise ValueError("Test is only diagnostic for a dev-selected blend")
        test = pl.read_parquet(test_path / "per_query.parquet")
        if test["stage"].unique().to_list() != ["test"]:
            raise ValueError("Diagnostic test stage changed")
        result["test_path"] = str(test_path)
        result["test_summary"] = summaries(test)
        result["test_comparisons"] = {scheme: compare(test, scheme,
            result["decision"]["selected"], CONTROL) for scheme in result["test_summary"]}
        result["test_timing"] = json.loads((test_path / "timing.json")
                                           .read_text(encoding="utf-8"))
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    lines = ["# Смешивание оценок B3000 и B15000", "",
             "Контроль — сохранённый B15000. Для каждого запроса обе модели оценили один и тот же полный прежний пул кандидатов. Каждая шкала центрируется и делится на стандартное отклонение по этому пулу; при нулевом отклонении используется нулевой вектор. Затем проверены только заранее заданные α = 0.25, 0.50, 0.75. E5, история, корпус и признаки не менялись.", "",
             f"SHA256 контрольного B15000 CSV: `{control['answer_sha256']}`. Оценка полного dev заняла {result['dev_timing']['seconds']:.1f} с, пик одного рабочего процесса {result['dev_timing']['worker_peak_rss_bytes']/2**30:.2f} GiB. Обе контрольные выдачи воспроизведены по каждому запросу.", "",
             "## Полный dev", "",
             "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | Своя локация | Другая локация |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for scheme, models in dev_summary.items():
        for name, row in models.items():
            lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | {row['pool_recall']:.4f} | {row['selection_losses']} | {row['same_location_recall50']:.4f} | {row['other_location_recall50']:.4f} |")
    lines += ["", "Парный bootstrap по кластерам нормализованного текста: 1000 повторов, seed 20260926. Интервалы диагностические и не скорректированы на три проверенные смеси.", "",
              "| Сценарий | Смесь против B15000 | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
              "| --- | --- | ---: | --- | ---: |"]
    for scheme, entries in pairs.items():
        for name, row in entries.items():
            lo, hi = row["ci95"]
            lines.append(f"| {scheme} | {name} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['improved']} / {row['worsened']} |")
    if "decision" in result:
        lines += ["", f"По полному dev выбран **{result['decision']['selected']}**: нижняя граница парного интервала на union положительная, средний Recall@50 в benchmark + ответы не ниже B15000. Среди прошедших смесей выбирается максимальный union/dev Recall@50."]
    if "test_summary" in result:
        lines += ["", "## Диагностический test", "",
                  f"Оценка test заняла {result['test_timing']['seconds']:.1f} с; пик одного рабочего процесса {result['test_timing']['worker_peak_rss_bytes']/2**30:.2f} GiB.", "",
                  "| Сценарий | Вариант | Recall@50 | Потери отбора | Своя локация | Другая локация |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for scheme, models in result["test_summary"].items():
            for name, row in models.items():
                lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | {row['selection_losses']} | {row['same_location_recall50']:.4f} | {row['other_location_recall50']:.4f} |")
        lines += ["", "| Сценарий | Δ к B15000 | 95% интервал | Улучшено / ухудшено |",
                  "| --- | ---: | --- | ---: |"]
        for scheme, row in result["test_comparisons"].items():
            lo, hi = row["ci95"]
            lines.append(f"| {scheme} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['improved']} / {row['worsened']} |")
    if (OUT / "fallback_repeat.json").exists():
        saved = json.loads((OUT / "fallback_repeat.json").read_text(encoding="utf-8"))
        lines += ["", f"Смесь не выбрана. B15000 пересчитан без сети: SHA256 повторного CSV `{saved['repeat_sha256']}` совпал с контрольным `{saved['control_sha256']}`. Текущим ответом остаётся `answer_semantic_15000.csv`."]
    config = Path("models/semantic_blend_v1/config.json")
    if config.exists():
        saved = json.loads(config.read_text(encoding="utf-8"))
        lines += ["", f"Подготовлен `{saved['answer']}` с α={saved['alpha']:.2f}: SHA256 `{saved['answer_sha256']}`; {saved['validation']['queries']} строк по {saved['validation']['min_candidates']}–{saved['validation']['max_candidates']} кандидатов. Два офлайн-прогона совпали побайтно."]
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
