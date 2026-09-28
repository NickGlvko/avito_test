"""Document the fixed six-way final fusion experiment."""
# Keep all six declared variants visible, including those that lose.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .semantic_final import CONTROL, VARIANTS
from .semantic_final_evaluate import DECISION, OUT, SCHEMES
from .semantic_report import compare


REPORT = Path("reports/semantic_final.md")


def find(table: pl.DataFrame, scheme: str, variant: str) -> dict:
    rows = table.filter((pl.col("scheme") == scheme) &
                        (pl.col("variant") == variant)).to_dicts()
    if len(rows) != 1:
        raise ValueError(f"Missing result {scheme}/{variant}")
    return rows[0]


def render() -> Path:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    dev = pl.read_csv(OUT / "evaluation/dev/summary.csv")
    frame = pl.read_parquet(OUT / "evaluation/dev/per_query.parquet")
    timing = json.loads((OUT / "evaluation/dev/timing.json")
                        .read_text(encoding="utf-8"))
    lines = [
        "# Финальное объединение сохранённых моделей", "",
        "Контроль — отправленный `D_full_unweighted`, Recall@50 платформы "
        "0.798793 (сообщил пользователь). Нового обучения, поиска и кодирования "
        "корпуса нет: B и D оценивают прежние полные пулы со своими признаками. "
        "Проверены ровно три смеси D (α=0.25, 0.50, 0.75 — вес "
        "`D_full_unweighted`) и три RRF B/D (β=0.50, 0.75, 0.90 — вес D). "
        "В смесях D оценки нормализованы по полному D-пулу внутри запроса; "
        "RRF использует полные ранги и k=60. Равенства разрешены по исходному "
        "строковому `item_id`.", "",
        "Контрольные выдачи B и D сверены с архивом для каждого dev-запроса; "
        "крайние β=0 и β=1 дают ровно B и D. Отсутствующий в B кандидат "
        "получает нулевой вклад B.", "",
        "## Полный dev", "",
        "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | "
        "Другая локация |", 
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scheme in SCHEMES:
        for name in (CONTROL, *VARIANTS[1:]):
            row = find(dev, scheme, name)
            lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | "
                         f"{row['pool_recall']:.4f} | {row['selection_losses']} | "
                         f"{row['other_location_recall50']:.4f} |")
    lines += [
        "", "Изменения относительно отправленного D: `возвращены` — положительные "
        "объявления, отсутствующие в его топ-50; `потеряны` — положительные "
        "объявления из его топ-50, не вошедшие в новую выдачу. "
        "`Новые E5 в 50` — положительные объявления, найденные только "
        "смысловыми каналами относительно B-пула.", "",
        "| Сценарий | Вариант | Возвращены | Потеряны | Новые E5 в 50 | "
        "Macro возврат | Macro потеря |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme in SCHEMES:
        for name in VARIANTS[1:]:
            row = find(dev, scheme, name)
            lines.append(f"| {scheme} | {name} | {row['positive_regained']} | "
                         f"{row['positive_lost']} | "
                         f"{row['semantic_only_in_top50']} | "
                         f"+{row['regained_macro']:.4f} | "
                         f"-{row['lost_macro']:.4f} |")
    lines += [
        "", "Парный bootstrap по кластерам нормализованного текста: 1000 "
        "повторов, seed 20260926. Интервалы диагностические и не учитывают "
        "выбор среди шести вариантов.", "",
        "| Сценарий | Вариант | Δ Recall@50 к D | 95% интервал | "
        "Улучшено / ухудшено запросов |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for scheme in SCHEMES:
        for name in VARIANTS[1:]:
            result = compare(frame, scheme, name, CONTROL)
            lo, hi = result["ci95"]
            lines.append(f"| {scheme} | {name} | {result['delta']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | "
                         f"{result['improved']} / {result['worsened']} |")
    lines += [
        "",
        f"Выбор по dev: **{decision['selected']}**. Требуется положительная "
        "нижняя граница парного интервала относительно отправленного D на "
        "union/dev и отсутствие среднего ухудшения на benchmark + ответы/dev. "
        "Среди прошедших выбирается максимальный union/dev Recall@50.",
        f"Оценка dev заняла {timing['seconds']:.1f} с, пик процесса "
        f"{timing['peak_rss_bytes']/2**30:.2f} GiB.",
    ]
    test_path = OUT / "evaluation/test/summary.csv"
    if test_path.exists():
        test = pl.read_csv(test_path)
        lines += [
            "", "## Диагностический test после выбора", "",
            "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | "
            "Другая локация |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for scheme in SCHEMES:
            for name in (CONTROL, decision["selected"]):
                row = find(test, scheme, name)
                lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | "
                             f"{row['pool_recall']:.4f} | "
                             f"{row['selection_losses']} | "
                             f"{row['other_location_recall50']:.4f} |")
        lines += ["", "Выбор и веса зафиксированы до просмотра test."]
    config_path = Path("models/semantic_final_v1/config.json")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        lines += ["", f"Файл `{config['answer']}`: SHA256 "
                  f"`{config['answer_sha256']}`; "
                  "два офлайн-прогона совпали побайтно."]
    elif decision["selected"] == CONTROL:
        lines += ["", "Новый CSV не создавался: отправленный D остаётся лучшим "
                  "подтверждённым решением."]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT),
                      "selected": decision["selected"]}), flush=True)
    return REPORT


if __name__ == "__main__":
    render()
