"""Write the capacity experiment and its dev-only selection in Russian."""
# Read saved measurements so reporting cannot trigger another search.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .semantic_tune_cache import OUT
from .semantic_tune_evaluate import CONTROL
from .semantic_tune_final import VARIANTS


REPORT = Path("reports/semantic_tune.md")


def _metrics(table: pl.DataFrame, scheme: str, variant: str) -> dict:
    rows = table.filter((pl.col("scheme") == scheme) &
                        (pl.col("variant") == variant)).to_dicts()
    if len(rows) != 1:
        raise ValueError(f"Missing metrics for {scheme}/{variant}")
    return rows[0]


def render() -> Path:
    checkpoint = pl.read_csv(OUT / "checkpoints/summary.csv")
    chosen = json.loads((OUT / "checkpoint_decision.json").read_text(encoding="utf-8"))
    final = pl.read_csv(OUT / "final/dev/summary.csv")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    checkpoint_timing = json.loads((OUT / "checkpoints/timing.json").read_text(
        encoding="utf-8"))
    cache_folder = Path(checkpoint_timing["cache"])
    cache = json.loads((cache_folder / "manifest.json").read_text(encoding="utf-8"))
    cache_bytes = sum(path.stat().st_size for path in cache_folder.glob("part_*.npz"))
    if chosen["selected_model_sha256"] != decision["checkpoint_model_sha256"]:
        raise ValueError("Checkpoint and final decision disagree")
    lines = ["# Глубина и длительность обучения CatBoost на B15000", "",
             "Контроль — отправленная смесь B3000/B15000 с α=0.75 (платформа: Recall@50 **0.744280**, сообщил пользователь). Обе новые модели обучены на одном и том же сохранённом OOF-файле B15000: 2 022 658 строк, 13 366 групп, 17 758 положительных пар. Порядок 44 признаков, метки, до 150 альтернатив на запрос, PairLogit, learning rate 0.06, seed и регуляризация не менялись. Отличаются только глубина 6 или 8 и число использованных деревьев.", "",
             f"Полный прежний пул кандидатов и его признаки закэшированы для всех {cache['queries']} dev-запросов в двух сценариях: {cache['rows']:,} строк, {cache_bytes/2**30:.2f} GiB на диске, подготовка {cache['seconds']:.1f} с, пик одного рабочего процесса {cache['worker_peak_rss_bytes']/2**30:.2f} GiB. BM25, история и E5 не пересчитываются между контрольными точками. Отсутствующие в пуле положительные ответы остаются в знаменателе macro Recall@50.", "",
             "## Два обучения", "",
             "| Глубина | Максимум деревьев | Время обучения, с | Пик памяти процесса, GiB |",
             "| ---: | ---: | ---: | ---: |"]
    for depth in (6, 8):
        record = json.loads((OUT / f"depth_{depth}_960.json").read_text(encoding="utf-8"))
        lines.append(f"| {depth} | 960 | {record['train_seconds']:.1f} | "
                     f"{record['peak_rss_bytes']/2**30:.2f} |")
    lines += ["", "Обучение глубины 6 шло одновременно с построением кэша dev; глубина 8 обучалась после него. Поэтому измеренные времена показывают стоимость фактических запусков, но не служат чистым сравнением скорости двух глубин."]
    lines += ["", "## Полный dev: точки обучения", "",
              "| Глубина | Деревьев | Union Recall@50 | Benchmark + ответы Recall@50 | Другая локация, union |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for depth in (6, 8):
        for trees in (240, 480, 720, 960):
            name = f"d{depth}_t{trees}"
            union = _metrics(checkpoint, "union", name)
            augmented = _metrics(checkpoint, "benchmark_plus_positives", name)
            lines.append(f"| {depth} | {trees} | {union['recall50']:.4f} | "
                         f"{augmented['recall50']:.4f} | "
                         f"{union['other_location_recall50']:.4f} |")
    for scheme in ("union", "benchmark_plus_positives"):
        control = _metrics(checkpoint, scheme, CONTROL)
        b15 = _metrics(checkpoint, scheme, "B15000")
        expected = 0.881834554125325 if scheme == "union" else 0.9268887310836696
        if abs(control["pool_recall"] - expected) > 1e-12:
            raise ValueError("Pool completeness changed")
        lines += ["", f"{scheme}: отправленная смесь {control['recall50']:.4f}; одиночный B15000 {b15['recall50']:.4f}; полнота неизменного пула {control['pool_recall']:.4f}."]
    lines += ["", f"Для финального сравнения на dev зафиксирована новая модель **{chosen['selected_checkpoint']}**. Правило выбора контрольной точки: максимум union/dev среди моделей без среднего ухудшения во втором сценарии относительно отправленной смеси; при отсутствии таких моделей — максимум union/dev. Шаги 240, 480, 720 и 960 получены из двух обучений; выбранная часть модели сохранена через CatBoost `shrink`.",
              "", "## Финальное сравнение на dev", "",
              "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | Своя локация | Другая локация |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for scheme in ("union", "benchmark_plus_positives"):
        for name in (CONTROL, "B15000", *VARIANTS):
            row = _metrics(final, scheme, name)
            lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | "
                         f"{row['pool_recall']:.4f} | {row['selection_losses']} | "
                         f"{row['same_location_recall50']:.4f} | "
                         f"{row['other_location_recall50']:.4f} |")
    lines += ["", "Парный bootstrap по кластерам нормализованного текста, 1000 повторов. После выбора среди восьми контрольных точек интервалы диагностические и не скорректированы на подбор модели.", "",
              "| Сценарий | Вариант против отправленной смеси | Δ Recall@50 | 95% интервал | Улучшено / ухудшено запросов |",
              "| --- | --- | ---: | --- | ---: |"]
    for scheme in ("union", "benchmark_plus_positives"):
        for name in VARIANTS:
            comparison = decision["dev_comparisons"][scheme][name]
            lo, hi = comparison["ci95"]
            lines.append(f"| {scheme} | {name} | {comparison['delta']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | "
                         f"{comparison['improved']} / {comparison['worsened']} |")
    lines += ["", f"Выбор по dev: **{decision['selected']}**. Требовались положительная нижняя граница парного интервала против отправленной смеси на union/dev и отсутствие среднего ухудшения в benchmark + ответы/dev. Превосходство над одиночным B15000 не заменяет этого условия."]
    for name, title in (("checkpoints", "Оценка восьми точек"),
                        ("final/dev", "Оценка финальных вариантов")):
        timing = json.loads((OUT / name / "timing.json").read_text(encoding="utf-8"))
        lines.append(f"{title}: {timing['seconds']:.1f} с на сохранённых признаках; "
                     f"пик процесса {timing['peak_rss_bytes']/2**30:.2f} GiB.")
    lines.append("Чистое получение оценок моделей по полному dev-кэшу: "
                 f"глубина 6 — {checkpoint_timing['model_prediction_seconds']['d6']:.1f} с, "
                 f"глубина 8 — {checkpoint_timing['model_prediction_seconds']['d8']:.1f} с; "
                 "общее время выше включает чтение кэша, выбор топ-50 и расчёт метрик.")
    test = OUT / "final/test/summary.csv"
    if test.exists():
        table = pl.read_csv(test)
        lines += ["", "## Диагностический test зафиксированного решения", "",
                  "| Сценарий | Вариант | Recall@50 | Потери отбора | Своя локация | Другая локация |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for scheme in ("union", "benchmark_plus_positives"):
            for name in (CONTROL, *VARIANTS):
                row = _metrics(table, scheme, name)
                lines.append(f"| {scheme} | {name} | {row['recall50']:.4f} | "
                             f"{row['selection_losses']} | "
                             f"{row['same_location_recall50']:.4f} | "
                             f"{row['other_location_recall50']:.4f} |")
    package = Path("models/semantic_tune_v1/config.json")
    if package.exists():
        config = json.loads(package.read_text(encoding="utf-8"))
        lines += ["", f"Подготовлен `{config['answer']}`: SHA256 `{config['answer_sha256']}`; "
                  "2452 запросов по 50 уникальных допустимых объявлений. Два запуска "
                  "без сети дали одинаковый файл."]
    elif decision["selected"] == CONTROL:
        lines += ["", "Новый CSV не создавался. Текущим ответом остаётся "
                  "`answer_semantic_blend.csv`, SHA256 "
                  "`8124b99e143fcc1091705be80e9bc4c2b62a42c3cf600909ae352afdf54ee707`. "
                  "Проверка формата и зависимостей повторно прошла без сети."]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "selected": decision["selected"]}),
          flush=True)
    return REPORT


if __name__ == "__main__":
    render()
