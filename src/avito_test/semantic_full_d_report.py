"""Render the complete D-pool experiment and its frozen dev decision."""
# Report selection losses separately from retrieval misses.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .semantic_full_d_evaluate import CONTROL, DECISION, NEW, SCHEMES
from .semantic_full_d_oof import OUT
from .semantic_full_d_pool_data import find_folder
from .semantic_report import compare


REPORT = Path("reports/semantic_full_D.md")


def row(table: pl.DataFrame, scheme: str, variant: str) -> dict:
    selected = table.filter((pl.col("scheme") == scheme) &
                            (pl.col("variant") == variant)).to_dicts()
    if len(selected) != 1:
        raise ValueError(f"Missing metrics: {scheme}/{variant}")
    return selected[0]


def render() -> Path:
    folder = find_folder(0)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    audit = manifest["audit"]
    quantized = json.loads((folder / "quantized.json").read_text(encoding="utf-8"))
    dev = pl.read_csv(OUT / "evaluation/dev/summary.csv")
    frame = pl.read_parquet(OUT / "evaluation/dev/per_query.parquet")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    timing = json.loads((OUT / "evaluation/dev/timing.json").read_text(encoding="utf-8"))
    lines = [
        "# Обучение CatBoost на полном D-пуле кандидатов", "",
        "Контроль — отправленный B_full_balanced, Recall@50 платформы 0.786292 "
        "(сообщил пользователь). D добавляет к прежнему B-пулу до 200 глобальных "
        "и до 200 локальных кандидатов E5. Запросы, три OOF-части, веса E5, "
        "параметры поиска и CatBoost неизменны.", "",
        "## Подготовка обучающего пула", "",
        f"На {audit['queries']:,} OOF-запросах известно "
        f"{audit['known_positives']:,} положительных пар. D нашёл "
        f"{audit['found_positives']:,} пар, из них "
        f"{audit['semantic_only_found']:,} отсутствовали в B-пуле. "
        f"Для PairLogit пригодны {audit['used_groups']:,} групп; "
        f"B находил {audit['b_found']:,} пар в 13 366 группах. "
        "Ненайденные положительные объявления не добавлялись в обучающий пул.", "",
        f"Полная D-таблица: {audit['full_rows']:,} строк, "
        f"{audit['full_negative_rows']:,} неразмеченных альтернатив и "
        f"{audit['full_pairs']:,} автоматически создаваемых пар против "
        f"{audit['old_pairs']:,} пар при 150 альтернативах. "
        "В варианте balanced вес каждой группы равен "
        "`150 / число неразмеченных альтернатив полного D-пула`; "
        "в unweighted он равен 1. Оба обучения используют один квантованный "
        f"Pool размером {quantized['bytes']/2**30:.2f} GiB.", "",
        "Признаки E5, зависящие от состава кандидатов (ранг и разрыв до "
        "лучшего сходства), пересчитаны на полном D-пуле; позиции глобального "
        "и локального каналов рассчитаны по их фактическим выдачам. "
        "При benchmark + ответы корпус ограничивался до смыслового топ-200.", "",
        f"Подготовка OOF заняла {manifest['seconds']:.1f} с "
        f"(пик {manifest['peak_rss_bytes']/2**30:.2f} GiB), "
        f"квантизация — {quantized['seconds']:.1f} с "
        f"(пик {quantized['peak_rss_bytes']/2**30:.2f} GiB).", "",
        "| Вариант | Обучение, с | Пик процесса, GiB | Пар |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("unweighted", "balanced"):
        train = json.loads((OUT / f"full_{name}.json").read_text(encoding="utf-8"))
        lines.append(f"| D_full_{name} | {train['train_seconds']:.1f} | "
                     f"{train['peak_rss_bytes']/2**30:.2f} | "
                     f"{train['automatically_generated_pairs']:,} |")
    lines += [
        "", "## Полный dev", "",
        "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | Своя локация | Другая локация |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme in SCHEMES:
        for name in (CONTROL, "D15000", *NEW):
            value = row(dev, scheme, name)
            lines.append(f"| {scheme} | {name} | {value['recall50']:.4f} | "
                         f"{value['pool_recall']:.4f} | "
                         f"{value['selection_losses']} | "
                         f"{value['same_location_recall50']:.4f} | "
                         f"{value['other_location_recall50']:.4f} |")
    lines += [
        "",
        "Ниже изменения относительно отправленного B_full_balanced. "
        "«Новые» — положительные объявления только смысловых каналов, "
        "«возвращены» — объявления старого B-пула, которые B не выбрал, "
        "«потеряны» — положительные объявления из выдачи B. "
        "Macro-вклады усредняются по полным запросам и потому не равны "
        "простому отношению суммарных количеств к числу пар.", "",
        "| Сценарий | Вариант | Новые в 50 | Возвращены | Потеряны | Вклад новых | Вклад возврата | Вклад потерь |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme in SCHEMES:
        for name in ("D15000", *NEW):
            value = row(dev, scheme, name)
            lines.append(
                f"| {scheme} | {name} | "
                f"{value['new_semantic_positives_selected']} | "
                f"{value['old_positive_regained']} | "
                f"{value['old_positive_lost']} | "
                f"+{value['new_macro_gain']:.4f} | "
                f"+{value['old_macro_gain']:.4f} | "
                f"-{value['old_macro_loss']:.4f} |"
            )
    lines += [
        "",
        "Парный bootstrap по кластерам нормализованного текста: 1000 повторов. "
        "Интервалы диагностические и не учитывают выбор между двумя D-моделями.",
        "",
        "| Сценарий | Вариант | Δ Recall@50 к B | 95% интервал | Улучшено / ухудшено запросов |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for scheme in SCHEMES:
        for name in ("D15000", *NEW):
            result = compare(frame, scheme, name, CONTROL)
            lo, hi = result["ci95"]
            lines.append(f"| {scheme} | {name} | {result['delta']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | "
                         f"{result['improved']} / {result['worsened']} |")
    lines += [
        "",
        f"Решение по dev: **{decision['selected']}**. Проход требуется при "
        "положительной нижней границе парного интервала на union/dev и "
        "отсутствии среднего ухудшения в benchmark + ответы/dev. "
        "При различии union/dev не более 0.001 между прошедшими моделями "
        "предпочтение отдаётся более сильной во втором сценарии.",
        f"Оценка по сохранённым dev-признакам заняла {timing['seconds']:.1f} с; "
        f"пик процесса {timing['peak_rss_bytes']/2**30:.2f} GiB.",
    ]
    test = OUT / "evaluation/test/summary.csv"
    if test.exists():
        table = pl.read_csv(test)
        lines += [
            "", "## Диагностический test зафиксированного решения", "",
            "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | Своя локация | Другая локация |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for scheme in SCHEMES:
            for name in (CONTROL, decision["selected"]):
                value = row(table, scheme, name)
                lines.append(f"| {scheme} | {name} | {value['recall50']:.4f} | "
                             f"{value['pool_recall']:.4f} | "
                             f"{value['selection_losses']} | "
                             f"{value['same_location_recall50']:.4f} | "
                             f"{value['other_location_recall50']:.4f} |")
        lines += ["", "Выбор модели был зафиксирован до просмотра test."]
    config = Path("models/semantic_full_D_v1/config.json")
    if config.exists():
        result = json.loads(config.read_text(encoding="utf-8"))
        lines += ["", f"Файл `{result['answer']}`: SHA256 "
                  f"`{result['answer_sha256']}`; два офлайн-прогона "
                  "совпали побайтно."]
    elif decision["selected"] == CONTROL:
        lines += ["", "Новый CSV не создавался; контролем остаётся "
                  "`answer_semantic_full.csv`."]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "selected": decision["selected"]}),
          flush=True)
    return REPORT


if __name__ == "__main__":
    render()
