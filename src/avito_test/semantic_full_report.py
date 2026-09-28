"""Summarize the frozen full-pool PairLogit experiment."""
# Compare full-pool variants against the previously submitted control.
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .semantic_report import compare
from .semantic_full_evaluate import CONTROL, VARIANTS
from .semantic_full_oof import OUT
from .semantic_full_pool_data import find_folder


REPORT = Path("reports/semantic_full_pool.md")
SCHEMES = ("union", "benchmark_plus_positives")
SOURCES = ("bm25_geo16", "char_geo16", "bm25_stats", "char_stats", "history", "e5")


def metric(table: pl.DataFrame, scheme: str, variant: str) -> dict:
    rows = table.filter((pl.col("scheme") == scheme) &
                        (pl.col("variant") == variant)).to_dicts()
    if len(rows) != 1:
        raise ValueError(f"Missing full-pool metrics: {scheme}/{variant}")
    return rows[0]


def render() -> Path:
    folder = find_folder(0)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    audit = manifest["audit"]
    quantized = json.loads((folder / "quantized.json").read_text(encoding="utf-8"))
    queries = pl.read_parquet(folder / "queries.parquet")
    if queries.height != 15000 or audit["used_groups"] != 13366:
        raise ValueError("Incomplete full OOF audit")
    dev = pl.read_csv(OUT / "evaluation/dev/summary.csv")
    frame = pl.read_parquet(OUT / "evaluation/dev/per_query.parquet")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    timing = json.loads((OUT / "evaluation/dev/timing.json").read_text(encoding="utf-8"))

    lines = [
        "# Обучение CatBoost на полном B-пуле кандидатов", "",
        "Контроль — отправленная смесь B3000/B15000 с весом B15000 0.75 "
        "(Recall@50 платформы 0.744280, сообщён пользователем). Изменён только состав "
        "неразмеченных альтернатив при обучении B. Запросы, три OOF-части, признаки, "
        "источники кандидатов, E5 и параметры PairLogit прежние.", "",
        "## Что покрывали прежние 150 альтернатив", "",
        "На каждый пригодный запрос брали 70 первых неразмеченных кандидатов по RRF "
        "пяти источников, затем до 30 кандидатов только одного источника (тоже по "
        "RRF), затем случайно добирали до 150 из остальных. Все найденные "
        "положительные кандидаты сохранялись сверх лимита. Случайный seed привязан "
        "к group_id. Отдельного отбора по E5 не было.", "",
        f"Фактически на {audit['used_groups']:,} пригодных группах выбрано "
        f"{audit['sample_top70']:,} из первой категории, "
        f"{audit['sample_single_source']:,} из второй, "
        f"{audit['sample_random']:,} случайно; всего "
        f"{audit['old_negative_rows']:,} = 150 на группу.", "",
        "Покрытие означает долю *неразмеченных* объявлений из верхних K позиций "
        "канала, которые были в прежней обучающей таблице. E5 здесь — порядок "
        "сходства внутри старого B-пула, а не отдельный источник кандидатов. "
        "Срез «все» включает запросы без найденного положительного ответа: они "
        "участвуют в оценке, но не в PairLogit.", "",
        "| Источник | K | Пригодные группы | Все 15 000 групп |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in SOURCES:
        for top in (10, 50, 100, 500):
            used = audit["coverage"][f"usable/{name}@{top}"]
            all_groups = audit["coverage"][f"all/{name}@{top}"]
            lines.append(
                f"| {name} | {top} | {used[0]/used[1]:.3f} "
                f"({used[0]:,}/{used[1]:,}) | "
                f"{all_groups[0]/all_groups[1]:.3f} "
                f"({all_groups[0]:,}/{all_groups[1]:,}) |"
            )
    lines += [
        "",
        f"Вне прежней обучающей таблицы остались "
        f"{audit['top50_any_source_missed']:,} неразмеченных кандидатов из топ-50 "
        f"хотя бы одного исходного канала (сумма по запросам после удаления "
        f"повторов внутри каждого запроса), а также {audit['e5_top50_missed']:,} "
        f"из топ-50 по E5 внутри пула. Следовательно, прежняя выборка пропускала "
        f"и сильных конкурентов, а не только длинный хвост.", "",
        "## Полный пул, пары и память", "",
        f"Сформировано {audit['full_rows']:,} строк для "
        f"{audit['used_groups']:,} пригодных групп: "
        f"{audit['found_positives']:,} положительных и "
        f"{audit['full_negative_rows']:,} неразмеченных. Во всех 15 000 группах "
        f"известно {audit['known_positives']:,} положительных объявлений. "
        "Пропущенные поиском положительные объявления в пул не добавлялись. "
        "Число найденных положительных и пригодных групп совпало с B15000.", "",
        f"В прежней таблице было {audit['old_pairs']:,} пар с разными метками; "
        f"в полном пуле — {audit['full_pairs']:,}. Это подсчёт всех возможных "
        "положительно-неразмеченных пар в группах. В настройках нет max_pairs; "
        "по [документации CatBoost]"
        "(https://catboost.ai/docs/en/concepts/loss-functions-ranking) "
        "его значение по умолчанию — все возможные пары. Обычные веса строк "
        "PairLogit не использует, а вес автоматически созданной пары умножается "
        "на вес группы.", "",
        "Во втором варианте каждому объекту группы задан одинаковый "
        "group_weight = 150 / число неразмеченных кандидатов полного пула. "
        "Положительные объявления не изменены, поэтому суммарная масса пар "
        "каждой группы остаётся на уровне прежней таблицы. Веса переданы через "
        "[CatBoost Pool](https://catboost.ai/docs/en/concepts/python-reference_pool).",
        "",
        f"Пакеты признаков хранятся как float32, для обучения один раз сохранён "
        f"квантованный пул размером {quantized['bytes']/2**30:.2f} GiB. "
        f"Подготовка OOF заняла {manifest['seconds']:.1f} с "
        f"(пик {manifest['peak_rss_bytes']/2**30:.2f} GiB); "
        f"квантизация — {quantized['seconds']:.1f} с "
        f"(пик {quantized['peak_rss_bytes']/2**30:.2f} GiB).", "",
        "| Вариант | Обучение, с | Пик процесса, GiB | Пар |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in VARIANTS:
        train = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        lines.append(
            f"| {name} | {train['train_seconds']:.1f} | "
            f"{train['peak_rss_bytes']/2**30:.2f} | "
            f"{train['automatically_generated_pairs']:,} |"
        )
    weights = queries.filter(pl.col("used_for_training"))["new_group_weight"]
    lines += [
        "",
        f"Скорректированный вес группы: медиана {weights.median():.3f}, "
        f"минимум {weights.min():.3f}, максимум {weights.max():.3f}.", "",
        "## Полный dev", "",
        "| Сценарий | Вариант | Recall@50 | C@pool | Потери отбора | Своя локация | Другая локация |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme in SCHEMES:
        for name in (CONTROL, "B15000", *VARIANTS):
            row = metric(dev, scheme, name)
            lines.append(
                f"| {scheme} | {name} | {row['recall50']:.4f} | "
                f"{row['pool_recall']:.4f} | {row['selection_losses']} | "
                f"{row['same_location_recall50']:.4f} | "
                f"{row['other_location_recall50']:.4f} |"
            )
    lines += [
        "",
        "Парный bootstrap по кластерам нормализованного текста, 1000 повторов. "
        "Интервалы диагностические, без поправки на выбор между двумя моделями.", "",
        "| Сценарий | Вариант | Контроль | Δ Recall@50 | 95% интервал | Улучшено / ухудшено |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for scheme in SCHEMES:
        for name in VARIANTS:
            for baseline in ("B15000", CONTROL):
                comparison = compare(frame, scheme, name, baseline)
                lo, hi = comparison["ci95"]
                lines.append(
                    f"| {scheme} | {name} | {baseline} | "
                    f"{comparison['delta']:+.4f} | "
                    f"[{lo:+.4f}, {hi:+.4f}] | "
                    f"{comparison['improved']} / {comparison['worsened']} |"
                )
    lines += [
        "",
        f"По dev выбрано: **{decision['selected']}**. Новый вариант должен был "
        "получить положительную нижнюю границу парного интервала против "
        "отправленной смеси на union/dev и не ухудшить средний Recall@50 "
        "в benchmark + ответы/dev.", "",
        f"Оценка на сохранённых полных dev-признаках заняла {timing['seconds']:.1f} с; "
        f"пик процесса {timing['peak_rss_bytes']/2**30:.2f} GiB. "
        f"Время получения оценок моделей: "
        f"{timing['model_predict_seconds']['full_unweighted']:.1f} с и "
        f"{timing['model_predict_seconds']['full_balanced']:.1f} с.",
    ]
    test = OUT / "evaluation/test/summary.csv"
    if test.exists():
        table = pl.read_csv(test)
        test_timing = json.loads(
            (OUT / "evaluation/test/timing.json").read_text(encoding="utf-8")
        )
        test_cache_path = Path(test_timing["cache"])
        test_cache = json.loads(
            (test_cache_path / "manifest.json").read_text(encoding="utf-8")
        )
        lines += [
            "", "## Диагностический test зафиксированного решения", "",
            "| Сценарий | Вариант | Recall@50 | Потери отбора | Своя локация | Другая локация |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for scheme in SCHEMES:
            for name in (CONTROL, *VARIANTS):
                row = metric(table, scheme, name)
                lines.append(
                    f"| {scheme} | {name} | {row['recall50']:.4f} | "
                    f"{row['selection_losses']} | "
                    f"{row['same_location_recall50']:.4f} | "
                    f"{row['other_location_recall50']:.4f} |"
                )
        lines += [
            "",
            "Модель full_balanced зафиксирована до просмотра test; отдельный "
            "full_unweighted показан только для диагностики и не меняет выбор.",
            f"Подготовка test-кэша заняла {test_cache['seconds']:.1f} с, "
            f"оценка сохранённых признаков — {test_timing['seconds']:.1f} с.",
        ]
    package = Path("models/semantic_full_v1/config.json")
    if package.exists():
        config = json.loads(package.read_text(encoding="utf-8"))
        lines += [
            "",
            f"Файл {config['answer']}: SHA256 {config['answer_sha256']}; "
            "два офлайн-предсказания совпали побайтно. "
            f"Время одного запуска {config['seconds']:.1f} с, "
            f"пик процесса {config['peak_rss_bytes']/2**30:.2f} GiB.",
        ]
    elif decision["selected"] == CONTROL:
        lines += [
            "", "Новый CSV не создан. Контролем остаётся answer_semantic_blend.csv, "
            "SHA256 8124b99e143fcc1091705be80e9bc4c2b62a42c3cf600909ae352afdf54ee707.",
        ]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "selected": decision["selected"]}),
          flush=True)
    return REPORT


if __name__ == "__main__":
    render()
