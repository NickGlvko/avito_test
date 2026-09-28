"""Build a reviewable experiment report from saved measurements, without new retrieval."""
# Use the same metric definitions in tables and saved evaluations.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import SEED, write_json


def build_report(output: Path, config_path: Path, destination: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = {name: json.loads((output / name / "summary.json").read_text(encoding="utf-8"))
            for name in ["group_dev", "group_test", "unseen_dev"]}
    best = config["experiment"]
    dev = pl.read_parquet(output / "group_dev" / "per_query.parquet")
    best_scores = dev.filter(pl.col("experiment") == best).sort("group_id")["recall@50"].to_numpy()
    comparisons = []
    for name in ["rrf_stats", "rrf_geo16", "rrf_history"]:
        other = dev.filter(pl.col("experiment") == name).sort("group_id")["recall@50"].to_numpy()
        delta = best_scores - other
        rng = np.random.default_rng(SEED)
        samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
        comparisons.append({"compared_with": name, "delta": float(delta.mean()),
                            "ci95": np.quantile(samples, [.025, .975]).tolist(),
                            "improved_queries": int((delta > 0).sum()),
                            "worse_queries": int((delta < 0).sum())})
    write_json(output / "paired_comparisons.json", comparisons)
    lines = ["# Результаты первого CPU baseline", "",
             "Дополнение: на платформе получен Recall@50 **0.666981**. Этот файл сохраняет "
             "результаты первого этапа. Расширенное сравнение корпусов описано в "
             "[отдельном отчёте](extended_validation.md).", "",
             "Конфигурация выбрана только на group/dev и затем зафиксирована. "
             "Это результаты локальной проверки, не оценки платформы.", "",
             "## Эксперименты на dev", "",
             "| Вариант | Recall@50 | Recall@100 | Recall@500 | Полнота объединения источников варианта |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in runs["group_dev"]["metrics"]:
        lines.append(f"| {row['experiment']} | {row['recall@50']:.4f} | {row['recall@100']:.4f} | "
                     f"{row['recall@500']:.4f} | {row['selected_union_recall']:.4f} |")
    lines += ["", f"Выбран `{best}`. Веса источников: `{config['weights']}`. "
              "Глубина каждого источника — до 500, RRF k=60. "
              "Исследованы исходный текстовый поиск, два множителя локации, "
              "мягкие статистики и два веса истории. Остальные параметры не подбирались.", "",
              "## Проверка зафиксированной конфигурации", "",
              "| Выборка | Всего групп части | Исключено: нет всех ответов в корпусе | Оценено | Положительных | Recall@50 | Recall@500 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, run in runs.items():
        e = run["eligibility"]
        metric = next(row for row in run["metrics"] if row["experiment"] == best)
        lines.append(f"| {name} | {e['total_groups']} | {e['excluded_missing_positives']} | "
                     f"{e['evaluated_groups']} | {e['evaluated_positives']} | "
                     f"{metric['recall@50']:.4f} | {metric['recall@500']:.4f} |")
    lines += ["", "В каждом случае поиск выполняется по всем 189 212 объявлениям. "
              "Группы не пересекаются с обучающей частью своего протокола. В unseen "
              "из обучения исключены все варианты текста проверочных запросов. "
              "Unseen — отдельное диагностическое разбиение, оно может пересекаться "
              "с dev/test основного разбиения; это не третий независимый тест.", "",
              "Ограничение: в корпусе присутствует лишь около 5% уникальных объявлений train. "
              "Оценка на доступных положительных группах может быть смещена в пользу истории. "
              "Проверка на объединённом корпусе ещё не проведена.", "",
              "## Устойчивость на dev", "",
              "Парный bootstrap по полным запросам, 2000 повторов, seed 20260926. "
              "Интервалы диагностические: без поправки на выбор лучшего варианта; "
              "запросы с одинаковым текстом могут зависеть друг от друга.", "",
              "| Сравнение победителя с | Δ Recall@50 | 95% интервал | Улучшено запросов | Ухудшено |",
              "| --- | ---: | --- | ---: | ---: |"]
    for c in comparisons:
        lines.append(f"| {c['compared_with']} | {c['delta']:+.4f} | "
                     f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                     f"{c['improved_queries']} | {c['worse_queries']} |")
    lines += ["", "Небольшое отличие двух весов истории само по себе не доказывает "
              "превосходство одного веса; окончательный выбор следует правилу максимума dev.", "",
              "## Где теряются положительные объявления", "",
              "Здесь считаются отдельные положительные пары, а не macro Recall. "
              "Пропуск источников означает отсутствие в объединении **выбранных** каналов.", "",
              "| Выборка | Найдено в 50 | Есть среди кандидатов, потеряно при отборе | Нет среди кандидатов |",
              "| --- | ---: | ---: | ---: |"]
    for name in runs:
        errors = json.loads((output / name / "errors_summary.json").read_text(encoding="utf-8"))
        lines.append(f"| {name} | {errors['retrieved']} | {errors['lost_at_selection']} | {errors['missed_by_sources']} |")
    lines += ["", "Реальные примеры dev:", "",
              "- «реконструкция волос» → «Кератин Ботокс Холодное восстановление»: "
              "BM25 со статистиками на позиции 128, символьный канал не нашёл в 500. "
              "Совпадение смысла слабо выражено общими словами.",
              "- «выравнивание полов» → «Полусухая стяжка пола с гарантией»: "
              "позиции 96 и 134 в двух текстовых каналах; потеря на выборе пятидесятки.",
              "- «парикмахер осветление» → «Окрашивание, стрижки, кератин»: "
              "BM25 на позиции 22, символьный канал не нашёл; RRF вытеснил полезного "
              "кандидата, поддержанного только одним источником.",
              "- «ремонт швейных машин» → «Ремонт и настройка швейного оборудования»: "
              "локация отличается, в исходном BM25 позиция 489, в выбранных каналах "
              "за пределами 500. Мягкий контекст тоже способен терять выездные услуги.", "",
              "Что уже сделано: сохранены полные описания и отдельные веса полей; "
              "добавлены символьные совпадения, мягкая география вместо отсекающего "
              "фильтра, история похожих запросов; веса истории проверены на dev. "
              "Все эти правила общие, без обработки конкретных query_id. "
              "Перечисленные примеры остаются ошибками текущей версии.", "",
              "Приоритет следующего этапа: проверить отбор из объединения кандидатов "
              "и потери кандидатов, найденных только одним каналом; затем добавить "
              "семантический канал и географическую совместимость для выездных услуг. "
              "Для обучаемого отбора нужны out-of-fold признаки. После новых подборов "
              "уже просмотренный test нельзя считать нетронутым финальным holdout.", "",
              "## Воспроизводимость и затраты", ""]
    for name in runs:
        timing = json.loads((output / name / "timing.json").read_text(encoding="utf-8"))
        lines.append(f"- {name}: {timing['search_seconds']:.1f} с, "
                     f"{timing['seconds_per_query']*1000:.1f} мс/запрос (все диагностические каналы).")
    manifest_path = output / "answer_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lines += [f"- Предсказание: {manifest['prediction_seconds']:.1f} с; "
                  f"{manifest['validation']['queries']} строк по {manifest['validation']['min_candidates']} кандидатов.",
                  f"- SHA256 `answer.csv`: `{manifest['answer_sha256']}`."]
    lines += ["", "Сборка индекса: около 63 с, 403 MiB на диске. Проверенная среда: "
              "Windows, Python 3.14, 12 логических CPU, 31.7 GiB RAM. "
              "Времена не включают установку зависимостей. Индекс и результаты "
              "кэшируются локально; модельных весов и внешних API нет.", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
