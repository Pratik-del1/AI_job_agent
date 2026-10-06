"""Compare the legacy and hybrid matchers on one frozen set of jobs.

Run from the project root::

    python -m jobagent.matching.evaluate snapshot            # freeze the live feed
    python -m jobagent.matching.evaluate template SNAPSHOT   # write eval/labels.csv to fill in
    python -m jobagent.matching.evaluate compare SNAPSHOT    # metrics, once labelled

Quality metrics need ``eval/labels.csv``. Without labels the report only
describes how the two rankings differ; it cannot say which is better.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd

from jobagent.config import BASE_DIR, Settings, get_settings
from jobagent.matching.pipeline import (
    HybridMatcher,
    build_embedder,
    load_profile,
    score_jobs,
)
from jobagent.matching.ranking import dedupe_jobs
from jobagent.matching.scoring import COMPONENTS

# Grades accepted in the label column, as words or numbers.
LABEL_GRADES = {
    "relevant": 2,
    "maybe": 1,
    "not_relevant": 0,
    "2": 2,
    "1": 1,
    "0": 0,
}

RELEVANT_GRADE = 2

LEGACY_COMPONENTS = {
    "semantic": "similarity_score",
    "skill": "skill_match",
    "role": "role_match",
    "experience": "experience_match",
}


# ------------------------------------------------------------
# Snapshots and labels
# ------------------------------------------------------------

def save_snapshot(jobs: list[dict], path: Path, source: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": source,
                "jobs": jobs,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def load_snapshot(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["jobs"]


def write_label_template(jobs: list[dict], path: Path) -> Path:
    """A CSV with one row per posting and an empty ``label`` column."""

    unique, _ = dedupe_jobs(jobs)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "title": job["title"],
                "location": job["location"],
                "label": "",
            }
            for job in unique
        ]
    ).to_csv(path, index=False)

    return path


def load_labels(path: Path) -> dict[str, int]:
    """job_id -> grade. Rows with an empty label are skipped."""

    path = Path(path)
    if not path.exists():
        return {}

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    labels = {}
    for _, row in frame.iterrows():
        value = row["label"].strip().lower().replace(" ", "_")
        if not value:
            continue
        if value not in LABEL_GRADES:
            raise ValueError(
                f"Unknown label {row['label']!r} for job {row['job_id']}. "
                f"Use one of: relevant, maybe, not_relevant"
            )
        labels[str(row["job_id"])] = LABEL_GRADES[value]

    return labels


# ------------------------------------------------------------
# Metrics. ``ranked`` is a list of job IDs, best first; unlabelled jobs
# are ignored.
# ------------------------------------------------------------

def _labelled(ranked: list[str], labels: dict[str, int]) -> list[int]:
    return [labels[job_id] for job_id in ranked if job_id in labels]


def precision_at_k(ranked, labels, k: int = 5) -> Optional[float]:
    grades = _labelled(ranked, labels)[:k]
    if not grades:
        return None
    return sum(grade >= RELEVANT_GRADE for grade in grades) / len(grades)


def ndcg_at_k(ranked, labels, k: int = 10) -> Optional[float]:
    grades = _labelled(ranked, labels)
    if not grades:
        return None

    def dcg(values):
        return sum(
            (2 ** grade - 1) / math.log2(position + 2)
            for position, grade in enumerate(values[:k])
        )

    ideal = dcg(sorted(grades, reverse=True))
    if ideal == 0:
        return None

    return dcg(grades) / ideal


def pairwise_accuracy(scores: dict[str, float], labels) -> Optional[float]:
    """Share of job pairs with different grades that the scores order the
    same way. A tied score counts as half."""

    correct = 0.0
    total = 0

    for a, b in combinations(sorted(labels), 2):
        if a not in scores or b not in scores or labels[a] == labels[b]:
            continue

        total += 1
        better, worse = (a, b) if labels[a] > labels[b] else (b, a)

        if scores[better] > scores[worse]:
            correct += 1
        elif scores[better] == scores[worse]:
            correct += 0.5

    return correct / total if total else None


def ranking_metrics(scores: dict[str, float], labels) -> dict:
    ranked = sorted(scores, key=lambda job_id: (-scores[job_id], job_id))

    return {
        "precision@5": precision_at_k(ranked, labels, 5),
        "ndcg@10": ndcg_at_k(ranked, labels, 10),
        "pairwise_accuracy": pairwise_accuracy(scores, labels),
    }


# ------------------------------------------------------------
# Scoring a snapshot with each matcher
# ------------------------------------------------------------

def score_hybrid(jobs, settings: Settings, embedder=None) -> pd.DataFrame:
    matcher = HybridMatcher(
        load_profile(settings),
        settings,
        embedder=embedder or build_embedder(settings),
    )
    return score_jobs(jobs, matcher)


def score_legacy(jobs) -> pd.DataFrame:
    """The legacy scorer from automation/job_updater.py, on the same
    de-duplicated jobs the hybrid matcher scores."""

    automation = str(BASE_DIR / "automation")
    if automation not in sys.path:
        sys.path.insert(0, automation)

    import job_updater

    unique, _ = dedupe_jobs(jobs)

    model = job_updater.SentenceTransformer(
        str(job_updater.find_model_path())
    )

    return job_updater.score_jobs_legacy(
        unique,
        job_updater.load_candidate_profile(),
        model,
    )


def _scores(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        str(job_id): float(value)
        for job_id, value in zip(frame["job_id"], frame[column])
        if pd.notna(value)
    }


def contribution_shares(hybrid: pd.DataFrame, settings: Settings) -> dict:
    """Average share of the overall score each component supplies."""

    weights = settings.match_weights
    totals = {name: 0.0 for name in COMPONENTS}
    counted = 0

    for _, row in hybrid.iterrows():
        available = {
            name: weights[name]
            for name in COMPONENTS
            if pd.notna(row[f"{name}_score"]) and weights[name] > 0
        }
        weight_sum = sum(available.values())
        if weight_sum <= 0 or row["overall_score"] <= 0:
            continue

        counted += 1
        for name, weight in available.items():
            totals[name] += (
                weight / weight_sum * row[f"{name}_score"]
            ) / row["overall_score"]

    return {
        name: (totals[name] / counted if counted else None)
        for name in COMPONENTS
    }


def compare(
    jobs: list[dict],
    labels: dict[str, int],
    settings: Settings,
    hybrid: Optional[pd.DataFrame] = None,
    legacy: Optional[pd.DataFrame] = None,
) -> dict:
    hybrid = score_hybrid(jobs, settings) if hybrid is None else hybrid
    legacy = score_legacy(jobs) if legacy is None else legacy

    hybrid_scores = _scores(hybrid, "overall_score")
    legacy_scores = _scores(legacy, "final_score")

    def top(scores, n):
        return sorted(scores, key=lambda job_id: (-scores[job_id], job_id))[:n]

    report = {
        "jobs": len(hybrid_scores),
        "labelled": sum(job_id in labels for job_id in hybrid_scores),
        "legacy": ranking_metrics(legacy_scores, labels),
        "hybrid": ranking_metrics(hybrid_scores, labels),
        "score_spread": {
            "legacy": float(pd.Series(legacy_scores).std(ddof=0)),
            "hybrid": float(pd.Series(hybrid_scores).std(ddof=0)),
        },
        "top10_overlap": len(
            set(top(legacy_scores, 10)) & set(top(hybrid_scores, 10))
        ),
        "contribution_share": contribution_shares(hybrid, settings),
        "components": {"hybrid": {}, "legacy": {}},
    }

    # Each component on its own, as if it were the whole ranking.
    for name in COMPONENTS:
        report["components"]["hybrid"][name] = ranking_metrics(
            _scores(hybrid, f"{name}_score"), labels
        )
    for name, column in LEGACY_COMPONENTS.items():
        report["components"]["legacy"][name] = ranking_metrics(
            _scores(legacy, column), labels
        )

    return report


def format_report(report: dict) -> str:
    def number(value):
        return "  n/a" if value is None else f"{value:5.2f}"

    def row(name, metrics):
        return (
            f"  {name:<12}"
            f" P@5 {number(metrics['precision@5'])}"
            f"  NDCG@10 {number(metrics['ndcg@10'])}"
            f"  pairwise {number(metrics['pairwise_accuracy'])}"
        )

    lines = [
        f"Jobs scored: {report['jobs']}   labelled: {report['labelled']}",
    ]

    if not report["labelled"]:
        lines.append(
            "No labels yet, so no quality metrics. Fill in eval/labels.csv "
            "to compare the matchers."
        )

    lines += [
        "",
        "Ranking quality (labelled jobs only)",
        row("legacy", report["legacy"]),
        row("hybrid", report["hybrid"]),
        "",
        "Score spread (standard deviation of the overall score)",
        f"  legacy {report['score_spread']['legacy']:.3f}"
        f"   hybrid {report['score_spread']['hybrid']:.3f}",
        f"Jobs in both top 10s: {report['top10_overlap']}",
        "",
        "Hybrid: average share of the overall score per component",
    ]
    lines += [
        f"  {name:<12} {number(share)}"
        for name, share in report["contribution_share"].items()
    ]

    for matcher in ("hybrid", "legacy"):
        lines += ["", f"Each {matcher} component as the only ranking signal"]
        lines += [
            row(name, metrics)
            for name, metrics in report["components"][matcher].items()
        ]

    return "\n".join(lines)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main(argv=None) -> int:
    settings = get_settings()
    snapshots = settings.eval_dir / "snapshots"
    labels_file = settings.eval_dir / "labels.csv"

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser(
        "snapshot", help="Fetch the live feed and freeze it."
    )
    snapshot.add_argument("--name", default=None)

    template = commands.add_parser(
        "template", help="Write a labels file to fill in."
    )
    template.add_argument("snapshot", type=Path)
    template.add_argument("--labels", type=Path, default=labels_file)

    comparison = commands.add_parser(
        "compare", help="Score a snapshot with both matchers."
    )
    comparison.add_argument("snapshot", type=Path)
    comparison.add_argument("--labels", type=Path, default=labels_file)

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        automation = str(BASE_DIR / "automation")
        if automation not in sys.path:
            sys.path.insert(0, automation)
        import job_updater

        name = args.name or f"jobs-{datetime.now():%Y-%m-%d}"
        path = save_snapshot(
            job_updater.fetch_lever_jobs(),
            snapshots / f"{name}.json",
            source=f"lever:{job_updater.COMPANY_SLUG}",
        )
        print(f"Saved: {path}")

    elif args.command == "template":
        if args.labels.exists():
            print(f"Not overwriting existing labels: {args.labels}")
            return 1
        path = write_label_template(load_snapshot(args.snapshot), args.labels)
        print(f"Saved: {path}")
        print("Fill the label column with: relevant, maybe or not_relevant")

    else:
        report = compare(
            load_snapshot(args.snapshot),
            load_labels(args.labels),
            settings,
        )
        print(format_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
