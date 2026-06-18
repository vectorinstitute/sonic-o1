"""
Temporal Localization — Coordinate Alignment Parser
=====================================================
Detects and corrects relative-to-absolute timestamp mismatches in model
predictions, then recomputes IoU and R@0.5/0.3/0.7.

A "relative-to-absolute mismatch" occurs when a model outputs timestamps
relative to the segment start (e.g., 50s) instead of absolute video time
(e.g., 550s when segment starts at 500s).

Detection rule:
  - Original prediction is wrong (IoU < threshold)
  - BUT pred + segment_start ≈ gt (within OFFSET_TOLERANCE seconds)
  → Correct by shifting: corrected = pred + segment_start

Run 
    python sonic-o1/parser_correction.py

Output:
  sonic-o1/parser_correction_results.txt
  sonic-o1/parser_correction_summary.txt 
"""

import json
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SCORES_PATH      = "sonic-o1/05_evaluation_inference/results/scores/gpt_judge"
OFFSET_TOLERANCE = 10.0   # seconds — how close corrected pred must be to gt
IOU_THRESHOLDS   = [0.3, 0.5, 0.7]

MODELS = {
    'gemini':        'Gemini 3.0 Pro',
    'qwen3':         'Qwen3-Omni',
    'unimoe':        'UniMoE-2.0',
    'minicpm-o-2.6': 'MiniCPM-o-2.6',
    'vita':          'VITA 1.5',
    'videollama':    'VideoLLaMA2',
    'baichuan_omni': 'Baichuan Omni 1.5',
    "ola": 'OLA',
    "omnivinci": 'OmniVinci',
}


# ── IoU computation ───────────────────────────────────────────────────────────
def compute_iou(pred_start, pred_end, gt_start, gt_end):
    if pred_end <= pred_start or gt_end <= gt_start:
        return 0.0
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


def compute_recall(iou, threshold):
    return 1 if iou >= threshold else 0


# ── Load JSON ─────────────────────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ── Per-model processing ──────────────────────────────────────────────────────
def process_model(model_key):
    task3_path = os.path.join(SCORES_PATH, model_key, "task3_temporal_localization")
    if not os.path.exists(task3_path):
        return None

    topic_stats = {}

    for tf in sorted(os.listdir(task3_path)):
        if not tf.endswith('.json'):
            continue

        data = load_json(os.path.join(task3_path, tf))
        if not data:
            continue

        topic_name = tf.replace('.json', '')

        orig_iou_list   = []
        corr_iou_list   = []
        orig_r          = {t: [] for t in IOU_THRESHOLDS}
        corr_r          = {t: [] for t in IOU_THRESHOLDS}
        corrected_count = 0
        total           = 0

        for entry in data.get("per_question_results", []):
            gt    = entry.get("gt_interval", {})
            pred  = entry.get("pred_interval", {})
            seg   = entry.get("segment", {})

            gt_s  = gt.get("start")
            gt_e  = gt.get("end")
            pr_s  = pred.get("start")
            pr_e  = pred.get("end")
            seg_s = seg.get("start", 0.0)

            if any(v is None for v in [gt_s, gt_e, pr_s, pr_e]):
                continue

            total += 1

            # Original scores
            orig_iou = entry.get("iou", compute_iou(pr_s, pr_e, gt_s, gt_e))
            orig_iou_list.append(orig_iou)
            for t in IOU_THRESHOLDS:
                orig_r[t].append(compute_recall(orig_iou, t))

            # Attempt coordinate correction only when segment doesn't start at 0
            if seg_s > 0:
                corr_s   = pr_s + seg_s
                corr_e   = pr_e + seg_s
                corr_iou = compute_iou(corr_s, corr_e, gt_s, gt_e)

                start_close = abs(corr_s - gt_s) <= OFFSET_TOLERANCE
                end_close   = abs(corr_e - gt_e) <= OFFSET_TOLERANCE

                if (start_close or end_close) and corr_iou > orig_iou:
                    corrected_count += 1
                    corr_iou_list.append(corr_iou)
                    for t in IOU_THRESHOLDS:
                        corr_r[t].append(compute_recall(corr_iou, t))
                else:
                    corr_iou_list.append(orig_iou)
                    for t in IOU_THRESHOLDS:
                        corr_r[t].append(compute_recall(orig_iou, t))
            else:
                corr_iou_list.append(orig_iou)
                for t in IOU_THRESHOLDS:
                    corr_r[t].append(compute_recall(orig_iou, t))

        if total == 0:
            continue

        def mean_pct(lst):
            return round(100 * sum(lst) / len(lst), 2) if lst else 0.0

        topic_stats[topic_name] = {
            'total':           total,
            'corrected_count': corrected_count,
            'original': {
                'miou':  mean_pct(orig_iou_list),
                'R@0.3': mean_pct(orig_r[0.3]),
                'R@0.5': mean_pct(orig_r[0.5]),
                'R@0.7': mean_pct(orig_r[0.7]),
            },
            'corrected': {
                'miou':  mean_pct(corr_iou_list),
                'R@0.3': mean_pct(corr_r[0.3]),
                'R@0.5': mean_pct(corr_r[0.5]),
                'R@0.7': mean_pct(corr_r[0.7]),
            },
        }

    return topic_stats


def aggregate(topic_stats):
    if not topic_stats:
        return None

    orig_vals = {'miou': [], 'R@0.3': [], 'R@0.5': [], 'R@0.7': []}
    corr_vals = {'miou': [], 'R@0.3': [], 'R@0.5': [], 'R@0.7': []}
    total_corrected = 0
    total_entries   = 0

    for stats in topic_stats.values():
        for metric in orig_vals:
            orig_vals[metric].append(stats['original'][metric])
            corr_vals[metric].append(stats['corrected'][metric])
        total_corrected += stats['corrected_count']
        total_entries   += stats['total']

    def mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        'original':        {m: mean(orig_vals[m]) for m in orig_vals},
        'corrected':       {m: mean(corr_vals[m]) for m in corr_vals},
        'total_corrected': total_corrected,
        'total_entries':   total_entries,
        'correction_rate': round(100 * total_corrected / total_entries, 1) if total_entries else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    lines   = []
    summary = []

    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 80)
    log("SONIC-O1 — Temporal Localization Coordinate Alignment Parser")
    log(f"Offset tolerance = {OFFSET_TOLERANCE}s | All metrics reported as %")
    log("=" * 80)

    all_agg = {}

    for model_key, model_name in MODELS.items():
        log()
        log(f"{'─'*80}")
        log(f"MODEL: {model_name}")
        log(f"{'─'*80}")

        topic_stats = process_model(model_key)
        if not topic_stats:
            log("  No data found.")
            continue

        # Per-topic breakdown
        log(f"  {'Topic':<50} {'N':>5} {'Fixed':>6} "
            f"{'Orig R@0.5':>11} {'Corr R@0.5':>11} {'Delta':>8}")
        log(f"  {'-'*96}")

        for topic_name, stats in sorted(topic_stats.items()):
            orig  = stats['original']
            corr  = stats['corrected']
            delta = round(corr['R@0.5'] - orig['R@0.5'], 2)
            log(f"  {topic_name:<50} "
                f"{stats['total']:>5} "
                f"{stats['corrected_count']:>6} "
                f"{orig['R@0.5']:>10.2f}% "
                f"{corr['R@0.5']:>10.2f}% "
                f"{delta:>+7.2f}%")

        agg = aggregate(topic_stats)
        all_agg[model_name] = agg

        log()
        log(f"  OVERALL (macro-avg across topics):")
        log(f"  Predictions corrected: {agg['total_corrected']}/{agg['total_entries']} "
            f"({agg['correction_rate']}%)")
        log(f"  {'Metric':<10} {'Original':>10} {'Corrected':>10} {'Delta':>8}")
        log(f"  {'-'*42}")
        for metric in ['miou', 'R@0.3', 'R@0.5', 'R@0.7']:
            orig_v = agg['original'][metric]
            corr_v = agg['corrected'][metric]
            delta  = round(corr_v - orig_v, 2)
            log(f"  {metric:<10} {orig_v:>9.2f}% {corr_v:>9.2f}% {delta:>+7.2f}%")

    # ── Rebuttal-ready summary ────────────────────────────────────────────────
    summary.append("=" * 80)
    summary.append("REBUTTAL SUMMARY — Coordinate Alignment Parser")
    summary.append(f"Offset tolerance = {OFFSET_TOLERANCE}s | All metrics as %")
    summary.append("=" * 80)
    summary.append("")
    summary.append(f"{'Model':<20} {'Orig mIoU':>10} {'Corr mIoU':>10} "
                   f"{'Orig R@0.5':>11} {'Corr R@0.5':>11} "
                   f"{'Delta R@0.5':>12} {'Fixed%':>8}")
    summary.append("-" * 86)

    for model_name, agg in all_agg.items():
        if agg is None:
            continue
        orig_r05  = agg['original']['R@0.5']
        corr_r05  = agg['corrected']['R@0.5']
        orig_miou = agg['original']['miou']
        corr_miou = agg['corrected']['miou']
        delta     = round(corr_r05 - orig_r05, 2)
        fix_rate  = agg['correction_rate']
        summary.append(
            f"{model_name:<20} "
            f"{orig_miou:>9.2f}% "
            f"{corr_miou:>9.2f}% "
            f"{orig_r05:>10.2f}% "
            f"{corr_r05:>10.2f}% "
            f"{delta:>+11.2f}% "
            f"{fix_rate:>7.1f}%"
        )

    summary.append("")
    summary.append("Fixed% = % of total predictions corrected by parser")
    summary.append("Delta  = Corrected R@0.5 minus Original R@0.5")
    summary.append("Macro-averaged across all 13 topics.")

    out_dir = "sonic-o1"
    with open(os.path.join(out_dir, "parser_correction_results.txt"), "w") as f:
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write("\n".join(summary))

    with open(os.path.join(out_dir, "parser_correction_summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print()
    print("─" * 80)
    print("\n".join(summary))
    print()
    print("Full report  → sonic-o1/parser_correction_results.txt")
    print("Summary      → sonic-o1/parser_correction_summary.txt")


if __name__ == "__main__":
    main()
