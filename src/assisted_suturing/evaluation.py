"""
evaluation.py

Evaluation der bestehenden Nadel- und Fadenerkennung.

Die Detection-Algorithmen in needle_thread_detection.py werden nicht verändert.
Diese Datei übernimmt ausschließlich:
- Frame-Sampling
- manuelle Ground-Truth-Erstellung
- Vergleich Prediction vs. Ground Truth
- TP, FP, FN, TN
- Precision, Recall, F1-Score, Accuracy
- Processing Time
- CSV- und JSON-Ausgabe

Ground-Truth-Tasten:
    0 = weder Nadel noch Faden
    n = nur Nadel
    f = nur Faden
    b = Nadel und Faden
    s = Frame überspringen
    q = speichern und beenden
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

import needle_thread_detection as detection


# =============================================================================
# KONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = PROJECT_ROOT / "data" / "phantom_videos"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.csv"
FRAME_RESULTS_FILE = OUTPUT_DIR / "frame_results.csv"
VIDEO_METRICS_FILE = OUTPUT_DIR / "video_metrics.csv"
SUMMARY_JSON_FILE = OUTPUT_DIR / "evaluation_summary.json"

# Bisherige Sampling-Einstellung: jeder 39. Frame.
# Bei 12.93 FPS entspricht das etwa einem Frame alle 3 Sekunden.
SAMPLE_EVERY_N_FRAMES = 39

VIDEO_NAMES = [
    "01_needle_normal_light.mp4",
    "02_thread_different_angles.mp4",
    "03_needle_different_shapes.mp4",
    "04_needle_and_thread.mp4",
    "05_shadow_reflection.mp4",
    "06_shadow_lowlight_reflection.mp4",
]

# ROI exakt entsprechend process_frame() in needle_thread_detection.py.
ROI_Y_MIN = 0.14
ROI_Y_MAX = 0.98
ROI_X_MIN = 0.26
ROI_X_MAX = 0.96

# Feste Parameter für reproduzierbare Evaluation.
DETECTION_PARAMETERS = {
    "Faden Schwelle": 8,
    "Min Faden": 20,
    "Canny niedrig": 40,
    "Canny hoch": 140,
    "Dunkle Nadel": 6,
}

GT_FIELDS = ["video", "frame_idx", "needle_gt", "thread_gt", "skip"]


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_video_paths() -> list[Path]:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Videoordner wurde nicht gefunden:\n{VIDEO_DIR}")

    paths: list[Path] = []

    for name in VIDEO_NAMES:
        path = VIDEO_DIR / name
        if path.exists():
            paths.append(path)
        else:
            print(f"[WARNUNG] Video nicht gefunden: {name}")

    if paths:
        return paths

    fallback = sorted(VIDEO_DIR.glob("*.mp4"))
    if not fallback:
        raise FileNotFoundError(f"Keine MP4-Dateien gefunden in:\n{VIDEO_DIR}")

    print("[INFO] Alle MP4-Dateien werden verwendet.")
    return fallback


def resize_for_display(frame: np.ndarray, max_width: int = 1280, max_height: int = 800) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return frame

    return cv.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv.INTER_AREA,
    )


def _write_csv_once(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """CSV atomar speichern, damit keine teilweise geschriebene Datei entsteht."""
    temp_path = path.with_name(f".{path.stem}_temp_{os.getpid()}{path.suffix}")

    try:
        with temp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def write_csv_with_retry(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    retries: int = 10,
    wait_seconds: float = 0.5,
) -> None:
    last_error: PermissionError | None = None

    for attempt in range(1, retries + 1):
        try:
            _write_csv_once(path, rows, fieldnames)
            return
        except PermissionError as error:
            last_error = error
            print(
                f"[WARNUNG] {path.name} ist gesperrt. "
                f"Versuch {attempt}/{retries} ..."
            )
            time.sleep(wait_seconds)

    raise PermissionError(
        f"{path} konnte nicht gespeichert werden. Bitte Excel schließen."
    ) from last_error


def write_json_with_retry(path: Path, data: dict[str, Any]) -> None:
    for attempt in range(1, 11):
        try:
            with path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            return
        except PermissionError:
            print(f"[WARNUNG] {path.name} ist gesperrt. Versuch {attempt}/10 ...")
            time.sleep(0.5)

    raise PermissionError(f"{path} konnte nicht gespeichert werden.")


# =============================================================================
# GROUND TRUTH
# =============================================================================

def ground_truth_rows(
    ground_truth: dict[tuple[str, int], dict[str, int]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "video": video_name,
            "frame_idx": frame_idx,
            "needle_gt": label["needle_gt"],
            "thread_gt": label["thread_gt"],
            "skip": label.get("skip", 0),
        }
        for (video_name, frame_idx), label in ground_truth.items()
    ]

    rows.sort(key=lambda row: (row["video"], int(row["frame_idx"])))
    return rows


def save_ground_truth(
    ground_truth: dict[tuple[str, int], dict[str, int]],
) -> Path:
    """Ground Truth speichern; bei Dateisperre automatisch Recovery-Datei anlegen."""
    ensure_output_dir()
    rows = ground_truth_rows(ground_truth)

    try:
        write_csv_with_retry(GROUND_TRUTH_FILE, rows, GT_FIELDS)

        # Nach erfolgreichem Speichern alte Recovery-Dateien entfernen.
        for recovery in OUTPUT_DIR.glob("ground_truth_recovery_*.csv"):
            try:
                recovery.unlink()
            except OSError:
                pass

        return GROUND_TRUTH_FILE

    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recovery = OUTPUT_DIR / f"ground_truth_recovery_{timestamp}.csv"
        _write_csv_once(recovery, rows, GT_FIELDS)

        print("\n[WARNUNG] ground_truth.csv war gesperrt.")
        print(f"[INFO] Ground Truth wurde sicher gespeichert in:\n{recovery}")
        print("[INFO] Excel schließen; Recovery wird beim nächsten Start mit eingelesen.")
        return recovery


def _load_gt_file(
    path: Path,
    ground_truth: dict[tuple[str, int], dict[str, int]],
) -> None:
    if not path.exists():
        return

    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if not row.get("video"):
                    continue

                key = (row["video"], int(row["frame_idx"]))
                ground_truth[key] = {
                    "needle_gt": int(row["needle_gt"]),
                    "thread_gt": int(row["thread_gt"]),
                    "skip": int(row.get("skip", 0)),
                }

    except PermissionError:
        print(f"[WARNUNG] {path.name} konnte nicht gelesen werden. Bitte Excel schließen.")


def load_ground_truth() -> dict[tuple[str, int], dict[str, int]]:
    ground_truth: dict[tuple[str, int], dict[str, int]] = {}

    _load_gt_file(GROUND_TRUTH_FILE, ground_truth)

    recovery_files = sorted(
        OUTPUT_DIR.glob("ground_truth_recovery_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )

    for recovery in recovery_files:
        _load_gt_file(recovery, ground_truth)

    return ground_truth

def annotate_missing_ground_truth(
    video_paths: list[Path],
    ground_truth: dict[tuple[str, int], dict[str, int]],
) -> bool:

    print("\n" + "=" * 70)
    print("GROUND-TRUTH-ERSTELLUNG")
    print("=" * 70)
    print(
        "0=nichts | n=Nadel | f=Faden | "
        "b=beide | s=skip | q=speichern und beenden"
    )

    # Ground-Truth-Fenster erzeugen
    cv.namedWindow("Ground-Truth", cv.WINDOW_NORMAL)
    cv.resizeWindow("Ground-Truth", 1200, 750)
    cv.moveWindow("Ground-Truth", 50, 50)

    try:
        for video_path in video_paths:
            video = cv.VideoCapture(str(video_path))

            if not video.isOpened():
                print(
                    f"[FEHLER] Video konnte nicht geöffnet werden: "
                    f"{video_path}"
                )
                continue

            fps = float(video.get(cv.CAP_PROP_FPS))
            total_frames = int(
                video.get(cv.CAP_PROP_FRAME_COUNT)
            )

            sample_count = (
                ((total_frames - 1) // SAMPLE_EVERY_N_FRAMES) + 1
                if total_frames > 0
                else 0
            )

            print(f"\nVideo: {video_path.name}")
            print(f"FPS: {fps:.2f}")
            print(f"Frames: {total_frames}")
            print(
                f"Jeder {SAMPLE_EVERY_N_FRAMES}. Frame wird bewertet."
            )
            print(
                f"Geplante Stichprobe: ca. {sample_count} Frames"
            )

            frame_idx = 0

            try:
                while True:

                    success, frame = video.read()

                    if not success:
                        break

                    if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
                        frame_idx += 1
                        continue

                    gt_key = (
                        video_path.name,
                        frame_idx,
                    )

                    # Bereits vorhandene Ground-Truth-Daten überspringen
                    if gt_key in ground_truth:
                        frame_idx += 1
                        continue

                    timestamp_seconds = (
                        frame_idx / fps
                        if fps > 0
                        else 0.0
                    )

                    display = frame.copy()

                    cv.putText(
                        display,
                        (
                            f"{video_path.name} | "
                            f"Frame {frame_idx} | "
                            f"{timestamp_seconds:.1f} s"
                        ),
                        (20, 35),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv.LINE_AA,
                    )

                    cv.putText(
                        display,
                        (
                            "0:nichts | n:Nadel | f:Faden | "
                            "b:beide | s:skip | q:Ende"
                        ),
                        (20, 70),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (255, 255, 255),
                        2,
                        cv.LINE_AA,
                    )

                    # Frame anzeigen
                    cv.imshow(
                        "Ground-Truth",
                        resize_for_display(display),
                    )

                    # Fenster wirklich darstellen
                    cv.waitKey(50)

                    label: dict[str, int] | None = None

                    while label is None:

                        key = cv.waitKey(0) & 0xFF

                        if key == ord("0"):
                            label = {
                                "needle_gt": 0,
                                "thread_gt": 0,
                                "skip": 0,
                            }

                        elif key in (ord("n"), ord("N")):
                            label = {
                                "needle_gt": 1,
                                "thread_gt": 0,
                                "skip": 0,
                            }

                        elif key in (ord("f"), ord("F")):
                            label = {
                                "needle_gt": 0,
                                "thread_gt": 1,
                                "skip": 0,
                            }

                        elif key in (ord("b"), ord("B")):
                            label = {
                                "needle_gt": 1,
                                "thread_gt": 1,
                                "skip": 0,
                            }

                        elif key in (ord("s"), ord("S")):
                            label = {
                                "needle_gt": 0,
                                "thread_gt": 0,
                                "skip": 1,
                            }

                        elif key in (ord("q"), ord("Q")):

                            save_ground_truth(
                                ground_truth
                            )

                            print(
                                "\n[INFO] Ground Truth gespeichert. "
                                "Programm beendet."
                            )

                            return False

                    ground_truth[gt_key] = label

                    save_ground_truth(
                        ground_truth
                    )

                    print(
                        f"Frame {frame_idx}: "
                        f"needle={label['needle_gt']}, "
                        f"thread={label['thread_gt']}, "
                        f"skip={label['skip']}"
                    )

                    frame_idx += 1

            finally:
                video.release()

    finally:
        try:
            cv.destroyWindow("Ground-Truth")
        except cv.error:
            pass

    return True


# =============================================================================
# BESTEHENDE DETECTION AUSFÜHREN
# =============================================================================

def initialise_detection() -> tuple[np.ndarray, np.ndarray]:
    """Kalibrierung und bestehende Trackbars initialisieren."""
    camera_matrix, distortion = detection.load_calibration()
    detection.create_controls()

    for name, value in DETECTION_PARAMETERS.items():
        try:
            cv.setTrackbarPos(name, detection.WINDOW, int(value))
        except cv.error:
            print(f"[WARNUNG] Trackbar '{name}' konnte nicht gesetzt werden.")

    return camera_matrix, distortion


def run_existing_detection(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[bool, bool, float]:
    """
    Exakt dieselbe Detection-Pipeline wie process_frame(), jedoch ohne Zeichnen.

    Rückgabe:
        needle_detected, thread_detected, processing_time_ms
    """
    start = time.perf_counter()

    # Entzerrung.
    image = cv.undistort(frame, camera_matrix, distortion)

    # Graustufenbild und Vorverarbeitung.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    height, width = gray.shape

    # ROI exakt wie im aktuellen process_frame().
    roi = np.zeros_like(gray)
    roi[
        int(height * ROI_Y_MIN):int(height * ROI_Y_MAX),
        int(width * ROI_X_MIN):int(width * ROI_X_MAX),
    ] = 255

    # Bestehende Fadenerkennung unverändert aufrufen.
    _, thread_contours = detection.detect_thread(gray, roi)

    # Bestehende Nadelerkennung unverändert aufrufen.
    needle, _ = detection.detect_needle(image, roi, thread_contours)

    processing_time_ms = (time.perf_counter() - start) * 1000.0

    thread_detected = len(thread_contours) > 0
    needle_detected = needle is not None

    return needle_detected, thread_detected, processing_time_ms


# =============================================================================
# METRIKEN
# =============================================================================

def empty_confusion() -> dict[str, int]:
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


def update_confusion(confusion: dict[str, int], gt: int, pred: int) -> str:
    if gt == 1 and pred == 1:
        confusion["TP"] += 1
        return "TP"
    if gt == 0 and pred == 1:
        confusion["FP"] += 1
        return "FP"
    if gt == 1 and pred == 0:
        confusion["FN"] += 1
        return "FN"

    confusion["TN"] += 1
    return "TN"


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(confusion: dict[str, int]) -> dict[str, float]:
    tp = confusion["TP"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    tn = confusion["TN"]

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    video_paths: list[Path],
    ground_truth: dict[tuple[str, int], dict[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    camera_matrix, distortion = initialise_detection()

    frame_results: list[dict[str, Any]] = []
    confusion_by_video = defaultdict(
        lambda: {
            "needle": empty_confusion(),
            "thread": empty_confusion(),
        }
    )
    processing_times_by_video: dict[str, list[float]] = defaultdict(list)

    try:
        for video_path in video_paths:
            video = cv.VideoCapture(str(video_path))

            if not video.isOpened():
                print(f"[FEHLER] Video konnte nicht geöffnet werden: {video_path}")
                continue

            fps = float(video.get(cv.CAP_PROP_FPS))
            frame_idx = 0
            evaluated = 0
            skipped = 0
            missing = 0

            print(f"\nVideo: {video_path.name}")
            print(f"FPS: {fps:.2f}")

            try:
                while True:
                    success, frame = video.read()
                    if not success:
                        break

                    if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
                        frame_idx += 1
                        continue

                    gt = ground_truth.get((video_path.name, frame_idx))

                    if gt is None:
                        missing += 1
                        frame_idx += 1
                        continue

                    if int(gt.get("skip", 0)) == 1:
                        skipped += 1
                        frame_idx += 1
                        continue

                    needle_detected, thread_detected, processing_time_ms = run_existing_detection(
                        frame,
                        camera_matrix,
                        distortion,
                    )

                    needle_gt = int(gt["needle_gt"])
                    thread_gt = int(gt["thread_gt"])
                    needle_pred = int(needle_detected)
                    thread_pred = int(thread_detected)

                    needle_result = update_confusion(
                        confusion_by_video[video_path.name]["needle"],
                        needle_gt,
                        needle_pred,
                    )

                    thread_result = update_confusion(
                        confusion_by_video[video_path.name]["thread"],
                        thread_gt,
                        thread_pred,
                    )

                    processing_times_by_video[video_path.name].append(processing_time_ms)

                    frame_results.append({
                        "video": video_path.name,
                        "frame_idx": frame_idx,
                        "timestamp_s": round(frame_idx / fps if fps > 0 else 0.0, 6),
                        "needle_gt": needle_gt,
                        "needle_pred": needle_pred,
                        "needle_result": needle_result,
                        "thread_gt": thread_gt,
                        "thread_pred": thread_pred,
                        "thread_result": thread_result,
                        "processing_time_ms": round(processing_time_ms, 6),
                    })

                    evaluated += 1
                    frame_idx += 1

            finally:
                video.release()

            print(f"Ausgewertete Frames: {evaluated}")
            if skipped:
                print(f"Manuell übersprungen: {skipped}")
            if missing:
                print(f"Ohne Ground Truth: {missing}")

    finally:
        try:
            cv.destroyWindow(detection.WINDOW)
        except cv.error:
            pass

    # Ergebnisse pro Video und Objekt.
    video_metrics: list[dict[str, Any]] = []

    for video_name in sorted(confusion_by_video.keys()):
        times = processing_times_by_video[video_name]
        mean_time = float(np.mean(times)) if times else 0.0
        std_time = float(np.std(times)) if times else 0.0

        for object_name in ("needle", "thread"):
            confusion = confusion_by_video[video_name][object_name]
            metrics = calculate_metrics(confusion)

            video_metrics.append({
                "video": video_name,
                "object": object_name,
                "samples": sum(confusion.values()),
                "positive_gt": confusion["TP"] + confusion["FN"],
                "negative_gt": confusion["TN"] + confusion["FP"],
                "TP": confusion["TP"],
                "FP": confusion["FP"],
                "FN": confusion["FN"],
                "TN": confusion["TN"],
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1_score": round(metrics["f1_score"], 6),
                "accuracy": round(metrics["accuracy"], 6),
                "mean_processing_time_ms": round(mean_time, 6),
                "std_processing_time_ms": round(std_time, 6),
            })

    # Gesamtmetriken.
    overall_confusion = {
        "needle": empty_confusion(),
        "thread": empty_confusion(),
    }

    for video_data in confusion_by_video.values():
        for object_name in ("needle", "thread"):
            for key in ("TP", "FP", "FN", "TN"):
                overall_confusion[object_name][key] += video_data[object_name][key]

    all_times = [float(row["processing_time_ms"]) for row in frame_results]

    mean_processing = float(np.mean(all_times)) if all_times else 0.0
    std_processing = float(np.std(all_times)) if all_times else 0.0
    min_processing = float(np.min(all_times)) if all_times else 0.0
    max_processing = float(np.max(all_times)) if all_times else 0.0

    summary: dict[str, Any] = {
        "configuration": {
            "sample_every_n_frames": SAMPLE_EVERY_N_FRAMES,
            "roi": {
                "x_min": ROI_X_MIN,
                "x_max": ROI_X_MAX,
                "y_min": ROI_Y_MIN,
                "y_max": ROI_Y_MAX,
            },
            "detection_parameters": DETECTION_PARAMETERS,
            "video_directory": str(VIDEO_DIR),
        },
        "total_evaluated_frames": len(frame_results),
        "processing_time": {
            "mean_ms": round(mean_processing, 6),
            "std_ms": round(std_processing, 6),
            "min_ms": round(min_processing, 6),
            "max_ms": round(max_processing, 6),
            "theoretical_fps": round(safe_divide(1000.0, mean_processing), 6),
        },
        "overall": {},
    }

    for object_name in ("needle", "thread"):
        confusion = overall_confusion[object_name]
        metrics = calculate_metrics(confusion)
        summary["overall"][object_name] = {
            **confusion,
            **{key: round(value, 6) for key, value in metrics.items()},
        }

    return frame_results, video_metrics, summary


# =============================================================================
# SPEICHERN UND AUSGABE
# =============================================================================

def save_results(
    frame_results: list[dict[str, Any]],
    video_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    frame_fields = [
        "video", "frame_idx", "timestamp_s",
        "needle_gt", "needle_pred", "needle_result",
        "thread_gt", "thread_pred", "thread_result",
        "processing_time_ms",
    ]

    metric_fields = [
        "video", "object", "samples", "positive_gt", "negative_gt",
        "TP", "FP", "FN", "TN",
        "precision", "recall", "f1_score", "accuracy",
        "mean_processing_time_ms", "std_processing_time_ms",
    ]

    write_csv_with_retry(FRAME_RESULTS_FILE, frame_results, frame_fields)
    write_csv_with_retry(VIDEO_METRICS_FILE, video_metrics, metric_fields)
    write_json_with_retry(SUMMARY_JSON_FILE, summary)


def print_results(video_metrics: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("\n" + "=" * 112)
    print("ERGEBNISSE PRO VIDEO")
    print("=" * 112)

    header = (
        f"{'Video':34} {'Objekt':8} {'N':>5} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
        f"{'Prec.':>8} {'Recall':>8} {'F1':>8} {'Acc.':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in video_metrics:
        print(
            f"{row['video'][:34]:34} {row['object']:8} {row['samples']:5d} "
            f"{row['TP']:4d} {row['FP']:4d} {row['FN']:4d} {row['TN']:4d} "
            f"{row['precision']:8.3f} {row['recall']:8.3f} "
            f"{row['f1_score']:8.3f} {row['accuracy']:8.3f}"
        )

    print("\n" + "=" * 70)
    print("GESAMTERGEBNIS")
    print("=" * 70)
    print(f"Ausgewertete Frames: {summary['total_evaluated_frames']}")

    for object_name in ("needle", "thread"):
        result = summary["overall"][object_name]
        print(f"\n{object_name.upper()}")
        print(f"TP={result['TP']} | FP={result['FP']} | FN={result['FN']} | TN={result['TN']}")
        print(f"Precision = {result['precision']:.3f}")
        print(f"Recall    = {result['recall']:.3f}")
        print(f"F1-Score  = {result['f1_score']:.3f}")
        print(f"Accuracy  = {result['accuracy']:.3f}")

    processing = summary["processing_time"]
    print("\nPROCESSING TIME")
    print(f"Mittelwert: {processing['mean_ms']:.3f} ms")
    print(f"Standardabweichung: {processing['std_ms']:.3f} ms")
    print(f"Minimum: {processing['min_ms']:.3f} ms")
    print(f"Maximum: {processing['max_ms']:.3f} ms")
    print(f"Theoretische Verarbeitungsrate: {processing['theoretical_fps']:.2f} FPS")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_output_dir()

    print("=" * 70)
    print("EVALUATION DER NADEL- UND FADENERKENNUNG")
    print("=" * 70)

    video_paths = get_video_paths()

    print("\nVerwendete Videos:")
    for path in video_paths:
        print(f"  - {path.name}")

    ground_truth = load_ground_truth()
    print(f"\nVorhandene Ground-Truth-Einträge: {len(ground_truth)}")

    # 1. Fehlende Ground-Truth-Daten ergänzen.
    completed = annotate_missing_ground_truth(video_paths, ground_truth)
    if not completed:
        return

    save_ground_truth(ground_truth)

    # 2. Bestehende Detection objektiv evaluieren.
    frame_results, video_metrics, summary = evaluate(video_paths, ground_truth)

    # 3. Ergebnisse speichern und anzeigen.
    save_results(frame_results, video_metrics, summary)
    print_results(video_metrics, summary)

    print("\n" + "=" * 70)
    print("DATEIEN GESPEICHERT")
    print("=" * 70)
    print(f"Ground Truth:       {GROUND_TRUTH_FILE}")
    print(f"Frame-Ergebnisse:   {FRAME_RESULTS_FILE}")
    print(f"Video-Metriken:     {VIDEO_METRICS_FILE}")
    print(f"JSON-Zusammenfassung: {SUMMARY_JSON_FILE}")


if __name__ == "__main__":
    main()