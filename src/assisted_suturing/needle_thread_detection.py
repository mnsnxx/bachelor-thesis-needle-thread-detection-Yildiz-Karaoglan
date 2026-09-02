"""Erkennung einer chirurgischen Nadel und eines Fadens."""

from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "phantom_videos"
    / "05_shadow_reflection.mp4"
)

CALIBRATION_DIR = PROJECT_ROOT / "calibration"
WINDOW = "Parameter"


def load_calibration() -> tuple[np.ndarray, np.ndarray]:
    """Das neueste validierte Kalibrierungspaket laden."""

    files = list(
        CALIBRATION_DIR.glob(
            "calibration_intrinsic-*.json"
        )
    )

    if not files:
        raise RuntimeError(
            "Keine Kalibrierung gefunden."
        )

    path = max(
        files,
        key=lambda file: file.stat().st_mtime, #Schau dir die Dateien in der Liste an, prüfe bei jeder Datei den Zeitpunkt der letzten Änderung und wähle die zuletzt geänderte Datei aus.
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if data["validation"]["status"] != "PASS":
        raise RuntimeError(
            "Die Kamerakalibrierung ist nicht gültig."
        )

    print(f"Kalibrierung: {path}")

    return (
        np.asarray(
            data["camera"]["K"],
            dtype=np.float64,
        ),
        np.asarray(
            data["camera"]["distortion"],
            dtype=np.float64,
        ),
    )


def create_controls() -> None:
    """Parameter für die Erkennung erzeugen."""

    cv.namedWindow(
        WINDOW,
        cv.WINDOW_NORMAL,
    )

    callback = lambda _: None

    cv.createTrackbar(
        "Faden Schwelle",
        WINDOW,
        8,
        50,
        callback,
    )

    cv.createTrackbar(
        "Min Faden",
        WINDOW,
        20,
        500,
        callback,
    )

    cv.createTrackbar(
        "Canny niedrig",
        WINDOW,
        40,
        255,
        callback,
    )

    cv.createTrackbar(
        "Canny hoch",
        WINDOW,
        140,
        255,
        callback,
    )

    cv.createTrackbar(
        "Top-Hat Schwelle",
        WINDOW,
        15,
        255,
        callback,
    )

    cv.createTrackbar(
        "Min Umfang",
        WINDOW,
        20,
        2000,
        callback,
    )

    cv.createTrackbar(
        "Min Helligkeit",
        WINDOW,
        110,
        255,
        callback,
    )

    cv.createTrackbar(
        "Max Saettigung",
        WINDOW,
        100,
        255,
        callback,
    )
    cv.createTrackbar(
    "Dunkle Nadel",
    WINDOW,
    6,
    50,
    callback,
)

def detect_thread(
    gray: np.ndarray,
    roi: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Dünne dunkle Fadenstrukturen erkennen."""

    threshold = cv.getTrackbarPos(
        "Faden Schwelle",
        WINDOW,
    )

    min_thread = cv.getTrackbarPos(
        "Min Faden",
        WINDOW,
    )

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (15, 15),
    )

    # Dünne dunkle Strukturen hervorheben.
    black_hat = cv.morphologyEx(
        gray,
        cv.MORPH_BLACKHAT,
        kernel,
    )

    _, thread_mask = cv.threshold(
        black_hat,
        threshold,
        255,
        cv.THRESH_BINARY,
    )

    thread_mask = cv.bitwise_and(
        thread_mask,
        roi,
    )

    thread_mask = cv.morphologyEx(
        thread_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    contours, _ = cv.findContours(
        thread_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE,
    )

    thread_contours = []

    for contour in contours:
        perimeter = cv.arcLength(
            contour,
            False,
        )

        area = cv.contourArea(
            contour
        )

        x, y, w, h = cv.boundingRect(
            contour
        )

        if perimeter < min_thread or area < 1:
            continue

        thickness = area / max(
            perimeter,
            1,
        )

        fill_ratio = area / max(
            w * h,
            1,
        )

        # Dünne Strukturen bevorzugen und
        # breite Silikoneinschnitte verwerfen.
        if (
            thickness < 2.2
            and fill_ratio < 0.18
            and max(w, h) > 12
        ):
            thread_contours.append(
                contour
            )

    return thread_mask, thread_contours
def get_thread_endpoints(
    thread_contours: list[np.ndarray],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Die beiden Endbereiche des längsten Fadens näherungsweise bestimmen."""

    if not thread_contours:
        return None

    longest = max(
        thread_contours,
        key=lambda contour: cv.arcLength(
            contour,
            False,
        ),
    )

    points = longest.reshape(-1, 2).astype(
        np.float32
    )

    if len(points) < 2:
        return None

    center = np.mean(
        points,
        axis=0,
    )

    centered = points - center

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    main_axis = vh[0]

    projection = centered @ main_axis

    first = points[
        np.argmin(projection)
    ]

    second = points[
        np.argmax(projection)
    ]

    return (
        (
            int(first[0]),
            int(first[1]),
        ),
        (
            int(second[0]),
            int(second[1]),
        ),
    )
def detect_needle(
    image: np.ndarray,
    roi: np.ndarray,
    thread_contours: list[np.ndarray],
) -> tuple[
    tuple[int, int, int, int] | None,
    np.ndarray,
    dict[str, float] | None,
]:
    """
    Dunkle, dünne Nadel in der Nähe eines Fadenendes erkennen.

    Die Fadenmaske wird nicht von der Nadelsuche subtrahiert.
    """

    gray_original = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY,
    )

    # Lokalen Kontrast verbessern.
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(
        gray_original
    )

    blurred = cv.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    threshold = cv.getTrackbarPos(
        "Dunkle Nadel",
        WINDOW,
    )

    canny_low = cv.getTrackbarPos(
        "Canny niedrig",
        WINDOW,
    )

    canny_high = cv.getTrackbarPos(
        "Canny hoch",
        WINDOW,
    )

    # ---------------------------------------------------------
    # Dunkle, dünne Strukturen hervorheben.
    # ---------------------------------------------------------

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (21, 21),
    )

    black_hat = cv.morphologyEx(
        blurred,
        cv.MORPH_BLACKHAT,
        kernel,
    )

    _, dark_mask = cv.threshold(
        black_hat,
        threshold,
        255,
        cv.THRESH_BINARY,
    )

    # Nur innerhalb des Silikonpads suchen.
    dark_mask = cv.bitwise_and(
        dark_mask,
        roi,
    )

    edges = cv.Canny(
        blurred,
        canny_low,
        max(
            canny_low + 1,
            canny_high,
        ),
    )

    needle_mask = cv.bitwise_and(
        dark_mask,
        edges,
    )

    small_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (3, 3),
    )

    needle_mask = cv.morphologyEx(
        needle_mask,
        cv.MORPH_CLOSE,
        small_kernel,
        iterations=1,
    )

    # ---------------------------------------------------------
    # Fadenmaske nur zur Bewertung verwenden.
    # Sie wird NICHT von needle_mask abgezogen.
    # ---------------------------------------------------------

    thread_mask = np.zeros_like(
        gray_original
    )

    if thread_contours:
        cv.drawContours(
            thread_mask,
            thread_contours,
            -1,
            255,
            3,
        )

    thread_endpoints = get_thread_endpoints(
        thread_contours
    )

    if thread_endpoints is None:
        return None, needle_mask, None

    # ---------------------------------------------------------
    # Gerade Nadelabschnitte suchen.
    # ---------------------------------------------------------

    lines = cv.HoughLinesP(
        needle_mask,
        1,
        np.pi / 180,
        threshold=7,
        minLineLength=12,
        maxLineGap=10,
    )

    if lines is None:
        return None, needle_mask, None

    best_line = None
    best_score = -1e9
    best_metrics = None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = float(
            np.hypot(
                x2 - x1,
                y2 - y1,
            )
        )

        # Zu kurze bzw. zu lange Pad-Kanten verwerfen.
        if length < 12 or length > 110:
            continue

        # Mittelpunkt muss im ROI liegen.
        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        if not (
            0 <= center_x < roi.shape[1]
            and 0 <= center_y < roi.shape[0]
        ):
            continue

        if roi[
            center_y,
            center_x,
        ] == 0:
            continue

        # -----------------------------------------------------
        # Ein Ende der Nadel muss nahe an einem Fadenende liegen.
        # -----------------------------------------------------

        line_points = [
            np.array(
                [x1, y1],
                dtype=np.float32,
            ),
            np.array(
                [x2, y2],
                dtype=np.float32,
            ),
        ]

        thread_points = [
            np.array(
                thread_endpoints[0],
                dtype=np.float32,
            ),
            np.array(
                thread_endpoints[1],
                dtype=np.float32,
            ),
        ]

        endpoint_distance = min(
            float(
                np.linalg.norm(
                    lp - tp
                )
            )
            for lp in line_points
            for tp in thread_points
        )

        # Nadel sollte mit einem Fadenende verbunden sein.
        if endpoint_distance > 65:
            continue

        # -----------------------------------------------------
        # Prüfen, ob die Linie selbst der Faden ist.
        # -----------------------------------------------------

        line_mask = np.zeros_like(
            gray_original
        )

        cv.line(
            line_mask,
            (x1, y1),
            (x2, y2),
            255,
            3,
        )

        line_pixels = cv.countNonZero(
            line_mask
        )

        overlap = cv.bitwise_and(
            line_mask,
            thread_mask,
        )

        overlap_ratio = (
            cv.countNonZero(overlap)
            / max(
                line_pixels,
                1,
            )
        )

        # Linien, die fast vollständig zum Faden gehören,
        # nicht als Nadel verwenden.
        if overlap_ratio > 0.45:
            continue

        # -----------------------------------------------------
        # Lokalen Helligkeitskontrast bestimmen.
        # -----------------------------------------------------

        needle_brightness = cv.mean(
            gray_original,
            mask=line_mask,
        )[0]

        surrounding = cv.dilate(
            line_mask,
            np.ones(
                (13, 13),
                dtype=np.uint8,
            ),
            iterations=1,
        )

        surrounding = cv.bitwise_and(
            surrounding,
            cv.bitwise_not(
                line_mask
            ),
        )

        background_brightness = cv.mean(
            gray_original,
            mask=surrounding,
        )[0]

        local_contrast = (
            background_brightness
            - needle_brightness
        )

        # Die Nadel ist in diesem Video dunkler
        # als das umgebende Silikon.
        if local_contrast < 1.5:
            continue

        score = (
            1.5 * length
            + 4.0 * local_contrast
            - 1.2 * endpoint_distance
            - 100.0 * overlap_ratio
        )

        if score > best_score:
            best_score = score
            best_line = (
                x1,
                y1,
                x2,
                y2,
            )

            best_metrics = {
                "length": length,
                "endpoint_distance": endpoint_distance,
                "overlap_ratio": overlap_ratio,
                "local_contrast": local_contrast,
                "score": score,
            }

    return best_line, needle_mask, best_metrics


def process_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Nadel und Faden gemeinsam erkennen."""

    # Linsenverzerrung korrigieren.
    image = cv.undistort(
        frame,
        camera_matrix,
        distortion,
    )

    gray = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY,
    )

    gray = cv.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    height, width = gray.shape

    # ROI für den Bereich des Silikonpads erzeugen.
    roi = np.zeros_like(gray)

    roi[
        int(height * 0.14):int(height * 0.98),
        int(width * 0.26):int(width * 0.96),
    ] = 255

    # Faden erkennen.
    thread_mask, thread_contours = detect_thread(
        gray,
        roi,
    )

    # Nadel erkennen.
    needle, needle_mask, needle_metrics = detect_needle(
        image,
        roi,
        thread_contours,
    )

    result = image.copy()

    # ---------------------------------------------------------
    # Faden darstellen
    # ---------------------------------------------------------

    if thread_contours:
        cv.drawContours(
            result,
            thread_contours,
            -1,
            (255, 0, 0),
            2,
        )

        cv.putText(
            result,
            "Chirurgischer Faden",
            (30, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    # ---------------------------------------------------------
    # Nadel darstellen
    # ---------------------------------------------------------

    if needle is not None:
        x1, y1, x2, y2 = needle

        # Zweidimensionale geometrische Informationen der Nadel.
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Approximative Endpunkte des längsten Fadenabschnitts.
        thread_endpoints = get_thread_endpoints(
            thread_contours
        )

        # Mathematische Detektionsergebnisse im Terminal ausgeben.
        print("\n--- DETEKTIONSERGEBNIS ---")
        print(f"Nadelpunkt 1: ({x1}, {y1})")
        print(f"Nadelpunkt 2: ({x2}, {y2})")
        print(
            "Nadelmittelpunkt: "
            f"({center_x:.1f}, {center_y:.1f})"
        )

        if needle_metrics is not None:
            print(
                "Nadellaenge: "
                f"{needle_metrics['length']:.2f} px"
            )
            print(
                "Abstand zum Fadenende: "
                f"{needle_metrics['endpoint_distance']:.2f} px"
            )
            print(
                "Overlap-Verhaeltnis: "
                f"{needle_metrics['overlap_ratio']:.3f}"
            )
            print(
                "Lokaler Kontrast: "
                f"{needle_metrics['local_contrast']:.2f}"
            )
            print(
                "Heuristischer Score: "
                f"{needle_metrics['score']:.2f}"
            )

        if thread_endpoints is not None:
            print(
                f"Fadenende 1: {thread_endpoints[0]}"
            )
            print(
                f"Fadenende 2: {thread_endpoints[1]}"
            )

        cv.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv.circle(
            result,
            (x1, y1),
            5,
            (0, 0, 255),
            -1,
        )

        cv.circle(
            result,
            (x2, y2),
            5,
            (0, 0, 255),
            -1,
        )

        cv.putText(
            result,
            "Chirurgische Nadel",
            (
                min(x1, x2),
                max(
                    30,
                    min(y1, y2) - 10,
                ),
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Messwerte direkt im Ergebnisbild anzeigen.
        if needle_metrics is not None:
            text_x = min(x1, x2)
            text_y = min(
                result.shape[0] - 70,
                max(y1, y2) + 30,
            )

            cv.putText(
                result,
                f"L = {needle_metrics['length']:.1f} px",
                (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv.putText(
                result,
                f"d = {needle_metrics['endpoint_distance']:.1f} px",
                (text_x, text_y + 22),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv.putText(
                result,
                f"S = {needle_metrics['score']:.1f}",
                (text_x, text_y + 44),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    else:
        cv.putText(
            result,
            "Keine Nadel erkannt",
            (30, 75),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Debug-Fenster.
    cv.imshow(
        "Nadelmaske",
        needle_mask,
    )

    cv.imshow(
        "Fadenmaske",
        thread_mask,
    )

    return result

def main() -> None:
    """Das Video verarbeiten."""

    camera_matrix, distortion = (
        load_calibration()
    )

    if not VIDEO_PATH.exists():
        raise RuntimeError(
            f"Video wurde nicht gefunden: {VIDEO_PATH}"
        )

    video = cv.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Video konnte nicht geöffnet werden: {VIDEO_PATH}"
        )

    create_controls()

    paused = False
    frame = None

    print("Leertaste: Pause oder Fortsetzen")
    print("q: Programm beenden")

    try:
        while True:
            if not paused:
                success, frame = video.read()

                if not success:
                    print("Videoende erreicht.")
                    break

            if frame is None:
                break

            result = process_frame(
                frame,
                camera_matrix,
                distortion,
            )

            cv.imshow(
                "Nadel und Faden",
                result,
            )

            key = cv.waitKey(
                30 if paused else 1
            ) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                paused = not paused

    finally:
        video.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()