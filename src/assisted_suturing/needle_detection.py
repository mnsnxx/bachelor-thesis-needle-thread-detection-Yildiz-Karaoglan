from __future__ import annotations

import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np


# Projektstammverzeichnis unabhängig vom Terminalpfad bestimmen.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "phantom_videos"
    / "01_needle_normal_light.mp4"
)

CALIBRATION_DIRECTORY = PROJECT_ROOT / "calibration"
PARAMETER_WINDOW = "Parameter"

# Relevanter Arbeitsbereich des Silikonphantoms.
# Bereiche außerhalb des Pads werden für die Nadelerkennung ignoriert.
ROI_X_MIN = 0.12
ROI_X_MAX = 0.96
ROI_Y_MIN = 0.14
ROI_Y_MAX = 0.98


def load_latest_calibration() -> tuple[np.ndarray, np.ndarray]:
    """Das neueste validierte Kalibrierungspaket laden."""

    calibration_files = list(
        CALIBRATION_DIRECTORY.glob(
            "calibration_intrinsic-*.json"
        )
    )

    if not calibration_files:
        raise RuntimeError(
            "Kein Kalibrierungspaket gefunden."
        )

    latest_file = max(
        calibration_files,
        key=lambda path: path.stat().st_mtime,
    )

    package = json.loads(
        latest_file.read_text(encoding="utf-8")
    )

    if package["validation"]["status"] != "PASS":
        raise RuntimeError(
            "Die Kamerakalibrierung ist nicht gültig."
        )

    camera_matrix = np.asarray(
        package["camera"]["K"],
        dtype=np.float64,
    )

    distortion = np.asarray(
        package["camera"]["distortion"],
        dtype=np.float64,
    )

    print(f"Kalibrierung: {latest_file}")

    return camera_matrix, distortion


def resize_for_display(
    image: np.ndarray,
    max_width: int = 1200,
    max_height: int = 750,
) -> np.ndarray:
    """Das Bild nur für die Bildschirmdarstellung verkleinern."""

    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_width / width,
        max_height / height,
    )

    if scale == 1.0:
        return image

    return cv.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv.INTER_AREA,
    )


def nothing(_: int) -> None:
    """Callback-Funktion für die Schieberegler."""

    pass


def create_controls() -> None:
    """Schieberegler für die Nadelerkennung erzeugen."""

    cv.namedWindow(
        PARAMETER_WINDOW,
        cv.WINDOW_NORMAL,
    )

    cv.createTrackbar(
        "Canny niedrig",
        PARAMETER_WINDOW,
        40,
        255,
        nothing,
    )

    cv.createTrackbar(
        "Canny hoch",
        PARAMETER_WINDOW,
        140,
        255,
        nothing,
    )

    cv.createTrackbar(
        "Top-Hat Schwelle",
        PARAMETER_WINDOW,
        15,
        255,
        nothing,
    )

    cv.createTrackbar(
        "Min Umfang",
        PARAMETER_WINDOW,
        20,
        2000,
        nothing,
    )

    cv.createTrackbar(
        "Min Helligkeit",
        PARAMETER_WINDOW,
        110,
        255,
        nothing,
    )

    cv.createTrackbar(
        "Max Saettigung",
        PARAMETER_WINDOW,
        100,
        255,
        nothing,
    )


def detect_needle(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Mögliche Nadelkonturen in einem Videoframe bestimmen."""

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    # Die Linsenverzerrung korrigieren.
    corrected = cv.undistort(
        frame,
        camera_matrix,
        distortion,
    )

    # ------------------------------------------------------------
    # REGION OF INTEREST
    # ------------------------------------------------------------
    # Nur der Bereich des Silikonphantoms wird ausgewertet. Dadurch
    # werden helle Tisch-, Rand- und Hintergrundstrukturen außerhalb
    # des eigentlichen Arbeitsbereichs nicht als Nadelkandidaten
    # berücksichtigt.
    roi_mask = np.zeros(
        (frame_height, frame_width),
        dtype=np.uint8,
    )

    x_min = int(frame_width * ROI_X_MIN)
    x_max = int(frame_width * ROI_X_MAX)
    y_min = int(frame_height * ROI_Y_MIN)
    y_max = int(frame_height * ROI_Y_MAX)

    roi_mask[y_min:y_max, x_min:x_max] = 255

    # Das Bild in Graustufen und HSV umwandeln.
    gray_original = cv.cvtColor(
        corrected,
        cv.COLOR_BGR2GRAY,
    )

    hsv = cv.cvtColor(
        corrected,
        cv.COLOR_BGR2HSV,
    )

    # Den lokalen Kontrast verbessern.
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray_original)

    # Bildrauschen reduzieren.
    blurred = cv.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Aktuelle Parameter auslesen.
    canny_low = cv.getTrackbarPos(
        "Canny niedrig",
        PARAMETER_WINDOW,
    )

    canny_high = cv.getTrackbarPos(
        "Canny hoch",
        PARAMETER_WINDOW,
    )

    top_hat_threshold = cv.getTrackbarPos(
        "Top-Hat Schwelle",
        PARAMETER_WINDOW,
    )

    minimum_perimeter = cv.getTrackbarPos(
        "Min Umfang",
        PARAMETER_WINDOW,
    )

    minimum_brightness = cv.getTrackbarPos(
        "Min Helligkeit",
        PARAMETER_WINDOW,
    )

    maximum_saturation = cv.getTrackbarPos(
        "Max Saettigung",
        PARAMETER_WINDOW,
    )

    # Helle und dünne Strukturen hervorheben.
    top_hat_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (21, 21),
    )

    top_hat = cv.morphologyEx(
        blurred,
        cv.MORPH_TOPHAT,
        top_hat_kernel,
    )

    _, top_hat_mask = cv.threshold(
        top_hat,
        top_hat_threshold,
        255,
        cv.THRESH_BINARY,
    )

    # Helle und wenig gesättigte Bereiche als Metall markieren.
    lower_metal = np.array(
        [0, 0, minimum_brightness],
        dtype=np.uint8,
    )

    upper_metal = np.array(
        [179, maximum_saturation, 255],
        dtype=np.uint8,
    )

    metal_mask = cv.inRange(
        hsv,
        lower_metal,
        upper_metal,
    )

    # Nur Metallkandidaten innerhalb des Silikonphantoms behalten.
    metal_mask = cv.bitwise_and(
        metal_mask,
        roi_mask,
    )

    # Kanten mithilfe des Canny-Verfahrens erkennen.
    edges = cv.Canny(
        blurred,
        canny_low,
        max(canny_low + 1, canny_high),
    )

    # Auch Kanten und Top-Hat-Strukturen auf den relevanten
    # Arbeitsbereich beschränken.
    edges = cv.bitwise_and(
        edges,
        roi_mask,
    )

    top_hat_mask = cv.bitwise_and(
        top_hat_mask,
        roi_mask,
    )

    # ------------------------------------------------------------
    # KANDIDATENMASKE
    # ------------------------------------------------------------
    #
    # Die Metallmaske zeigt die Nadel im vorliegenden Versuchsaufbau
    # bereits zuverlässig. Deshalb wird die Top-Hat-Maske nicht mehr
    # als zwingende Bedingung verwendet. Eine Nadel darf entweder
    # über eine Canny-Kante oder über eine lokale Top-Hat-Struktur
    # in die Kandidatenmaske gelangen, muss jedoch weiterhin innerhalb
    # der Metallmaske liegen.
    #
    # Dadurch gehen Nadelbereiche nicht verloren, nur weil sie lokal
    # nicht stark genug durch die Top-Hat-Transformation hervorgehoben
    # werden.

    edge_candidates = cv.bitwise_and(
        edges,
        metal_mask,
    )

    top_hat_candidates = cv.bitwise_and(
        top_hat_mask,
        metal_mask,
    )

    candidate_mask = cv.bitwise_or(
        edge_candidates,
        top_hat_candidates,
    )

    small_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (3, 3),
    )

    # Kleine Unterbrechungen innerhalb der Nadelkante schließen.
    candidate_mask = cv.morphologyEx(
        candidate_mask,
        cv.MORPH_CLOSE,
        small_kernel,
        iterations=2,
    )

    # Dünne Strukturen für die Konturerkennung verstärken.
    candidate_mask = cv.dilate(
        candidate_mask,
        small_kernel,
        iterations=1,
    )

    contours, _ = cv.findContours(
        candidate_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE,
    )

    best_contour = None
    best_score = 0.0
    border_margin = 5

    for contour in contours:
        perimeter = float(
            cv.arcLength(
                contour,
                closed=False,
            )
        )

        # Sehr kleine Konturen verwerfen.
        if perimeter < minimum_perimeter:
            continue

        x, y, box_width, box_height = cv.boundingRect(
            contour
        )

        if box_width < 3 or box_height < 3:
            continue

        bounding_area = box_width * box_height

        # Sehr große Flächen verwerfen.
        # Der Grenzwert wurde etwas gelockert, damit eine längere
        # gekrümmte Nadel nicht allein aufgrund ihrer Bounding Box
        # ausgeschlossen wird.
        if bounding_area > frame_area * 0.20:
            continue

        # Konturen direkt am Bildrand verwerfen.
        if (
            x <= border_margin
            or y <= border_margin
            or x + box_width >= frame_width - border_margin
            or y + box_height >= frame_height - border_margin
        ):
            continue

        # Kandidaten direkt an der ROI-Grenze verwerfen.
        # Dies unterdrückt insbesondere helle Pad- und Hintergrundkanten.
        roi_margin = 8
        if (
            x <= x_min + roi_margin
            or y <= y_min + roi_margin
            or x + box_width >= x_max - roi_margin
            or y + box_height >= y_max - roi_margin
        ):
            continue

        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0

        # Zusätzlich muss der Mittelpunkt einen Mindestabstand zur
        # ROI-Grenze besitzen. Eine echte Nadel liegt im Versuchsaufbau
        # auf der Oberfläche des Silikonphantoms und nicht direkt auf
        # dessen äußerem Rand.
        min_center_distance = min(
            center_x - x_min,
            x_max - center_x,
            center_y - y_min,
            y_max - center_y,
        )

        required_center_distance = max(
            12.0,
            0.025 * min(frame_width, frame_height),
        )

        if min_center_distance < required_center_distance:
            continue

        aspect_ratio = max(
            box_width,
            box_height,
        ) / max(
            1,
            min(box_width, box_height),
        )

        # Nahezu quadratische Störbereiche verwerfen.
        # Der Grenzwert ist bewusst weniger streng, da eine gekrümmte
        # Nadel in ihrer Bounding Box kompakter erscheinen kann.
        if aspect_ratio < 1.05:
            continue

        # Sehr große Bildstrukturen verwerfen. Der Grenzwert ist
        # größer als zuvor, damit längere Nadelabschnitte erhalten
        # bleiben.
        if (
            box_width > frame_width * 0.60
            or box_height > frame_height * 0.60
        ):
            continue

        # Ein großes Seitenverhältnis wird nicht mehr hart verworfen.
        # Dünne Nadelabschnitte können sehr länglich erscheinen.
        # Der Einfluss wird später lediglich im Score begrenzt.

        contour_mask = np.zeros_like(
            gray_original
        )

        cv.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            thickness=2,
        )

        mean_brightness = float(
            cv.mean(
                gray_original,
                mask=contour_mask,
            )[0]
        )

        mean_saturation = float(
            cv.mean(
                hsv[:, :, 1],
                mask=contour_mask,
            )[0]
        )

        # Helligkeit und Sättigung werden an dieser Stelle nicht mehr
        # als harte Ausschlusskriterien verwendet. Die Kandidatenmaske
        # wurde bereits mit der Metallmaske eingeschränkt. Ein harter
        # Mittelwertfilter kann dennoch eine echte Nadel verwerfen,
        # weil die Konturmaske auch einige Pixel des Silikonhintergrunds
        # umfasst. Beide Größen werden deshalb nur noch weich im Score
        # berücksichtigt.

        contour_area = float(
            cv.contourArea(contour)
        )

        fill_ratio = contour_area / max(
            1,
            bounding_area,
        )

        thinness = 1.0 - min(
            fill_ratio,
            1.0,
        )

        # Die photometrischen Faktoren werden begrenzt, damit eine
        # lokal etwas dunklere oder stärker gesättigte Nadel nicht
        # vollständig aus der Bewertung herausfällt.
        brightness_factor = max(
            0.35,
            mean_brightness / 255.0,
        )

        saturation_factor = max(
            0.35,
            1.0 - mean_saturation / 255.0,
        )

        # Anteil der Konturumgebung, der weiterhin von der Metallmaske
        # unterstützt wird.
        metal_overlap_mask = cv.bitwise_and(
            contour_mask,
            metal_mask,
        )

        contour_pixels = max(
            1,
            cv.countNonZero(contour_mask),
        )

        metal_support = (
            cv.countNonZero(metal_overlap_mask)
            / contour_pixels
        )

        # Abstand zur ROI-Grenze als weiches Bewertungskriterium.
        # Kandidaten nahe am Rand erhalten einen deutlich kleineren Wert.
        border_reference = max(
            1.0,
            0.10 * min(frame_width, frame_height),
        )

        border_factor = min(
            1.0,
            min_center_distance / border_reference,
        )

        # Sehr lange Konturen werden leicht begrenzt, damit eine
        # durchgehende Padkante nicht allein wegen ihrer Länge gewinnt.
        perimeter_factor = min(
            perimeter,
            220.0,
        )

        # Lange, dünne und metallisch unterstützte Strukturen erhalten
        # einen höheren Wert. Randnahe Strukturen werden abgewertet.
        score = (
            perimeter_factor
            * min(aspect_ratio, 10.0)
            * (0.5 + thinness)
            * brightness_factor
            * saturation_factor
            * (0.75 + 0.25 * metal_support)
            * (0.20 + 0.80 * border_factor)
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    result = corrected.copy()

    # ROI nur dezent zur Kontrolle anzeigen.
    cv.rectangle(
        result,
        (x_min, y_min),
        (x_max, y_max),
        (180, 180, 180),
        1,
    )

    if best_contour is not None:
        x, y, box_width, box_height = cv.boundingRect(
            best_contour
        )

        # Den besten Nadelkandidaten grün markieren.
        cv.drawContours(
            result,
            [best_contour],
            -1,
            (0, 255, 0),
            2,
        )

        cv.rectangle(
            result,
            (x, y),
            (
                x + box_width,
                y + box_height,
            ),
            (0, 255, 0),
            2,
        )

        center = (
            x + box_width // 2,
            y + box_height // 2,
        )

        cv.circle(
            result,
            center,
            5,
            (0, 0, 255),
            -1,
        )

        cv.putText(
            result,
            "Chirurgische Nadel",
            (
                x,
                max(30, y - 10),
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv.putText(
            result,
            f"Score: {best_score:.1f}",
            (
                x,
                min(
                    result.shape[0] - 10,
                    y + box_height + 25,
                ),
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    else:
        cv.putText(
            result,
            "Keine Nadel erkannt",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # ------------------------------------------------------------
    # ERKENNUNGSSTATUS
    # ------------------------------------------------------------

    # Wenn eine gültige beste Kontur gefunden wurde,
    # gilt die chirurgische Nadel als erkannt.
    needle_detected = best_contour is not None

    # ------------------------------------------------------------
    # RÜCKGABE
    # ------------------------------------------------------------
     
    return (
        result,
        metal_mask,
        candidate_mask,
        needle_detected,
    )


def main() -> None:
    """Das Phantomvideo verarbeiten."""

    camera_matrix, distortion = (
        load_latest_calibration()
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
            "Das Video konnte nicht geöffnet werden: "
            f"{VIDEO_PATH}"
        )

    create_controls()

    print(f"Video: {VIDEO_PATH}")
    print("Leertaste: Pause oder Fortsetzen")
    print("q: Programm beenden")

    paused = False
    current_frame = None

    try:
        while True:
            if not paused:
                success, current_frame = video.read()

                if not success:
                    print("Videoende erreicht.")
                    break

            if current_frame is None:
                break

            start_time = time.perf_counter()

            (
                result,
                metal_mask,
                candidate_mask,
                needle_detected,
            ) = detect_needle(
                current_frame,
                camera_matrix,
                distortion,
            )

            processing_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            cv.putText(
                result,
                f"Verarbeitung: {processing_ms:.1f} ms",
                (
                    20,
                    result.shape[0] - 20,
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv.imshow(
                "Nadelerkennung",
                resize_for_display(result),
            )

            cv.imshow(
                "Metallmaske",
                resize_for_display(metal_mask),
            )

            cv.imshow(
                "Kandidatenmaske",
                resize_for_display(candidate_mask),
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