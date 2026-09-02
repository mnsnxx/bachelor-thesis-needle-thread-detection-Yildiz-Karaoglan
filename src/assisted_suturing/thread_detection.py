"""Experimentelle Erkennung eines dunkelblauen chirurgischen Fadens."""

from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np


# ============================================================
# PROJEKTPFADE
# ============================================================

# Hauptverzeichnis des Projekts bestimmen.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pfad zum Testvideo mit dem chirurgischen Faden.
VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "phantom_videos"
    / "02_thread_different_angles.mp4"
)

# Verzeichnis mit den gespeicherten Kalibrierungspaketen.
CALIBRATION_DIR = (
    PROJECT_ROOT
    / "calibration"
)

# Fenstername für die einstellbaren Parameter.
WINDOW = "Fadenparameter"


# ============================================================
# KAMERAKALIBRIERUNG LADEN
# ============================================================

def load_calibration() -> tuple[np.ndarray, np.ndarray]:
    """
    Lädt das neueste erfolgreich validierte
    intrinsische Kalibrierungspaket.
    """

    calibration_files = list(
        CALIBRATION_DIR.glob(
            "calibration_intrinsic-*.json"
        )
    )

    if not calibration_files:
        raise RuntimeError(
            "Kein Kalibrierungspaket gefunden."
        )

    # Das zuletzt gespeicherte Kalibrierungspaket auswählen.
    calibration_path = max(
        calibration_files,
        key=lambda file: file.stat().st_mtime,
    )

    # JSON-Datei einlesen.
    data = json.loads(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )

    # Nur eine erfolgreich validierte Kalibrierung verwenden.
    if data["validation"]["status"] != "PASS":
        raise RuntimeError(
            "Die Kamerakalibrierung ist nicht gültig."
        )

    print(
        f"Kalibrierung: {calibration_path}"
    )

    # Kameramatrix laden.
    camera_matrix = np.asarray(
        data["camera"]["K"],
        dtype=np.float64,
    )

    # Distorsionskoeffizienten laden.
    distortion = np.asarray(
        data["camera"]["distortion"],
        dtype=np.float64,
    )

    return (
        camera_matrix,
        distortion,
    )


# ============================================================
# TRACKBARS ERZEUGEN
# ============================================================

def create_controls() -> None:
    """
    Erzeugt Schieberegler zur Anpassung der
    HSV-Farbsegmentierung und der Konturfilterung.
    """

    cv.namedWindow(
        WINDOW,
        cv.WINDOW_NORMAL,
    )

    callback = lambda _: None

    # --------------------------------------------------------
    # Farbton
    # --------------------------------------------------------

    # Untere Grenze des Farbtons.
    cv.createTrackbar(
        "H minimal",
        WINDOW,
        75,
        179,
        callback,
    )

    # Obere Grenze des Farbtons.
    cv.createTrackbar(
        "H maximal",
        WINDOW,
        170,
        179,
        callback,
    )

    # --------------------------------------------------------
    # Sättigung
    # --------------------------------------------------------

    # Der Faden erscheint teilweise nur schwach gesättigt.
    cv.createTrackbar(
        "S minimal",
        WINDOW,
        5,
        255,
        callback,
    )

    # --------------------------------------------------------
    # Helligkeit
    # --------------------------------------------------------

    cv.createTrackbar(
        "V minimal",
        WINDOW,
        0,
        255,
        callback,
    )

    cv.createTrackbar(
        "V maximal",
        WINDOW,
        245,
        255,
        callback,
    )

    # --------------------------------------------------------
    # Konturlänge
    # --------------------------------------------------------

    cv.createTrackbar(
        "Min Konturlaenge",
        WINDOW,
        15,
        1000,
        callback,
    )


# ============================================================
# FADENERKENNUNG
# ============================================================

def detect_thread(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Erkennt den dunkelblauen chirurgischen Faden.

    Verarbeitungsschritte:

    1. Bildentzerrung
    2. Umwandlung von BGR nach HSV
    3. Definition einer Region of Interest
    4. Farbsegmentierung des Fadens
    5. Morphologische Verstärkung
    6. Schließen kleiner Unterbrechungen
    7. Konturbestimmung
    8. Geometrische Filterung
    9. Darstellung gültiger Fadenabschnitte

    Rückgabewerte:

    result:
        Ergebnisbild mit eingezeichneten Fadenkonturen

    mask:
        Binärmaske der segmentierten Fadenbereiche

    thread_detected:
        True, wenn mindestens ein gültiger
        Fadenabschnitt erkannt wurde
    """

    # ========================================================
    # 1. BILD ENTZERREN
    # ========================================================

    corrected = cv.undistort(
        frame,
        camera_matrix,
        distortion,
    )

    # Kopie zur Darstellung erzeugen.
    result = corrected.copy()


    # ========================================================
    # 2. BGR -> HSV
    # ========================================================

    hsv = cv.cvtColor(
        corrected,
        cv.COLOR_BGR2HSV,
    )


    # ========================================================
    # 3. TRACKBAR-WERTE AUSLESEN
    # ========================================================

    h_min = cv.getTrackbarPos(
        "H minimal",
        WINDOW,
    )

    h_max = cv.getTrackbarPos(
        "H maximal",
        WINDOW,
    )

    s_min = cv.getTrackbarPos(
        "S minimal",
        WINDOW,
    )

    v_min = cv.getTrackbarPos(
        "V minimal",
        WINDOW,
    )

    v_max = cv.getTrackbarPos(
        "V maximal",
        WINDOW,
    )

    min_contour_length = cv.getTrackbarPos(
        "Min Konturlaenge",
        WINDOW,
    )

    # Eine minimale Konturlänge von 0 vermeiden.
    min_contour_length = max(
        min_contour_length,
        1,
    )


    # ========================================================
    # 4. REGION OF INTEREST
    # ========================================================

    height, width = corrected.shape[:2]

    roi = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # Der relevante Arbeitsbereich umfasst hauptsächlich
    # das Silikonphantom.
    #
    # Der obere Bereich wurde bewusst erweitert,
    # damit der dort verlaufende Faden nicht abgeschnitten wird.

    x_start = int(width * 0.20)
    x_end = int(width * 0.99)

    y_start = int(height * 0.08)
    y_end = int(height * 0.99)

    roi[
        y_start:y_end,
        x_start:x_end,
    ] = 255


    # Region of Interest zur visuellen Kontrolle darstellen.
    cv.rectangle(
        result,
        (x_start, y_start),
        (x_end, y_end),
        (0, 255, 255),
        1,
    )


    # ========================================================
    # 5. HSV-FARBSEGMENTIERUNG
    # ========================================================

    # Untere HSV-Grenze des dunkelblauen Fadens.
    lower_thread = np.array(
        [
            h_min,
            s_min,
            v_min,
        ],
        dtype=np.uint8,
    )

    # Obere HSV-Grenze.
    upper_thread = np.array(
        [
            h_max,
            255,
            v_max,
        ],
        dtype=np.uint8,
    )

    # Binärmaske erzeugen.
    #
    # Weiß:
    # Pixel innerhalb des definierten HSV-Bereichs
    #
    # Schwarz:
    # übrige Bildbereiche
    mask = cv.inRange(
        hsv,
        lower_thread,
        upper_thread,
    )


    # Nur Pixel innerhalb der Region of Interest verwenden.
    mask = cv.bitwise_and(
        mask,
        roi,
    )


    # ========================================================
    # 6. MORPHOLOGISCHE VERARBEITUNG
    # ========================================================

    # Der chirurgische Faden ist teilweise nur wenige
    # Pixel breit. Aus diesem Grund wird kein Medianfilter
    # eingesetzt, da dieser dünne Fadenbereiche entfernen kann.


    # --------------------------------------------------------
    # Kleine Dilatation
    # --------------------------------------------------------

    kernel_dilate = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (3, 3),
    )

    # Dünne segmentierte Fadenbereiche leicht verstärken.
    mask = cv.dilate(
        mask,
        kernel_dilate,
        iterations=1,
    )


    # --------------------------------------------------------
    # Closing
    # --------------------------------------------------------

    # Ein etwas größeres Strukturelement wird verwendet,
    # damit kleine Unterbrechungen des Fadens verbunden werden.
    kernel_close = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        kernel_close,
        iterations=2,
    )


    # ========================================================
    # 7. KONTUREN BESTIMMEN
    # ========================================================

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE,
    )


    # Liste aller gültigen Fadenkandidaten.
    valid_contours: list[np.ndarray] = []


    # ========================================================
    # 8. KONTUREN GEOMETRISCH FILTERN
    # ========================================================

    for contour in contours:

        # ----------------------------------------------------
        # Konturlänge
        # ----------------------------------------------------

        perimeter = cv.arcLength(
            contour,
            False,
        )

        if perimeter < min_contour_length:
            continue


        # ----------------------------------------------------
        # Konturfläche
        # ----------------------------------------------------

        area = cv.contourArea(
            contour
        )

        # Einzelne Pixel bzw. extrem kleine Bereiche entfernen.
        if area < 1:
            continue


        # ----------------------------------------------------
        # Begrenzungsrechteck
        # ----------------------------------------------------

        x, y, w, h = cv.boundingRect(
            contour
        )

        if w <= 1 and h <= 1:
            continue


        # Ein Fadenabschnitt sollte eine gewisse
        # räumliche Ausdehnung besitzen.
        if max(w, h) < 10:
            continue


        # ----------------------------------------------------
        # Geometrische Dicke
        # ----------------------------------------------------

        # Das Verhältnis aus Fläche und Konturlänge
        # dient als einfaches Maß für die Dicke.
        #
        # Dünne längliche Strukturen besitzen
        # einen vergleichsweise kleinen Wert.
        thickness = (
            area / perimeter
            if perimeter > 0
            else 999.0
        )

        # Zu breite Strukturen werden verworfen.
        if thickness > 4.0:
            continue


        # ----------------------------------------------------
        # Füllverhältnis
        # ----------------------------------------------------

        bounding_area = w * h

        fill_ratio = (
            area / bounding_area
            if bounding_area > 0
            else 1.0
        )

        # Kompakte Flächen sollen nicht als
        # chirurgischer Faden interpretiert werden.
        if fill_ratio > 0.45:
            continue


        # Kontur als möglichen Fadenabschnitt akzeptieren.
        valid_contours.append(
            contour
        )


    # ========================================================
    # 9. ERGEBNIS BESTIMMEN
    # ========================================================

    thread_detected = (
        len(valid_contours) > 0
    )


    if thread_detected:

        # ----------------------------------------------------
        # Alle gültigen Fadenabschnitte einzeichnen.
        #
        # Es wird bewusst kein großes Bounding-Rechteck
        # verwendet, da der Faden eine gekrümmte und
        # deformierbare Struktur besitzt.
        # ----------------------------------------------------

        cv.drawContours(
            result,
            valid_contours,
            -1,
            (255, 0, 0),
            2,
        )


        # ----------------------------------------------------
        # Gesamtlänge der erkannten Konturen bestimmen.
        # ----------------------------------------------------

        total_length = sum(
            cv.arcLength(
                contour,
                False,
            )
            for contour in valid_contours
        )


        # ----------------------------------------------------
        # Obersten Punkt der erkannten Konturen bestimmen.
        # ----------------------------------------------------

        all_points = np.vstack(
            valid_contours
        )

        x, y, w, h = cv.boundingRect(
            all_points
        )


        # ----------------------------------------------------
        # Beschriftung
        # ----------------------------------------------------

        cv.putText(
            result,
            "Chirurgischer Faden",
            (
                x,
                max(
                    y - 15,
                    30,
                ),
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )


        cv.putText(
            result,
            f"Konturlaenge: {total_length:.1f}",
            (
                x,
                min(
                    y + h + 30,
                    height - 15,
                ),
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )


    else:

        # ----------------------------------------------------
        # Keine gültige Fadenstruktur erkannt.
        # ----------------------------------------------------

        cv.putText(
            result,
            "Kein Faden erkannt",
            (30, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )


    # ========================================================
    # 10. ERGEBNISSE ZURÜCKGEBEN
    # ========================================================

    return (
        result,
        mask,
        thread_detected,
    )


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    """
    Lädt das Phantomvideo und führt die
    Fadenerkennung für jedes Videoframe aus.
    """

    # ========================================================
    # KAMERAKALIBRIERUNG LADEN
    # ========================================================

    camera_matrix, distortion = (
        load_calibration()
    )


    # ========================================================
    # VIDEO PRÜFEN
    # ========================================================

    if not VIDEO_PATH.exists():

        raise RuntimeError(
            f"Video wurde nicht gefunden: "
            f"{VIDEO_PATH}"
        )


    # ========================================================
    # VIDEO ÖFFNEN
    # ========================================================

    video = cv.VideoCapture(
        str(VIDEO_PATH)
    )


    if not video.isOpened():

        raise RuntimeError(
            f"Video konnte nicht geöffnet werden: "
            f"{VIDEO_PATH}"
        )


    # ========================================================
    # TRACKBARS ERZEUGEN
    # ========================================================

    create_controls()


    # ========================================================
    # PROGRAMMSTATUS
    # ========================================================

    paused = False
    frame = None


    print(
        f"Video: {VIDEO_PATH}"
    )

    print(
        "Leertaste: Pause oder Fortsetzen"
    )

    print(
        "q: Programm beenden"
    )


    # ========================================================
    # VIDEO-SCHLEIFE
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Neues Videoframe lesen.
            # ------------------------------------------------

            if not paused:

                success, frame = (
                    video.read()
                )

                if not success:

                    print(
                        "Videoende erreicht."
                    )

                    break


            if frame is None:
                break


            # ------------------------------------------------
            # Fadenerkennung durchführen.
            # ------------------------------------------------

            (
                result,
                mask,
                thread_detected,
            ) = detect_thread(
                frame,
                camera_matrix,
                distortion,
            )


            # ------------------------------------------------
            # Ergebnisbild anzeigen.
            # ------------------------------------------------

            cv.imshow(
                "Fadenerkennung",
                result,
            )


            # ------------------------------------------------
            # Binärmaske anzeigen.
            #
            # Weiß  = möglicher Faden
            # Schwarz = Hintergrund
            # ------------------------------------------------

            cv.imshow(
                "Fadenmaske",
                mask,
            )


            # ------------------------------------------------
            # Tastatureingabe verarbeiten.
            # ------------------------------------------------

            key = cv.waitKey(
                30 if paused else 1
            ) & 0xFF


            # q beendet das Programm.
            if key == ord("q"):
                break


            # Leertaste pausiert bzw. startet
            # die Videowiedergabe.
            if key == ord(" "):
                paused = not paused


    # ========================================================
    # RESSOURCEN FREIGEBEN
    # ========================================================

    finally:

        video.release()

        cv.destroyAllWindows()


# ============================================================
# PROGRAMMSTART
# ============================================================

if __name__ == "__main__":
    main()
    #fadenparametern 85  179 5 0 245 15