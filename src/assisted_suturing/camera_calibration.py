"""Production camera-intrinsic calibration module.

The implementation follows the behaviour of the supplied reference but does
not import or call it. The module:

* detects chessboard inner corners,
* calls OpenCV ``calibrateCamera``,
* validates the result on separate held-out images,
* writes a new versioned JSON calibration package,
* creates visual original/undistorted comparison images,
* never sends robot commands.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2 as cv
import numpy as np


class CalibrationError(RuntimeError):
    """Raised when calibration input, computation, or package validation fails."""


@dataclass(frozen=True)
class CornerCollection:
    object_points: list[np.ndarray]
    image_points: list[np.ndarray]
    image_size: tuple[int, int]
    accepted_images: list[Path]
    rejected_images: list[Path]


@dataclass(frozen=True)
class CalibrationResult:
    opencv_rms_px: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_vectors: list[np.ndarray]
    translation_vectors_m: list[np.ndarray]
    calibration_per_view_rmse_px: list[float]


@dataclass(frozen=True)
class ValidationResult:
    reprojection_rmse_px: float
    per_image_rmse_px: dict[str, float]
    accepted_images: list[Path]
    rejected_images: list[Path]


@dataclass(frozen=True)
class LoadedCalibration:
    version: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    resolution_px: tuple[int, int]
    camera_serial: str
    lens_setting: str


def create_object_points(
    inner_corners: tuple[int, int],
    square_size_m: float,
) -> np.ndarray:
    """Create chessboard coordinates in the chessboard frame, in metres.

    ``inner_corners`` is ``(columns, rows)``. The physical scale is applied
    exactly once here. Translation vectors returned by OpenCV therefore use
    metres as well.
    """
    columns, rows = inner_corners

    if columns < 2 or rows < 2:
        raise ValueError("Die Anzahl der inneren Ecken muss mindestens 2 x 2 sein.")
    if not math.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("square_size_m muss eine positive Zahl in Metern sein.")

    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
        * np.float32(square_size_m)
    )
    return points


def find_image_paths(pattern: str) -> list[Path]:
    """Expand a glob pattern and return sorted file paths."""
    return [Path(path) for path in sorted(glob.glob(pattern))]


def collect_chessboard_points(
    image_paths: Sequence[Path],
    inner_corners: tuple[int, int],
    square_size_m: float,
    expected_image_size: tuple[int, int] | None = None,
) -> CornerCollection:
    """Detect refined chessboard corners and enforce one image resolution."""
    if not image_paths:
        raise CalibrationError("Keine Bilder gefunden.")

    object_template = create_object_points(inner_corners, square_size_m)
    termination_criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted_images: list[Path] = []
    rejected_images: list[Path] = []
    detected_image_size = expected_image_size

    for image_path in image_paths:
        image = cv.imread(str(image_path))
        if image is None:
            rejected_images.append(image_path)
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]

        if detected_image_size is None:
            detected_image_size = current_size
        elif current_size != detected_image_size:
            raise CalibrationError(
                "Alle Kalibrierungs- und Validierungsbilder müssen dieselbe "
                f"Auflösung haben. Erwartet: {detected_image_size}, "
                f"gefunden in {image_path}: {current_size}"
            )

        found, corners = cv.findChessboardCorners(
            gray,
            inner_corners,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            rejected_images.append(image_path)
            continue

        refined_corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            termination_criteria,
        )

        # Copy because every view conceptually owns its object-point array.
        object_points.append(object_template.copy())
        image_points.append(refined_corners)
        accepted_images.append(image_path)

    if detected_image_size is None or not object_points:
        raise CalibrationError("Keine gültigen Schachbrettbeobachtungen gefunden.")

    return CornerCollection(
        object_points=object_points,
        image_points=image_points,
        image_size=detected_image_size,
        accepted_images=accepted_images,
        rejected_images=rejected_images,
    )


def calculate_view_reprojection_rmse(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    rotation_vectors: Sequence[np.ndarray],
    translation_vectors: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    """Calculate one pixel RMSE value for every calibration view."""
    errors: list[float] = []

    for object_view, image_view, rvec, tvec in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv.projectPoints(
            object_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        difference = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
        errors.append(rmse)

    return errors


def calibrate_camera(collection: CornerCollection) -> CalibrationResult:
    """Return the five OpenCV calibration outputs plus per-view error values."""
    rms, camera_matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
        collection.object_points,
        collection.image_points,
        collection.image_size,
        None,
        None,
    )

    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise CalibrationError("OpenCV hat ungültige Kalibrierungswerte geliefert.")

    per_view_errors = calculate_view_reprojection_rmse(
        collection.object_points,
        collection.image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )

    return CalibrationResult(
        opencv_rms_px=float(rms),
        camera_matrix=camera_matrix,
        distortion=distortion,
        rotation_vectors=list(rvecs),
        translation_vectors_m=list(tvecs),
        calibration_per_view_rmse_px=per_view_errors,
    )


def validate_on_held_out_images(
    image_paths: Sequence[Path],
    inner_corners: tuple[int, int],
    square_size_m: float,
    expected_image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> ValidationResult:
    """Measure reprojection RMSE on images not used by ``calibrateCamera``.

    For each held-out image, the board pose is estimated using ``solvePnP``.
    The known 3-D chessboard points are then projected back into the image and
    compared with the detected corners.
    """
    collection = collect_chessboard_points(
        image_paths=image_paths,
        inner_corners=inner_corners,
        square_size_m=square_size_m,
        expected_image_size=expected_image_size,
    )

    squared_error_sum = 0.0
    point_count = 0
    per_image_errors: dict[str, float] = {}
    accepted_images: list[Path] = []
    rejected_images = list(collection.rejected_images)

    for path, object_view, image_view in zip(
        collection.accepted_images,
        collection.object_points,
        collection.image_points,
        strict=True,
    ):
        success, rvec, tvec = cv.solvePnP(
            object_view,
            image_view,
            camera_matrix,
            distortion,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
        if not success:
            rejected_images.append(path)
            continue

        projected, _ = cv.projectPoints(
            object_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        difference = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        point_squared_errors = np.sum(difference**2, axis=1)

        image_rmse = float(np.sqrt(np.mean(point_squared_errors)))
        per_image_errors[path.name] = image_rmse
        squared_error_sum += float(np.sum(point_squared_errors))
        point_count += len(point_squared_errors)
        accepted_images.append(path)

    if point_count == 0:
        raise CalibrationError("Keine gültigen, getrennten Validierungsbilder vorhanden.")

    total_rmse = float(math.sqrt(squared_error_sum / point_count))
    return ValidationResult(
        reprojection_rmse_px=total_rmse,
        per_image_rmse_px=per_image_errors,
        accepted_images=accepted_images,
        rejected_images=rejected_images,
    )


def save_undistortion_comparisons(
    image_paths: Sequence[Path],
    output_directory: Path,
    expected_image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[Path]:
    """Save side-by-side images for the required visual straight-line check."""
    output_directory.mkdir(parents=True, exist_ok=True)
    width, height = expected_image_size

    new_matrix, _ = cv.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        1,
        (width, height),
    )

    saved_paths: list[Path] = []
    for image_path in image_paths:
        image = cv.imread(str(image_path))
        if image is None:
            continue
        if image.shape[1::-1] != expected_image_size:
            raise CalibrationError(
                f"Auflösungsabweichung bei visueller Prüfung: {image_path}"
            )

        undistorted = cv.undistort(
            image,
            camera_matrix,
            distortion,
            None,
            new_matrix,
        )

        comparison = np.hstack((image, undistorted))
        cv.putText(
            comparison,
            "Original",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv.putText(
            comparison,
            "Undistorted",
            (width + 20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        output_path = output_directory / f"comparison_{image_path.stem}.jpg"
        if not cv.imwrite(str(output_path), comparison):
            raise CalibrationError(f"Vergleichsbild konnte nicht gespeichert werden: {output_path}")
        saved_paths.append(output_path)

    return saved_paths


def sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return sanitized.strip("-") or "unknown-camera"


def create_version(camera_serial: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"intrinsic-{sanitize_for_filename(camera_serial)}-{timestamp}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def array_to_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array).tolist()


def build_calibration_package(
    *,
    version: str,
    camera_serial: str,
    lens_setting: str,
    mounting_date: str,
    inner_corners: tuple[int, int],
    square_size_m: float,
    collection: CornerCollection,
    result: CalibrationResult,
    validation: ValidationResult,
    max_validation_rmse_px: float,
    approved_by: str | None,
) -> dict[str, Any]:
    """Build a package compatible with the supplied example structure."""
    validation_passed = validation.reprojection_rmse_px <= max_validation_rmse_px
    width, height = collection.image_size

    calibration_views = []
    for path, rvec, tvec, view_error in zip(
        collection.accepted_images,
        result.rotation_vectors,
        result.translation_vectors_m,
        result.calibration_per_view_rmse_px,
        strict=True,
    ):
        calibration_views.append(
            {
                "image": path.name,
                "rotation_vector_rad": np.asarray(rvec).reshape(-1).tolist(),
                "translation_vector_m": np.asarray(tvec).reshape(-1).tolist(),
                "reprojection_rmse_px": view_error,
            }
        )

    return {
        "version": version,
        "camera": {
            "serial": camera_serial,
            "resolution_px": [width, height],
            "lens_setting": lens_setting,
            "mounting_date": mounting_date,
            "chessboard_inner_corners": list(inner_corners),
            "square_size_m": square_size_m,
            "coordinate_frame": "camera_optical_frame",
            "length_unit": "m",
            "K": array_to_list(result.camera_matrix),
            "distortion": np.asarray(result.distortion).reshape(-1).tolist(),
            "opencv_calibration_rms_px": result.opencv_rms_px,
            "reprojection_rmse_px": validation.reprojection_rmse_px,
            "calibration_images_used": [path.name for path in collection.accepted_images],
            "calibration_images_rejected": [path.name for path in collection.rejected_images],
            "calibration_views": calibration_views,
        },
        "transforms": {
            "T_global_camera": None,
            "T_robot1_global": None,
            "T_robot2_global": None,
        },
        "validation": {
            "date_utc": utc_now_iso(),
            "method": "Held-out chessboard images; solvePnP + projectPoints reprojection RMSE",
            "reprojection_rmse_px": validation.reprojection_rmse_px,
            "maximum_allowed_reprojection_rmse_px": max_validation_rmse_px,
            "status": "PASS" if validation_passed else "FAIL",
            "images_used": [path.name for path in validation.accepted_images],
            "images_rejected": [path.name for path in validation.rejected_images],
            "per_image_rmse_px": validation.per_image_rmse_px,
            "position_error_mm": None,
            "approved_by": approved_by,
        },
    }


def save_calibration_package(package: dict[str, Any], output_directory: Path) -> Path:
    """Write a new package and never overwrite the example or an existing version."""
    output_directory.mkdir(parents=True, exist_ok=True)
    version = str(package["version"])

    if version == "example-not-for-hardware":
        raise CalibrationError("Die Beispielversion darf nicht als Hardwarekalibrierung gespeichert werden.")

    output_path = output_directory / f"calibration_{sanitize_for_filename(version)}.json"
    if output_path.exists():
        raise FileExistsError(f"Kalibrierungspaket existiert bereits: {output_path}")

    output_path.write_text(
        json.dumps(package, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_calibration_package(
    package_path: Path,
    *,
    expected_camera_serial: str | None = None,
    expected_resolution_px: tuple[int, int] | None = None,
    expected_lens_setting: str | None = None,
    require_validation_pass: bool = True,
) -> LoadedCalibration:
    """Read K/distortion and reject example, invalid, or mismatching packages."""
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Kalibrierungspaket konnte nicht gelesen werden: {package_path}") from exc

    version = package.get("version")
    camera = package.get("camera", {})
    validation = package.get("validation", {})

    if not isinstance(version, str) or version == "example-not-for-hardware":
        raise CalibrationError("Ungültige Kalibrierungsversion.")

    try:
        camera_matrix = np.asarray(camera["K"], dtype=np.float64)
        distortion = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)
        resolution = tuple(int(value) for value in camera["resolution_px"])
        serial = str(camera["serial"])
        lens_setting = str(camera["lens_setting"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("Kalibrierungspaket enthält nicht alle Pflichtfelder.") from exc

    if camera_matrix.shape != (3, 3):
        raise CalibrationError("K muss eine 3x3-Matrix sein.")
    if distortion.size < 4:
        raise CalibrationError("Der Distorsionsvektor enthält zu wenige Werte.")
    if len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0:
        raise CalibrationError("Ungültige Kameraauflösung im Kalibrierungspaket.")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise CalibrationError("K oder Distorsion enthält ungültige Zahlen.")

    if require_validation_pass and validation.get("status") != "PASS":
        raise CalibrationError("Das Kalibrierungspaket hat keine bestandene Validierung.")
    if expected_camera_serial is not None and serial != expected_camera_serial:
        raise CalibrationError("Kameraseriennummer stimmt nicht mit dem Kalibrierungspaket überein.")
    if expected_resolution_px is not None and resolution != expected_resolution_px:
        raise CalibrationError("Kameraauflösung stimmt nicht mit dem Kalibrierungspaket überein.")
    if expected_lens_setting is not None and lens_setting != expected_lens_setting:
        raise CalibrationError("Linseneinstellung stimmt nicht mit dem Kalibrierungspaket überein.")

    return LoadedCalibration(
        version=version,
        camera_matrix=camera_matrix,
        distortion=distortion,
        resolution_px=(resolution[0], resolution[1]),
        camera_serial=serial,
        lens_setting=lens_setting,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produktive Kamerakalibrierung"
    )

    parser.add_argument(
        "--calibration-images",
        default="data/calibration_images/*.*", 
        help="Glob für die Kalibrierungsbilder",
    )

    parser.add_argument(
        "--validation-images",
        default="data/validation_images/*.*",
        help="Glob für getrennte Validierungsbilder",
    )

    # Checkerboard: 8 x 6 innere Ecken
    parser.add_argument(
        "--corners-x",
        type=int,
        default=8,
        help="Innere Ecken horizontal",
    )

    parser.add_argument(
        "--corners-y",
        type=int,
        default=6,
        help="Innere Ecken vertikal",
    )

    # 23 mm = 0.023 m
    parser.add_argument(
        "--square-size-m",
        type=float,
        default=0.023,
        help="Kantenlänge eines Quadrats in Metern",
    )

    parser.add_argument(
        "--camera-serial",
        default="4002755623",
        help="Seriennummer der Kamera",
    )

    parser.add_argument(
        "--lens-setting",
        default="fixed-focus",
        help="Verwendete Linseneinstellung",
    )

    parser.add_argument(
        "--mounting-date",
        default="2026-07-27",
        help="Montagedatum im Format YYYY-MM-DD",
    )

    parser.add_argument(
        "--max-validation-rmse-px",
        type=float,
        default=1.0,
        help="Maximal erlaubter Validierungsfehler in Pixeln",
    )

    parser.add_argument(
        "--version",
        default=None,
    )

    parser.add_argument(
        "--approved-by",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration"),
    )

    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=Path("data/validation_output"),
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.max_validation_rmse_px <= 0:
        raise ValueError(
            "--max-validation-rmse-px muss positiv sein."
        )

    # Fotoğraf yollarını bul
    calibration_paths = find_image_paths(args.calibration_images)
    validation_paths = find_image_paths(args.validation_images)

    if not calibration_paths:
        raise CalibrationError(
            f"Keine Kalibrierungsbilder gefunden: {args.calibration_images}"
        )

    if not validation_paths:
        raise CalibrationError(
            f"Keine Validierungsbilder gefunden: {args.validation_images}"
        )

    # Aynı fotoğraf iki grupta kullanılmasın
    calibration_resolved = {
        path.resolve() for path in calibration_paths
    }
    validation_resolved = {
        path.resolve() for path in validation_paths
    }

    overlap = calibration_resolved & validation_resolved

    if overlap:
        raise CalibrationError(
            "Kalibrierungs- und Validierungsbilder "
            "müssen getrennt sein."
        )

    inner_corners = (
        args.corners_x,
        args.corners_y,
    )

    # Checkerboard köşelerini topla
    collection = collect_chessboard_points(
        image_paths=calibration_paths,
        inner_corners=inner_corners,
        square_size_m=args.square_size_m,
    )

    print(
        f"Gefundene Kalibrierungsbilder: "
        f"{len(calibration_paths)}"
    )
    print(
        f"Akzeptierte Bilder: "
        f"{len(collection.accepted_images)}"
    )
    print(
        f"Abgelehnte Bilder: "
        f"{len(collection.rejected_images)}"
    )

    if collection.rejected_images:
        print("Abgelehnte Dateien:")

        for image_path in collection.rejected_images:
            print(f"  - {image_path.name}")

    if len(collection.accepted_images) < 10:
        raise CalibrationError(
            "Zu wenige gültige Kalibrierungsbilder. "
            "Mindestens 10, besser 15–25 verwenden."
        )

    # Asıl kamera kalibrasyonu
    result = calibrate_camera(collection)

    # Ayrı validation fotoğraflarıyla doğrulama
    validation = validate_on_held_out_images(
        image_paths=validation_paths,
        inner_corners=inner_corners,
        square_size_m=args.square_size_m,
        expected_image_size=collection.image_size,
        camera_matrix=result.camera_matrix,
        distortion=result.distortion,
    )

    # Original ve undistorted karşılaştırmalarını kaydet
    comparison_paths = save_undistortion_comparisons(
        image_paths=validation.accepted_images,
        output_directory=args.comparison_output_dir,
        expected_image_size=collection.image_size,
        camera_matrix=result.camera_matrix,
        distortion=result.distortion,
    )

    # Sürümlü JSON paketini oluştur
    version = args.version or create_version(
        args.camera_serial
    )

    package = build_calibration_package(
        version=version,
        camera_serial=args.camera_serial,
        lens_setting=args.lens_setting,
        mounting_date=args.mounting_date,
        inner_corners=inner_corners,
        square_size_m=args.square_size_m,
        collection=collection,
        result=result,
        validation=validation,
        max_validation_rmse_px=(
            args.max_validation_rmse_px
        ),
        approved_by=args.approved_by,
    )

    output_path = save_calibration_package(
        package,
        args.output_dir,
    )

    print("\nKalibrierung abgeschlossen")
    print(
        f"Gültige Kalibrierungsbilder: "
        f"{len(collection.accepted_images)}"
    )
    print(
        f"OpenCV RMS: "
        f"{result.opencv_rms_px:.4f} px"
    )
    print(
        f"Validierungs-RMSE: "
        f"{validation.reprojection_rmse_px:.4f} px"
    )
    print(
        f"Validierungsstatus: "
        f"{package['validation']['status']}"
    )
    print(
        f"Kalibrierungspaket: {output_path}"
    )
    print(
        f"Visuelle Vergleichsbilder: "
        f"{len(comparison_paths)} in "
        f"{args.comparison_output_dir}"
    )


if __name__ == "__main__":
    main()