import os

import cv2
import numpy as np
from pyueye import ueye


def check_ueye_result(result: int, operation: str) -> None:
    """Stop the program if an IDS uEye API call fails."""
    if result != ueye.IS_SUCCESS:
        raise RuntimeError(
            f"{operation} ist fehlgeschlagen. uEye-Fehlercode: {result}"
        )


def main() -> None:
    # Camera ID as configured in IDS Camera Manager: "Kam-ID: 1".
    camera = ueye.HIDS(1)

    image_memory = ueye.c_mem_p()
    memory_id = ueye.INT()
    pitch = ueye.INT()

    bits_per_pixel = ueye.INT(24)
    bytes_per_pixel = 3

    camera_initialized = False
    memory_allocated = False

    # Use "data/validation_images" for validation images.
    # Use "data/calibration_images" for calibration images.
    output_directory = "data/calibration_images"
    os.makedirs(output_directory, exist_ok=True)

    existing_images = [
        name
        for name in os.listdir(output_directory)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_counter = len(existing_images) + 1

    try:
        result = ueye.is_InitCamera(camera, None)
        check_ueye_result(result, "Kamera-Initialisierung")
        camera_initialized = True

        result = ueye.is_SetDisplayMode(
            camera,
            ueye.IS_SET_DM_DIB,
        )
        check_ueye_result(result, "Display-Modus setzen")

        # Use a 24-bit color image format compatible with OpenCV BGR.
        result = ueye.is_SetColorMode(
            camera,
            ueye.IS_CM_BGR8_PACKED,
        )
        check_ueye_result(result, "Farbmodus setzen")

        image_area = ueye.IS_RECT()

        result = ueye.is_AOI(
            camera,
            ueye.IS_AOI_IMAGE_GET_AOI,
            image_area,
            ueye.sizeof(image_area),
        )
        check_ueye_result(result, "Bildgröße lesen")

        width = image_area.s32Width
        height = image_area.s32Height

        result = ueye.is_AllocImageMem(
            camera,
            width,
            height,
            bits_per_pixel,
            image_memory,
            memory_id,
        )
        check_ueye_result(result, "Bildspeicher reservieren")
        memory_allocated = True

        result = ueye.is_SetImageMem(
            camera,
            image_memory,
            memory_id,
        )
        check_ueye_result(result, "Bildspeicher aktivieren")

        result = ueye.is_InquireImageMem(
            camera,
            image_memory,
            memory_id,
            width,
            height,
            bits_per_pixel,
            pitch,
        )
        check_ueye_result(result, "Bildspeicherinformationen lesen")

        result = ueye.is_CaptureVideo(
            camera,
            ueye.IS_DONT_WAIT,
        )
        check_ueye_result(result, "Live-Aufnahme starten")

        print(f"Kamera geöffnet: IDS Kam-ID 1")
        print(f"Auflösung: {width.value} x {height.value}")
        print("s: Bild speichern | q: Beenden")

        while True:
            image_data = ueye.get_data(
                image_memory,
                width,
                height,
                bits_per_pixel,
                pitch,
                copy=True,
            )

            frame = np.reshape(
                image_data,
                (
                    height.value,
                    width.value,
                    bytes_per_pixel,
                ),
            )

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                "s: Bild speichern | q: Beenden",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("IDS uEye Kamera", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                filename = os.path.join(
                    output_directory,
                    # Use "validation_" instead of "calibration_"
                    # when capturing validation images.
                    f"calibration_{image_counter:02d}.jpg",
                )

                if not cv2.imwrite(filename, frame):
                    print(f"Bild konnte nicht gespeichert werden: {filename}")
                    continue

                print(f"Bild gespeichert: {filename}")
                image_counter += 1

            elif key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()

        if camera_initialized:
            ueye.is_StopLiveVideo(
                camera,
                ueye.IS_FORCE_VIDEO_STOP,
            )

        if camera_initialized and memory_allocated:
            ueye.is_FreeImageMem(
                camera,
                image_memory,
                memory_id,
            )

        if camera_initialized:
            ueye.is_ExitCamera(camera)


if __name__ == "__main__":
    main()