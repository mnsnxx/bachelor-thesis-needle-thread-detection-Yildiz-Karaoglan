from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from pyueye import ueye


CAMERA_ID = 1
OUTPUT_DIRECTORY = Path("data/phantom_videos")
REQUESTED_FPS = 30.0


def check_ueye_result(result: int, operation: str) -> None:
    """Stop the program if a uEye API operation fails."""
    if result != ueye.IS_SUCCESS:
        raise RuntimeError(
            f"{operation} ist fehlgeschlagen. "
            f"uEye-Fehlercode: {result}"
        )


def create_video_writer(
    output_directory: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    """Create an MP4 writer; fall back to AVI if MP4 is unavailable."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mp4_path = output_directory / f"phantom_{timestamp}.mp4"
    mp4_codec = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(mp4_path),
        mp4_codec,
        fps,
        frame_size,
    )

    if writer.isOpened():
        return writer, mp4_path

    writer.release()

    avi_path = output_directory / f"phantom_{timestamp}.avi"
    avi_codec = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(avi_path),
        avi_codec,
        fps,
        frame_size,
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "Die Videodatei konnte weder als MP4 "
            "noch als AVI erstellt werden."
        )

    return writer, avi_path


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Camera ID configured in IDS Camera Manager.
    camera = ueye.HIDS(CAMERA_ID)

    image_memory = ueye.c_mem_p()
    memory_id = ueye.INT()
    pitch = ueye.INT()

    bits_per_pixel = ueye.INT(24)
    bytes_per_pixel = 3

    camera_initialized = False
    memory_allocated = False

    video_writer: cv2.VideoWriter | None = None
    current_video_path: Path | None = None

    recording = False
    recording_start_time = 0.0

    try:
        # Initialize the IDS camera.
        result = ueye.is_InitCamera(camera, None)
        check_ueye_result(
            result,
            "Kamera-Initialisierung",
        )
        camera_initialized = True

        # Set a display mode compatible with OpenCV.
        result = ueye.is_SetDisplayMode(
            camera,
            ueye.IS_SET_DM_DIB,
        )
        check_ueye_result(
            result,
            "Display-Modus setzen",
        )

        result = ueye.is_SetColorMode(
            camera,
            ueye.IS_CM_BGR8_PACKED,
        )
        check_ueye_result(
            result,
            "Farbmodus setzen",
        )

        # Read the current camera image dimensions.
        image_area = ueye.IS_RECT()

        result = ueye.is_AOI(
            camera,
            ueye.IS_AOI_IMAGE_GET_AOI,
            image_area,
            ueye.sizeof(image_area),
        )
        check_ueye_result(
            result,
            "Bildgröße lesen",
        )

        width = image_area.s32Width
        height = image_area.s32Height

        frame_width = int(width.value)
        frame_height = int(height.value)
        frame_size = (frame_width, frame_height)

        # Configure the requested camera frame rate.
        requested_fps = ueye.DOUBLE(REQUESTED_FPS)
        actual_fps = ueye.DOUBLE()

        result = ueye.is_SetFrameRate(
            camera,
            requested_fps,
            actual_fps,
        )

        if result == ueye.IS_SUCCESS and actual_fps.value > 0:
            camera_fps = float(actual_fps.value)
        else:
            camera_fps = REQUESTED_FPS
            print(
                "Warnung: Die Kamera-Bildrate konnte nicht "
                "gesetzt werden. 30 FPS werden verwendet."
            )

        # Allocate image memory for camera frames.
        result = ueye.is_AllocImageMem(
            camera,
            width,
            height,
            bits_per_pixel,
            image_memory,
            memory_id,
        )
        check_ueye_result(
            result,
            "Bildspeicher reservieren",
        )
        memory_allocated = True

        result = ueye.is_SetImageMem(
            camera,
            image_memory,
            memory_id,
        )
        check_ueye_result(
            result,
            "Bildspeicher aktivieren",
        )

        result = ueye.is_InquireImageMem(
            camera,
            image_memory,
            memory_id,
            width,
            height,
            bits_per_pixel,
            pitch,
        )
        check_ueye_result(
            result,
            "Bildspeicherinformationen lesen",
        )

        # Start the live camera stream.
        result = ueye.is_CaptureVideo(
            camera,
            ueye.IS_DONT_WAIT,
        )
        check_ueye_result(
            result,
            "Live-Aufnahme starten",
        )

        print(f"IDS-Kamera geöffnet: Kam-ID {CAMERA_ID}")
        print(f"Auflösung: {frame_width} x {frame_height}")
        print(f"Bildrate: {camera_fps:.2f} FPS")
        print("r: Aufnahme starten/stoppen")
        print("q: Programm beenden")

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
                    frame_height,
                    frame_width,
                    bytes_per_pixel,
                ),
            )

            # Ensure a contiguous memory layout for OpenCV VideoWriter.
            frame = np.ascontiguousarray(frame)

            # Do not include the REC overlay in the recorded video.
            if recording and video_writer is not None:
                video_writer.write(frame)

            display_frame = frame.copy()

            if recording:
                elapsed_seconds = (
                    time.monotonic() - recording_start_time
                )

                cv2.putText(
                    display_frame,
                    f"REC {elapsed_seconds:06.1f} s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "r: Aufnahme starten | q: Beenden",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "IDS Phantom-Videoaufnahme",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                if not recording:
                    video_writer, current_video_path = (
                        create_video_writer(
                            output_directory=OUTPUT_DIRECTORY,
                            fps=camera_fps,
                            frame_size=frame_size,
                        )
                    )

                    recording = True
                    recording_start_time = time.monotonic()

                    print(
                        f"Aufnahme gestartet: "
                        f"{current_video_path}"
                    )

                else:
                    recording = False

                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None

                    print(
                        f"Aufnahme gespeichert: "
                        f"{current_video_path}"
                    )

            elif key == ord("q"):
                break

    finally:
        if video_writer is not None:
            video_writer.release()

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