"""Reference-only Canny contour visualisation extracted from the supplied project.

It intentionally has no robot connection and is not imported by application code.
"""

import cv2


def detect_and_draw_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    preview = frame.copy()
    for contour in contours:
        cv2.drawContours(preview, [contour], -1, (0, 255, 0), 1)
        epsilon = 0.03 * cv2.arcLength(contour, True)
        for point in cv2.approxPolyDP(contour, epsilon, True):
            cv2.circle(preview, tuple(point[0]), 3, (255, 0, 255), 1)
    return edges, preview
