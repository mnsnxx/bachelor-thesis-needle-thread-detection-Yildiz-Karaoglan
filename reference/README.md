# Referenzquellen

Die in diesem Verzeichnis enthaltenen Dateien sind ausgewählte technische Referenzen aus bestehenden Projektarchiven. Sie stellen keine produktiven Module dar und enthalten keine direkte Robotersteuerung.

- `camera_calibration_reference.py`: Beispiel für die intrinsische Kamerakalibrierung mit OpenCV auf Basis detektierter Schachbrettecken sowie für die anschließende Bildentzerrung.

- `canny_edge_reference.py`: Prototyp zur Canny-Kantendetektion und Konturvisualisierung auf Live-Kamerabildern.

Das Kalibrierungsmodul orientiert sich hinsichtlich des Ein- und Ausgabeverhaltens an der Funktion `calibration(...)` der Datei `camera_calibration_reference.py`. Die eigentliche Implementierung erfolgt jedoch in einem eigenständigen Modul. Die erzeugten Kalibrierungsergebnisse werden im versionierten Paketformat unter `calibration/` gespeichert.

Die Datei `canny_edge_reference.py` dient ausschließlich als technische Referenz für grundlegende Bildaufnahme, Kantendetektion und Visualisierung. Sie stellt keine zuverlässige Objekterkennung für Wunden, chirurgische Nadeln oder chirurgische Fäden dar.
