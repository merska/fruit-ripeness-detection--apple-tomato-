import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread

class FruitDetector:
    def __init__(self):
        self.window_name = "Deteksi Buah Realtime"
        self.cap = None
        self.running = False

        self.fruit_profiles = {
            "MENTAH": np.array([50, 160, 60]),
            "SETENGAH MATANG": np.array([130, 150, 50]),
            "MATANG": np.array([159, 41, 20])
        }

    def match_fruit(self, rgb):
        best_match = "Tidak dikenali"
        best_dist = float("inf")
        for fruit, ref_rgb in self.fruit_profiles.items():
            dist = np.linalg.norm(rgb - ref_rgb)
            if dist < best_dist:
                best_dist = dist
                best_match = fruit
        return best_match if best_dist < 80 else "Tidak dikenali"

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Kesalahan", "Kamera tidak dapat diakses.")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            output = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 7)

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100,
                                       param1=100, param2=30, minRadius=30, maxRadius=150)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :1]:
                    x, y, r = circle
                    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                    mask = np.zeros_like(frame)
                    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                    roi = cv2.bitwise_and(frame, mask)

                    # Perbaikan untuk ROI cropping agar tidak keluar batas
                    h, w = frame.shape[:2]
                    x1 = max(0, x - r)
                    y1 = max(0, y - r)
                    x2 = min(w, x + r)
                    y2 = min(h, y + r)
                    roi_cropped = roi[y1:y2, x1:x2]

                    if roi_cropped.size > 0:
                        roi_rgb = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2RGB)
                        pixels = roi_rgb[np.any(roi_rgb != [0, 0, 0], axis=-1)]
                        if len(pixels) > 0:
                            avg_rgb = np.mean(pixels, axis=0)
                            fruit = self.match_fruit(avg_rgb)
                            text = f"{fruit} | RGB: {avg_rgb.astype(int)}"
                            cv2.putText(output, text, (x - r, y - r - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(self.window_name, output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_detection()

    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

class FruitApp:
    def __init__(self, root):
        self.detector = FruitDetector()
        self.root = root
        self.root.title("Aplikasi Deteksi Buah")
        self.root.geometry("300x150")

        self.btn_start = tk.Button(root, text="Mulai Deteksi", command=self.start_detection)
        self.btn_start.pack(pady=10)

        self.btn_quit = tk.Button(root, text="Keluar", command=self.quit_app)
        self.btn_quit.pack(pady=10)

    def start_detection(self):
        t = Thread(target=self.detector.start_detection)
        t.daemon = True
        t.start()

    def quit_app(self):
        self.detector.stop_detection()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitApp(root)
    root.mainloop()
