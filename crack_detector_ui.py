"""AI Heritage Guardian — local YOLOv8n crack detector desktop UI.

Place alongside runs/crack_detector/weights/best.pt in the CrackDetector project.
Requirements: ultralytics, opencv-python, pillow, torch, tkinter (included on Windows).
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "crack_detector" / "weights" / "best.pt"
OUTPUTS = ROOT / "outputs"
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
VIDEO_TYPES = [("Videos", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")]
BG = "#0B1220"
PANEL = "#111E31"
INPUT_BG = "#16253A"
TEXT = "#EBF4FF"
MUTED = "#A0B2C7"
ACCENT = "#32D6BC"


class CrackDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Heritage Guardian | YOLOv8 Nano Crack Detection")
        self.root.geometry("1270x790")
        self.root.minsize(940, 640)
        self.root.configure(bg=BG)
        self.model = None
        self.model_path = DEFAULT_WEIGHTS
        self.capture = None
        self.camera_running = False
        self.camera_after_id = None
        self.job_thread = None
        self.stop_event = threading.Event()
        self.events = queue.Queue(maxsize=40)
        self.source_rgb = None
        self.result_rgb = None
        self.source_photo = None
        self.result_photo = None
        self.last_result_file = None
        self.mode = "idle"
        self.conf = tk.DoubleVar(value=0.25)
        self.status = tk.StringVar(value="Ready — open an image, video, or camera")
        self.stats = tk.StringVar(value="Detections: —    |    Confidence: 25%")
        self.progress_value = tk.DoubleVar(value=0)
        self._build_ui()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        if not self.model_path.is_file():
            self.status.set("Model not found — click 'Select model' and choose best.pt")
        else:
            self.status.set("Model found — ready to detect cracks")

    def _button(self, parent, title, callback, color=INPUT_BG):
        return tk.Button(parent, text=title, command=callback, bg=color, fg=TEXT,
                         activebackground="#234563", activeforeground="white", bd=0,
                         padx=17, pady=12, font=("Segoe UI", 10, "bold"), cursor="hand2")

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=25, pady=(20, 12))
        tk.Label(header, text="AI HERITAGE GUARDIAN", fg=TEXT, bg=BG,
                 font=("Segoe UI", 21, "bold")).pack(anchor="w")
        tk.Label(header, text="Local crack detection  •  YOLOv8 Nano  •  Photos / Videos / Live Webcam",
                 fg=MUTED, bg=BG, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 0))

        toolbar = tk.Frame(self.root, bg=BG)
        toolbar.pack(fill="x", padx=25, pady=(0, 12))
        self._button(toolbar, "Open photo", self.open_photo, ACCENT).pack(side="left", padx=(0, 7))
        self._button(toolbar, "Open video", self.open_video).pack(side="left", padx=(0, 7))
        self._button(toolbar, "Start webcam", self.start_webcam).pack(side="left", padx=(0, 7))
        self._button(toolbar, "Stop", self.stop, "#723949").pack(side="left", padx=(0, 7))
        self._button(toolbar, "Save current frame", self.save_frame).pack(side="left", padx=(0, 7))
        self._button(toolbar, "Select model", self.select_model).pack(side="right")

        settings = tk.Frame(self.root, bg=PANEL, pady=11, padx=13)
        settings.pack(fill="x", padx=25, pady=(0, 12))
        tk.Label(settings, text="Confidence threshold", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        self.conf_label = tk.Label(settings, text="25%", fg=ACCENT, bg=PANEL, width=5,
                                   font=("Segoe UI", 10, "bold"))
        self.conf_label.pack(side="right")
        scale = ttk.Scale(settings, from_=0.05, to=0.85, variable=self.conf,
                          command=lambda _: self.conf_label.config(text=f"{self.conf.get():.0%}"))
        scale.pack(side="left", fill="x", expand=True)

        display = tk.Frame(self.root, bg=BG)
        display.pack(fill="both", expand=True, padx=25)
        display.grid_columnconfigure(0, weight=1, uniform="images")
        display.grid_columnconfigure(1, weight=1, uniform="images")
        display.grid_rowconfigure(0, weight=1)
        self.src_canvas = self._preview_panel(display, 0, "SOURCE")
        self.dst_canvas = self._preview_panel(display, 1, "DETECTIONS")
        self.src_canvas.bind("<Configure>", lambda event: self._render(self.src_canvas, self.source_rgb, "source"))
        self.dst_canvas.bind("<Configure>", lambda event: self._render(self.dst_canvas, self.result_rgb, "result"))

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=25, pady=(12, 20))
        tk.Label(footer, textvariable=self.stats, fg=TEXT, bg=BG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Progressbar(footer, maximum=100, variable=self.progress_value).pack(fill="x", pady=(0, 6))
        tk.Label(footer, textvariable=self.status, fg=MUTED, bg=BG,
                 font=("Segoe UI", 10), wraplength=1200, anchor="w", justify="left").pack(fill="x")

    def _preview_panel(self, parent, col, title):
        frame = tk.Frame(parent, bg=PANEL, padx=9, pady=9)
        frame.grid(row=0, column=col, sticky="nsew", padx=(0, 7) if col == 0 else (7, 0))
        tk.Label(frame, text=title, fg=ACCENT, bg=PANEL,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        canvas = tk.Canvas(frame, bg="#09111B", borderwidth=0, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        return canvas

    def _render(self, canvas, rgb, kind):
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 20 or height < 20:
            return
        if rgb is None:
            canvas.create_text(width // 2, height // 2,
                               text="No image loaded" if kind == "source" else "Detection preview appears here",
                               fill=MUTED, font=("Segoe UI", 12))
            return
        picture = Image.fromarray(rgb)
        picture.thumbnail((width - 14, height - 14), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(picture)
        canvas.create_image(width // 2, height // 2, image=photo, anchor="center")
        if kind == "source":
            self.source_photo = photo
        else:
            self.result_photo = photo

    def _set_images(self, src_bgr, dst_bgr, count=None, conf=None):
        self.source_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        self.result_rgb = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)
        self._render(self.src_canvas, self.source_rgb, "source")
        self._render(self.dst_canvas, self.result_rgb, "result")
        if count is not None:
            self.stats.set(f"Detections in frame: {count}    |    Threshold: {conf:.0%}")

    def _get_model(self):
        if self.model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Weights not found: {self.model_path}")
            self.status.set("Loading model...")
            self.root.update_idletasks()
            self.model = YOLO(str(self.model_path))
        return self.model

    def _predict(self, src_bgr, conf=None):
        if conf is None:
            conf = self.conf.get()
        result = self._get_model().predict(src_bgr, conf=conf, imgsz=416,
                                           device="cpu", verbose=False)[0]
        count = len(result.boxes) if result.boxes is not None else 0
        annotated = result.plot()
        return annotated, count

    def _busy(self):
        if self.job_thread is not None and self.job_thread.is_alive():
            messagebox.showinfo("Processing", "Stop the current video process first.")
            return True
        return False

    def select_model(self):
        if self._busy():
            return
        path = filedialog.askopenfilename(title="Choose trained YOLO weights", filetypes=[("PyTorch weights", "*.pt")])
        if path:
            self.stop()
            self.model_path = Path(path)
            self.model = None
            self.status.set(f"Selected weights: {self.model_path.name}")

    def open_photo(self):
        if self._busy():
            return
        path = filedialog.askopenfilename(title="Open image", filetypes=IMAGE_TYPES)
        if not path:
            return
        self.stop()
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("OpenCV could not open this image.")
            annotated, count = self._predict(image)
            self._set_images(image, annotated, count, self.conf.get())
            self.mode = "photo"
            self.progress_value.set(100)
            self.status.set(f"Photo processed: {Path(path).name} — {count} detection(s)")
        except Exception as exc:
            messagebox.showerror("Photo detection failed", str(exc))
            self.status.set("Photo processing failed")

    def start_webcam(self):
        if self._busy():
            return
        self.stop()
        try:
            self._get_model()
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("Could not open webcam. Check Windows camera permissions and other camera apps.")
            self.camera_running = True
            self.mode = "camera"
            self.progress_value.set(0)
            self.status.set("Webcam active — processing frames on CPU")
            self._camera_tick()
        except Exception as exc:
            self.stop()
            messagebox.showerror("Webcam error", str(exc))

    def _camera_tick(self):
        if not self.camera_running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if ok:
            try:
                conf = self.conf.get()
                annotated, count = self._predict(frame, conf)
                self._set_images(frame, annotated, count, conf)
            except Exception as exc:
                self.stop()
                messagebox.showerror("Inference error", str(exc))
                return
        else:
            self.status.set("Webcam returned no frame — trying again")
        if self.camera_running:
            self.camera_after_id = self.root.after(50, self._camera_tick)

    def open_video(self):
        if self._busy():
            return
        path = filedialog.askopenfilename(title="Open video", filetypes=VIDEO_TYPES)
        if not path:
            return
        self.stop()
        try:
            self._get_model()
        except Exception as exc:
            messagebox.showerror("Model error", str(exc))
            return
        self.mode = "video"
        self.stop_event.clear()
        self.progress_value.set(0)
        self.status.set(f"Processing video: {Path(path).name} — output will be saved in outputs/")
        conf = float(self.conf.get())  # Freeze threshold for the entire video
        self.job_thread = threading.Thread(target=self._video_worker, args=(path, conf), daemon=True)
        self.job_thread.start()

    def _emit(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass  # Drop preview frames instead of slowing down video processing

    def _video_worker(self, path, conf):
        cap = cv2.VideoCapture(path)
        writer = None
        output_path = None
        try:
            if not cap.isOpened():
                raise RuntimeError("Unable to open the selected video.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 240:
                fps = 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width < 1 or height < 1:
                raise RuntimeError("Invalid video dimensions")
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUTS / f"{Path(path).stem}_detected_{stamp}.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Could not create output MP4. Check disk space and codec support.")
            frame_index = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                # The model is used only by one mode at a time.
                annotated, count = self._predict(frame, conf)
                writer.write(annotated)
                frame_index += 1
                if frame_index == 1 or frame_index % 10 == 0:
                    self._emit(("preview", frame.copy(), annotated.copy(), count, conf,
                                round(100 * frame_index / total, 1) if total else 0,
                                frame_index, total))
            stopped = self.stop_event.is_set()
            self._emit(("done", str(output_path), frame_index, stopped))
        except Exception as exc:
            self._emit(("error", str(exc)))
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "preview":
                    _, src, dst, count, conf, pct, idx, total = event
                    self._set_images(src, dst, count, conf)
                    self.progress_value.set(pct)
                    self.status.set(f"Video processing: {idx}/{total if total else '?'} frames — {pct:.1f}%")
                elif kind == "done":
                    _, path, frames, stopped = event
                    self.last_result_file = Path(path)
                    self.progress_value.set(0 if stopped else 100)
                    self.status.set(("Stopped — partial video: " if stopped else "Finished — saved video: ") + path)
                    if not stopped:
                        messagebox.showinfo("Video processed", f"Processed {frames} frames.\nSaved to:\n{path}\n\nNote: output video has no audio.")
                elif kind == "error":
                    self.status.set("Video processing failed")
                    messagebox.showerror("Video error", event[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def save_frame(self):
        if self.result_rgb is None:
            messagebox.showinfo("Nothing to save", "Open an image or run the webcam first.")
            return
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(title="Save annotated frame", initialdir=str(OUTPUTS),
                                            initialfile=f"crack_detection_{stamp}.png",
                                            defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not path:
            return
        Image.fromarray(self.result_rgb).save(path)
        self.status.set(f"Saved annotated frame: {path}")

    def stop(self):
        self.camera_running = False
        self.stop_event.set()
        if self.camera_after_id:
            try:
                self.root.after_cancel(self.camera_after_id)
            except tk.TclError:
                pass
            self.camera_after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.job_thread and self.job_thread.is_alive():
            self.status.set("Stopping video processing after current frame...")
        elif self.mode == "camera":
            self.status.set("Webcam stopped")
        self.mode = "idle"

    def _close(self):
        self.stop()
        if self.job_thread and self.job_thread.is_alive():
            messagebox.showinfo("Finishing", "Video processing is stopping. Close the app again once it finishes.")
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    CrackDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
