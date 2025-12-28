
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import mask_utils
import threading
import time

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class MaskApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Mask Detection System")
        self.geometry(f"{800}x{600}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Mask Detector", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, rowspan=4, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Video Label
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Status Label
        self.status_label = ctk.CTkLabel(self.main_frame, text="Status: Initializing...", font=ctk.CTkFont(size=24, weight="bold"))
        self.status_label.pack(pady=10)

        # Load models
        prototxtPath = r"face_detector/deploy.prototxt"
        weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        maskModelPath = r"mask_detector.keras"
        
        try:
            self.detector = mask_utils.MaskDetector(prototxtPath, weightsPath, maskModelPath)
        except Exception as e:
            self.status_label.configure(text=f"Error loading models: {e}", text_color="red")
            return

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        
        self.update_frame()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize frame for GUI to speed up resizing in CV too 
            frame = cv2.resize(frame, (640, 480))
            
            # Detect masks
            (locs, preds) = self.detector.detect_and_predict_mask(frame)

            # Loop over detections to draw and update status
            status_text = "No Face Detected"
            status_color = "gray"

            if len(locs) > 0:
                # Prioritize "No Mask" warning if ANYONE is not wearing a mask
                any_no_mask = False
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    
                    if label == "No Mask":
                        any_no_mask = True

                    # Draw on frame (BGR for OpenCV)
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                if any_no_mask:
                    status_text = "WARNING: Please wear a mask!"
                    status_color = "red"
                else:
                    status_text = "Eligible to Pass"
                    status_color = "green"

            self.status_label.configure(text=status_text, text_color=status_color)

            # Convert to PIL Image for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = MaskApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
