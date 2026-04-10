import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Utility Suite Pro")
        self.root.geometry("850x650")
        self.root.configure(bg="#0f172a") # Consistently using the base dark theme
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#0f172a")
        
        # Styling buttons to match the electric accents requested
        self.style.configure("TButton", padding=10, relief="flat", background="#3b82f6", foreground="#ffffff", font=("Outfit", 10, "bold"))
        self.style.map("TButton", background=[("active", "#60a5fa")])
        self.style.configure("TLabel", background="#0f172a", foreground="#f8fafc", font=("Outfit", 12))
        
        self.current_image_path = None
        self.cv_image = None
        self.display_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header Area
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", pady=20)
        
        title = ttk.Label(header_frame, text="Image Utility Suite", font=("Outfit", 26, "bold"), foreground="#6ee7b7")
        title.pack()
        subtitle = ttk.Label(header_frame, text="Manipulate, Watermark, and Export to PDF", font=("Outfit", 12), foreground="#94a3b8")
        subtitle.pack()
        
        # Main Work Area
        self.work_frame = ttk.Frame(self.root)
        self.work_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left Panel (Controls) -> Acting as the "Surface Card"
        control_frame = tk.Frame(self.work_frame, bg="#1e293b", padx=20, pady=20, bd=0)
        control_frame.pack(side="left", fill="y", padx=(0, 20))
        
        ttk.Button(control_frame, text="Open Image", command=self.open_image).pack(fill="x", pady=10)
        
        # Separator / Label
        tk.Label(control_frame, text="Enhancements", bg="#1e293b", fg="#6ee7b7", font=("Outfit", 10, "bold")).pack(pady=(20, 5))
        ttk.Button(control_frame, text="Apply Grayscale", command=self.to_grayscale).pack(fill="x", pady=5)
        ttk.Button(control_frame, text="Edge Detection", command=self.edge_detection).pack(fill="x", pady=5)
        
        tk.Label(control_frame, text="Utilities", bg="#1e293b", fg="#6ee7b7", font=("Outfit", 10, "bold")).pack(pady=(20, 5))
        ttk.Button(control_frame, text="Add Watermark", command=self.add_watermark).pack(fill="x", pady=5)
        ttk.Button(control_frame, text="Export to PDF", command=self.export_to_pdf).pack(fill="x", pady=5)
        ttk.Button(control_frame, text="Save Image", command=self.save_image).pack(fill="x", pady=5)
        
        # Right Panel (Image Display)
        self.canvas_frame = tk.Frame(self.work_frame, bg="#000000", bd=2, relief="groove", highlightbackground="#334155")
        self.canvas_frame.pack(side="right", fill="both", expand=True)
        
        self.image_label = tk.Label(self.canvas_frame, bg="#000000", text="[ No Image Loaded ]", fg="#94a3b8", font=("Outfit", 14))
        self.image_label.pack(expand=True, fill="both")
        
        self.root.bind("<Configure>", self.on_resize)
        
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.current_image_path = file_path
            self.cv_image = cv2.imread(file_path)
            self.update_display()
            
    def on_resize(self, event):
        # Triggers visual update to fit canvas when resized
        if event.widget == self.root:
            self.update_display()

    def update_display(self):
        if self.cv_image is not None:
            # Convert BGR to RGB for tkinter
            rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            
            # Dynamic resizing based on canvas available space
            self.canvas_frame.update()
            canvas_width = self.canvas_frame.winfo_width()
            canvas_height = self.canvas_frame.winfo_height()
            
            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 500, 500 
                
            pil_img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.display_image, text="")
            
    def to_grayscale(self):
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.update_display()
            
    def edge_detection(self):
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian Blur first to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            self.cv_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.update_display()
            
    def add_watermark(self):
        if self.cv_image is not None:
            text = "PROPERTY OF AUTHOR"
            color = (255, 255, 255) # White watermark
            font = cv2.FONT_HERSHEY_DUPLEX
            
            height, width = self.cv_image.shape[:2]
            scale = width / 600.0
            thickness = int(max(1, scale * 2))
            
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = int((height + text_size[1]) / 2)
            
            # Create a semi-transparent overlay
            overlay = self.cv_image.copy()
            cv2.putText(overlay, text, (text_x, text_y), font, scale, color, thickness)
            # Add weighted blends them
            cv2.addWeighted(overlay, 0.4, self.cv_image, 0.6, 0, self.cv_image)
            
            self.update_display()
            
    def save_image(self):
        if self.cv_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG file", "*.png"), ("JPG file", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, self.cv_image)
                messagebox.showinfo("Success", "Image saved successfully.")
                
    def export_to_pdf(self):
        if self.cv_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF file", "*.pdf")])
            if file_path:
                rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_image)
                pil_img.save(file_path, "PDF", resolution=100.0)
                messagebox.showinfo("Success", f"Image exported to {os.path.basename(file_path)} effectively!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
