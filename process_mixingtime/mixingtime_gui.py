# %%
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for compatibility with Tkinter
from tkinter import (
    Tk,
    Frame,
    Label,
    Entry,
    Button,
    Checkbutton,
    IntVar,
    StringVar,
    OptionMenu,
    filedialog,
    messagebox,
    Radiobutton,
)

import process_mixingtime as pm

class VideoProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("Video Processing GUI")
        master.minsize(600, 400)  # Set a minimum window size

        self.frame = Frame(master, width=550, height=350)
        self.frame.pack(padx=30, pady=30, fill="both", expand=True)

        self.raw_videos_path = StringVar()
        self.use_same_mask = IntVar(value=1)
        self.total_frames = StringVar(value="271")
        self.channel = StringVar(value="1")
        self.span = StringVar(value="150")
        self.spanderivative = StringVar(value="150")
        self.threshstd = StringVar(value="0.025")
        self.newthresh = StringVar(value="0.95")
        self.injs = StringVar(value="10")
        self.gg = StringVar(value="30")
        self.ff = StringVar(value="900")
        # self.skip = StringVar(value="100")  # Removed skip variable

        Label(self.frame, text="Select Raw Videos Folder:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.raw_videos_path, width=40).grid(
            row=0, column=1, sticky="w", pady=12, padx=8
        )
        Button(self.frame, text="Browse", command=self.browse_videos).grid(
            row=0, column=2, padx=(8, 0), pady=12
        )

        # Mixing option radio buttons (moved here)
        self.mixing_option = StringVar(value="rock")
        Label(self.frame, text="Mixing option:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", pady=12, padx=8
        )
        Radiobutton(self.frame, text="Rock",font=("Arial", 8, "bold"), variable=self.mixing_option, value="rock").grid(
            row=1, column=1, sticky="w", pady=12, padx=2
        )
        Radiobutton(self.frame, text="Compression",  font=("Arial", 8, "bold"), variable=self.mixing_option, value="compression").grid(
            row=1, column=2, sticky="w", pady=12, padx=2
        )

        Label(self.frame, text="Use Same Mask for each day:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky="w", pady=12, padx=8
        )
        Checkbutton(self.frame, variable=self.use_same_mask).grid(
            row=2, column=1, sticky="w", pady=12, padx=8
        )

        Label(self.frame, text="Total Frames to Analyze:").grid(
            row=3, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.total_frames).grid(
            row=3, column=1, pady=12, padx=8
        )
        # Removed Skip label and entry
        # Label(self.frame, text="Skip:").grid(
        #     row=2, column=2, sticky="w", pady=12, padx=8
        # )
        # Entry(self.frame, textvariable=self.skip, width=10).grid(
        #     row=2, column=3, pady=12, padx=8
        # )

        # Add StringVars for GG and FF time display
        self.gg_time_text = StringVar(value="Equivalent to 1.0s")
        self.ff_time_text = StringVar(value="Equivalent to 30.0s")

        # GG and FF underneath Total Frames/Skip
        Label(self.frame, text="Frames to Avg (Start):").grid(
            row=4, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.gg).grid(row=4, column=1, pady=12, padx=8)
        Entry(
            self.frame,
            textvariable=self.gg_time_text,
            state="readonly",
            width=22,
            fg="blue",
        ).grid(row=4, column=2, padx=8, pady=12)

        Label(self.frame, text="Frames to Avg (End):").grid(
            row=5, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.ff).grid(row=5, column=1, pady=12, padx=8)
        Entry(
            self.frame,
            textvariable=self.ff_time_text,
            state="readonly",
            width=22,
            fg="blue",
        ).grid(row=5, column=2, padx=8, pady=12)

        Label(self.frame, text="Select Channel (0: R, 1: G, 2: B):").grid(
            row=6, column=0, sticky="w", pady=12, padx=8
        )
        OptionMenu(self.frame, self.channel, "0", "1", "2").grid(
            row=6, column=1, pady=12, padx=8
        )

        Label(self.frame, text="Smoothing Span:").grid(
            row=7, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.span).grid(row=7, column=1, pady=12, padx=8)

        Label(self.frame, text="Derivative Span:").grid(
            row=8, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.spanderivative).grid(
            row=8, column=1, pady=12, padx=8
        )

        Label(self.frame, text="Threshold STD:").grid(
            row=9, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.threshstd).grid(
            row=9, column=1, pady=12, padx=8
        )

        Label(self.frame, text="Threshold PCT:").grid(
            row=10, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.newthresh).grid(
            row=10, column=1, pady=12, padx=8
        )

        Label(self.frame, text="Injection time in seconds:", ).grid(
            row=11, column=0, sticky="w", pady=12, padx=8
        )
        Entry(self.frame, textvariable=self.injs).grid(
            row=11, column=1, pady=12, padx=8
        )

        Button(self.frame, text="Run Processing", command=self.run_processing).grid(
            row=12, columnspan=3, pady=24
        )

        # --- Save Normalised Plots Box ---
        self.save_box = Frame(self.frame, relief="groove", borderwidth=2, width=260, height=120)
        self.save_box.grid(row=7, column=2, rowspan=5, sticky="n", padx=(18,0), pady=8)

        Label(self.save_box, text="Save Normalised Plots", font=("Arial", 10, "bold")).pack(anchor="w", padx=8, pady=(8,4))

        self.save_raw_images = IntVar(value=0)  # Default unticked
        Checkbutton(self.save_box, text="Save raw images", variable=self.save_raw_images).pack(anchor="w", padx=8, pady=(0,6))

        Label(self.save_box, text="Skip time steps (s):").pack(anchor="w", padx=8)
        self.skip_time_steps = StringVar(value="10")
        Entry(self.save_box, textvariable=self.skip_time_steps, width=10).pack(anchor="w", padx=8)

    def browse_videos(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.raw_videos_path.set(folder_selected)
           

    def run_processing(self):
        try:
            #print(self.skip_time_steps.get(), self.use_same_mask.get(), self.save_raw_images.get())  # Debug print to check skip_time_steps value
            pm.main_processor(
                self.raw_videos_path.get(),
                self.mixing_option.get(),
                self.save_raw_images.get(),
                self.use_same_mask.get(),
                int(self.total_frames.get()),
                int(self.channel.get()),
                int(self.span.get()),
                int(self.spanderivative.get()),
                float(self.threshstd.get()),
                float(self.newthresh.get()),
                int(self.injs.get()),
                int(self.gg.get()),
                int(self.ff.get()),
                int(self.skip_time_steps.get()),  # Use skip_time_steps from the entry
                # int(self.skip.get()),  # Removed skip argument
            )
            messagebox.showinfo("Success", "Video processing completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Removed update_duration method

    def update_gg_time(self):
        try:
            frames = int(self.gg.get())
            total = frames / 30
            minutes = int(total // 60)
            seconds = total % 60
            if minutes > 0:
                self.gg_time_text.set(f"Equivalent to {minutes}m {seconds:.1f}s")
            else:
                self.gg_time_text.set(f"Equivalent to {seconds:.1f}s")
        except Exception:
            self.gg_time_text.set("Equivalent to -")

    def update_ff_time(self):
        try:
            frames = int(self.ff.get())
            total = frames / 30
            minutes = int(total // 60)
            seconds = total % 60
            if minutes > 0:
                self.ff_time_text.set(f"Equivalent to {minutes}m {seconds:.1f}s")
            else:
                self.ff_time_text.set(f"Equivalent to {seconds:.1f}s")
        except Exception:
            self.ff_time_text.set("Equivalent to -")


def main():
    root = Tk()
    app = VideoProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# %%
