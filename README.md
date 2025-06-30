## Mixing Time Analysis

This code analyzes the mixing time in solid suspension experiments by processing video files of the mixing process. The main steps are:

1. **Mask Definition:**  
   The user defines a circular region of interest (ROI) in the first frame of each video (or a saved mask is reused). Only pixels inside this ROI are analyzed.

2. **Frame Averaging:**  
   The code averages a set of frames at the beginning and end of the video to establish baseline and final mixing states.

3. **Frame Sampling:**  
   A subset of frames is sampled throughout the video. For each frame:
   - The selected color channel is extracted.
   - The ROI is applied.
   - The standard deviation and mean of pixel intensities are computed.

4. **Mixing Metrics:**  
   Two main metrics are calculated:
   - **STD (Standard Deviation):** Tracks the variability in pixel intensities over time.
   - **PCT (Percent Completion Threshold):** Measures the progression toward a fully mixed state.

5. **Mixing Time Detection:**  
   - The code detects when the STD curve flattens (indicating mixing completion).
   - It also finds when the PCT curve crosses a user-defined threshold (default: 0.95).
![experiment1_derivative_PCT](https://github.com/user-attachments/assets/99953855-bad0-4b06-b4fb-45a17f66b960) ![experiment1_derivative](https://github.com/user-attachments/assets/e75e9dcc-3903-49f3-8d4d-f44517ba067b) 

6. **Results & Visualization:**  
   - Mixing time results and plots are saved for each video and summarized across all videos.
   - Plots show the evolution of STD and PCT over time, with vertical lines marking the detected mixing times.

**Output:**  
- CSV files with mixing time results  
- PNG plots of STD and PCT analysis

**Parameters:**  
All analysis parameters (frame count, channel, thresholds, etc.) can be adjusted via command-line arguments or the GUI.

## Usage

1. Clone this repository:

    ```sh
    git clone https://github.com/WeheliyeHashi/Mixingtime_ORI.git
    cd Mixingtime_ORI
    ```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

Install dependencies with:

```sh
conda env create -f requirements.yaml
conda activate Mixingtime_code_ORI
pip install -e .
```



2. Run the GUI:

    ```sh
    mt_gui
    ```

3. In the GUI:
    - Browse and select the folder containing your raw videos/images.
    - Set the analysis parameters as needed (e.g., total frames, skip, channel, etc.).
    - Click "Run Processing" to start the analysis.

4. Results will be saved in a `Results` folder next to your raw videos directory.

5. Use the interface to select the necessary files and options, then start the processing.

## File Structure

- `mixingtime_gui.py` - Main GUI application
- `Process_main_images_GUI.py` - Image processing and mixing time analysis. 

## Notes

- Only video files with extensions `.mp4`, `.avi` are processed.

---

For questions or issues, please open an issue on this repository.
