# %%
#import matplotlib
%matplotlib qt
#matplotlib.use('QtAgg')  # or 'QtAgg' in newer versions
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
#from Reader.readVideomp4 import readVideomp4
import process_mixingtime as pm
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse
# %%

# ------- Define functions -------

def fit_circle_least_squares(pts):
    # pts: list of (x, y)
    pts = np.array(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center_x, center_y = c[0], c[1]
    radius = np.sqrt(c[2] + center_x**2 + center_y**2)
    return center_x, center_y, radius

def _return_masked_image(image, pts):
    center_x, center_y, radius = fit_circle_least_squares(pts)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    X, Y = np.meshgrid(x, y)
    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2  # inside is True
    return mask.astype(np.uint8)  # 1 inside, 0 outside

def selectVideoReader(params_input: str):
    video_file = params_input
    # open video to read
    

    isMP4video = video_file.endswith(".mp4")
    isAVIvideo = video_file.endswith(".avi")

    if isMP4video or isAVIvideo:
        # use opencv VideoCapture
        vid = pm.readVideomp4(video_file)
    else:
        raise Exception("Only *.mp4 and *.avi videos are supported so far")

    if vid.width == 0 or vid.height == 0:
        raise RuntimeError

    return vid

def average_frames(store, indices, channel, mask, filter_sigma):
    frames = []
    for i, t in enumerate(indices):
       # print(f"Processing frame {i+1}/{len(indices)}")
        frame = store.read_frame(int(t * store.fps))[1]
        if frame is None:
            print(f"Frame {i+1} could not be read.")
            continue
        frameg = frame[:, :, channel]
        doub = np.flipud(frameg.astype(float))
        doub[mask == 0] = np.nan
        frames.append(doub)
    if frames:
        M = np.nanmean(frames, axis=0)
        M = gaussian_filter(M, sigma=filter_sigma)
        return M
    else:
        return None
    

def analyze_mixing_time(
    video_path,
    mask_path=None,
    totalframes2analyze=271,
    channel=1,
    span=150,
    spanderivative=150,
    threshstd=0.025,
    newthresh=0.95,
    injs=10,
    GG=30,
    FF=900,
    skip=100,
    Filter=2,
    consecutivePoints=5,
    fromend=1,
    frombeg=1,
    
):
    """
    Analyze mixing time from a video file."""
  
    
    
    store = selectVideoReader(video_path)
    frame = store.read()[-1]
    # if the there arent any points defined, ask the user to click on the image
    use_existing_mask = False
    try:
        if mask_path.stat().st_size > 0:
            pts = np.load(mask_path)
            if pts.size > 0:
                use_existing_mask = True
            else:
                print(f"Warning: Loaded mask from {mask_path} is empty.")
        else:
            print(f"Warning: Mask file {mask_path} exists but is empty.")
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
    if use_existing_mask:
         print(f"Using cached mask for {Path(video_path).name}")
    else:
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='gray')
        ax.set_title("Click points to define mask area (Press Enter when done)")
        plt.axis("off")
        pts = plt.ginput(n=-1, timeout=0)
        plt.close()
        np.save(mask_path, pts)
    mask = _return_masked_image(frame, pts)

    fps = np.ceil(store.fps)
    totalframes = int(store.tot_frames)

    indicesi = [(frombeg + i) / fps for i in range(GG)]
    M0 = average_frames(store, indicesi, channel, mask, Filter)

    indicesfx = [((totalframes - FF - fromend + i) / fps) for i in range(FF)]
    indicesf = indicesfx[::10]
    M1 = average_frames(store, indicesf, channel, mask, Filter)

    totalpx = np.sum(~np.isnan(M0))
    AA = round(totalframes / totalframes2analyze)
    print(AA)
    times = [(i * AA) / fps for i in range(int(totalframes / AA) - 1)]
    #times = [(i*skip) / fps for i in range(totalframes2analyze)]
    #store = selectVideoReader(video_path)
    zGmean, zSTDM = [], []

    for t in tqdm(times, desc="Processing frames", total=len(times)):
        frame = store.read_frame(int(t * fps))[1]
        if frame is None:
            break
        frame = frame.astype(float)
        channels = [np.flipud(frame[:, :, i]) for i in [2, channel, 0]]
        for arr in channels:
            arr[mask == 0] = np.nan
        M = gaussian_filter(channels[1], sigma=Filter)
        G = 1 - (M - M1) / (M0 - M1)
        G[~np.isfinite(G)] = np.nan
        zGmean.append(np.nanmean(G))
        zSTDM.append(np.nanstd(M))
    store.release()

    smoothSTD = savgol_filter(zSTDM, span, 3, mode='interp')
    derivative = savgol_filter(np.gradient(smoothSTD), spanderivative, 3, mode='interp')
    flatIndices = np.where(np.abs(derivative) <= threshstd)[0]
    zallindices = len(derivative)
    flatIndices = flatIndices[flatIndices >= int(0.4 * zallindices)]
    flatIndex = None
    for i in range(len(flatIndices) - consecutivePoints + 1):
        if np.all(np.diff(flatIndices[i:i + consecutivePoints]) == 1):
            flatIndex = flatIndices[i]
            break
    empty = int(flatIndex is None)
    smooth_zGmean = savgol_filter(zGmean, span, 2, mode='interp')
    fmlpct = np.argmax(smooth_zGmean > newthresh) if np.any(smooth_zGmean > newthresh) else None
    empty1 = int(fmlpct is None)
    fml = flatIndex
    tm = times[fml] - injs if not empty else np.nan
    tmpct = times[fmlpct] - injs if not empty1 else np.nan

    return smoothSTD, smooth_zGmean, tm, tmpct, np.array(times) - injs


   
# Plot derivative
def _plot_results(results_path, records, newthresh=0.95):
    # Plot all STD curves in one figure
    tm_list , tmpct_list = [], []
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.box(True)
    plt.title('Mixing Time Analysis using STD')
    plt.xlabel('time [s]')
    plt.ylabel('std(G)')
    for record in records:
        #video_name = record['video']
        smoothSTD = record['smoothSTD']
        tm = record['tm']
        times = record['time']
        [line]=plt.plot(times, smoothSTD, linewidth=1)
        color = line.get_color() 
        if not (tm is None or np.isnan(tm)):
            plt.axvline(tm, linestyle='--', linewidth=1, color=color)
        tm_list.append(tm)
    tmc_mean, tmc_std  = np.nanmean(tm_list), np.nanstd(tm_list)
    plt.legend(loc='upper right', fontsize=14, title=f'Mean $t_m$ = {tmc_mean:.1f}s,\n STD = {tmc_std:.1f}s')
    plt.tight_layout()
    result_file_std = results_path / "all_videos_std_mixing_time.png"
    plt.savefig(result_file_std)
   # print(f"All STD results saved to {result_file_std}")

    # Plot all PCT curves in one figure
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.box(True)
    plt.title('PCT Analysis')
    plt.xlabel('time [s]')
    plt.ylabel('PCT [-]')
    for record in records:
       # video_name = record['video']
        smooth_zGmean = record['smooth_zGmean']
        tmpct = record['tmpct']
        times = record['time']
        [line] = plt.plot(times, smooth_zGmean, linewidth=1)
        color = line.get_color()
        if not (tmpct is None or np.isnan(tmpct)):
            plt.axvline(tmpct, linestyle='--', linewidth=1, color=color)
        tmpct_list.append(tmpct)
    plt.axhline(newthresh, linestyle='--', color='k', linewidth=1, label=f'threshold = {newthresh:.2f}')
    tmpct_mean, tmpct_std = np.nanmean(tmpct_list), np.nanstd(tmpct_list)
    plt.legend(loc='lower right', fontsize=14, title=f'Mean $t_m$ = {tmpct_mean:.1f}s,\n STD = {tmpct_std:.1f}s')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    result_file_pct = results_path / "all_videos_pct_analysis.png"
    plt.savefig(result_file_pct)
   # print(f"All PCT results saved to {result_file_pct}")

    plt.close('all')  # Close all figures to free memory

    return tmc_mean, tmc_std, tmpct_mean, tmpct_std

#%%

rawvideos_path= r'C:\Users\WeheliyeWeheliye\OneDrive - Oribiotech Ltd\Desktop\Ori_Weheliye\Mixing_code\Mixing_time\Data\RawVideos'
Use_same_mask=True
totalframes2analyze=271
channel=1
span=150
spanderivative=150                   
threshstd=0.025
newthresh=0.95
injs=10
GG=30 
FF=900
skip=100

    
rawvideos_path = Path(rawvideos_path).resolve()
results_path_global = Path(str(rawvideos_path).replace('RawVideos', 'Results').replace('\\', '/'))

if not rawvideos_path.exists():
    raise FileNotFoundError(f"Raw videos path '{rawvideos_path}' does not exist.")

# Get all "leaf" subfolders (no subdirectories)
subfolders = [
f for f in rawvideos_path.rglob('*') 
if f.is_dir() and not any(child.is_dir() for child in f.iterdir())
]
valid_extensions = ['.mp4', '.avi']
local_records, global_records = [], []

#%%
# Main processing loop

for subfolder in tqdm(subfolders, desc="Processing subfolders", unit="subfolder", total=len(subfolders)):
    video_files = [f for f in subfolder.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    results_path = Path(str(subfolder).replace('RawVideos', 'Results').replace('\\', '/'))
    results_path.mkdir(exist_ok=True, parents=True)
    mask_path = Path(str(subfolder).replace('RawVideos', 'Masks').replace('\\', '/'))
    mask_path.mkdir(exist_ok=True, parents=True)

    # Determine mask file to use for this subfolder
    mask_file = None
    if Use_same_mask:
        found_masks = list(mask_path.parent.rglob('*_mask.npy'))
        if found_masks:
            mask_file = found_masks[0]
            print(f"Using existing mask: {mask_file}")
        else:
            # Use a default mask name for the subfolder
            mask_file = mask_path / (subfolder.name + "_mask.npy")
    local_records = []  # Reset for each subfolder

    for video_file in video_files:
        try:
            if Use_same_mask:
                current_mask = mask_file
            else:
                current_mask = mask_path / (video_file.stem + "_mask.npy")
            results = analyze_mixing_time(str(video_file).replace('\\', '/'), 
                                            current_mask,
                                            totalframes2analyze,
                                            channel,
                                            span,
                                            spanderivative,
                                            threshstd,
                                            newthresh,
                                            injs,
                                            GG,
                                            FF,
                                            skip,
                                            )
            smoothSTD, smooth_zGmean, tm, tmpct, time = results
            local_records.append({
                'video': video_file.name,
                'tm': tm,
                'tmpct': tmpct,
                'smoothSTD': smoothSTD,
                'smooth_zGmean': smooth_zGmean,
                'time': time
            })
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            break

    if local_records:
        tmc_mean, tmc_std, tmpct_mean, tmpct_std = _plot_results(results_path, local_records)
        df = pd.DataFrame.from_records(local_records)
        df = df.explode(['smoothSTD', 'smooth_zGmean', 'time'])
        df.to_csv(results_path / 'mixing_time_results.csv', index=False)
        global_records.append({
            'date': subfolder.parent.stem,
            'folder': subfolder.name,
            'tmc_mean': tmc_mean,
            'tmc_std': tmc_std,
            'tmpct_mean': tmpct_mean,
            'tmpct_std': tmpct_std
        })
        # Save global results after each subfolder for robustness
        df_global = pd.DataFrame.from_records(global_records)
        df_global.to_csv(results_path_global / 'global_mixing_time_results.csv', index=False)


# %%
