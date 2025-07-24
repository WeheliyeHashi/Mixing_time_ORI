# %%
#import matplotlib

#matplotlib.use('QtAgg')  # or 'QtAgg' in newer versions
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
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

def _return_masked_image(image, pts, operating_system='rock', scale=2.9):
    center_x, center_y, radius = fit_circle_least_squares(pts)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    X, Y = np.meshgrid(x, y)
    if operating_system == 'compression':
        # 1 in the ring between large and small circle, 0 elsewhere
        mask_large = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
        mask_small = (X - center_x)**2 + (Y - center_y)**2 < (radius/scale)**2
        mask = mask_large & (~mask_small)
       
    else:  # 'rock'
        # 1 inside large circle, 0 outside
        mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
    return mask.astype(np.uint8)  # 1 inside, 0 outside

def selectVideoReader(params_input: str):
    video_file = params_input
    # open video to read
    
    
    isMP4video = video_file.lower().endswith(".mp4")
    isAVIvideo = video_file.lower().endswith(".avi")

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
    operating_system='rock',
    save_plots=False,
    Figures_path=None,
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
   # mask = _return_masked_image(frame, pts)
    mask = _return_masked_image(frame, pts, operating_system=operating_system, scale=2.8)
    if operating_system == 'compression':
        pts_np = np.array(pts, dtype=np.int32)
        pts_np = pts_np.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts_np], 0)
    mask = np.flipud(mask)

    fps = np.ceil(store.fps)
    totalframes = int(store.tot_frames)

    indicesi = [(frombeg + i) / fps for i in range(GG)]
    M0 = average_frames(store, indicesi, channel, mask, Filter)

    indicesfx = [((totalframes - FF - fromend + i) / fps) for i in range(FF)]
    indicesf = indicesfx[::10]
    M1 = average_frames(store, indicesf, channel, mask, Filter)

    # totalpx = np.sum(~np.isnan(M0))
    AA = round(totalframes / totalframes2analyze)
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
        if save_plots and t% skip == 0:
            plt.figure(figsize=(8, 5))
            plt.imshow(G, cmap='jet', vmin=0, vmax=1)
            plt.title(f"Frame at {t:.2f} seconds")
            cbar = plt.colorbar(label='G value')
            #cbar.set_clim(0, 1)  # Set colorbar limits from 0 to 1
            cbar.mappable.set_clim(0, 1)  # Correct way to set colorbar limits           
            plt.axis('off')
            plt.savefig(Figures_path / f"frame_{int(t)}.png")
            plt.close()
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

    return zSTDM, zGmean, smoothSTD, smooth_zGmean, tm, tmpct, np.array(times) - injs


   
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
        zSTDM = record['zSTDM']
        smoothSTD = record['smoothSTD']
        tm = record['tm']
        times = record['time']
        [line]=plt.plot(times, smoothSTD, linewidth=1)
        color = line.get_color() 
        plt.plot(times, zSTDM, linewidth=1, alpha=0.4, color=color)
        if not (tm is None or np.isnan(tm)):
            plt.axvline(tm, linestyle='--', linewidth=2, color=color)
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
        z_Gmean = record['zGmean']
        smooth_zGmean = record['smooth_zGmean']
        tmpct = record['tmpct']
        times = record['time']
        [line] = plt.plot(times, smooth_zGmean, linewidth=1)
        color = line.get_color()
        plt.plot(times, z_Gmean, linewidth=1, alpha=0.4, color=color)
        if not (tmpct is None or np.isnan(tmpct)):
            plt.axvline(tmpct, linestyle='--', linewidth=2, color=color)
        tmpct_list.append(tmpct)
    plt.axhline(newthresh, linestyle='--', color='k', linewidth=2, label=f'threshold = {newthresh:.2f}')
    tmpct_mean, tmpct_std = np.nanmean(tmpct_list), np.nanstd(tmpct_list)
    plt.legend(loc='lower right', fontsize=14, title=f'Mean $t_m$ = {tmpct_mean:.1f}s,\n STD = {tmpct_std:.1f}s')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    result_file_pct = results_path / "all_videos_pct_analysis.png"
    plt.savefig(result_file_pct)
   # print(f"All PCT results saved to {result_file_pct}")

    plt.close('all')  # Close all figures to free memory

    return tmc_mean, tmc_std, tmpct_mean, tmpct_std


def main_processor(rawvideos_path, operating_system = 'rock', save_plots=False, 
                   Use_same_mask=True, totalframes2analyze=271, channel=1, span=150, spanderivative=150, 
                   threshstd=0.025, newthresh=0.95, injs=10, GG=30, FF=900, skip=100):
    """
    Main function to process videos and analyze mixing time.
    Parameters:
    - rawvideos_path: Path to the folder containing raw videos.
    - operating_system: 'rock' or 'compression' to determine the type of analysis.
    - save_plots: Boolean to save plots of the analysis.
    - Use_same_mask: Boolean to determine if the same mask should be used for all videos in a subfolder.
    - totalframes2analyze: Total frames to analyze per video.
    - channel: Channel to analyze (0 for red, 1 for green, 2 for blue).
    -  span: Smoothing span for Savitzky-Golay filter.
    - spanderivative: Smoothing span for derivative Savitzky-Golay filter.
    - threshstd: Threshold for standard deviation to detect flat regions.
    - newthresh: New threshold for PCT analysis.
    - injs: Seconds between injection and agitation.
    - GG: Number of frames to average at the beginning.
    - FF: Number of frames to average at the end.
    - skip: Number of frames to skip between each analyzed frame.

    """

    rawvideos_path = Path(rawvideos_path).resolve()
    results_path_global = Path(str(rawvideos_path).replace('RawVideos', 'Results').replace('\\', '/'))

    if not rawvideos_path.exists():
        raise FileNotFoundError(f"Raw videos path '{rawvideos_path}' does not exist.")

    # Get all "leaf" subfolders (no subdirectories)
    subfolders = [
        f for f in rawvideos_path.rglob('*') 
        if f.is_dir() and not any(child.is_dir() for child in f.iterdir())
    ]
    valid_extensions = ['.mp4', '.avi','.MP4', '.AVI']
    local_records, global_records = [], []


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
                if save_plots:
                    Figures_path = results_path / 'Figures' /f'{video_file.stem}'
                    Figures_path.mkdir(exist_ok=True, parents=True)
                else:
                    Figures_path = None
                results = analyze_mixing_time(str(video_file).replace('\\', '/'), 
                                                operating_system,
                                                save_plots,
                                                Figures_path,
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
                                                skip
                                         
                                                )
                zSTDM, zGmean,smoothSTD, smooth_zGmean, tm, tmpct, time = results
                local_records.append({
                    'video': video_file.name,
                    'tm': tm,
                    'zSTDM': zSTDM,
                    'zGmean': zGmean,
                    'tmpct': tmpct,
                    'smoothSTD': smoothSTD,
                    'smooth_zGmean': smooth_zGmean,
                    'time': time
                })
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                continue

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


#%%
def main():
    parser = argparse.ArgumentParser(description="Process videos and analyze mixing time.")
    parser.add_argument('--rawvideos_path', type=str, required=True, help='Path to  the folder containing raw videos.')
    parser.add_argument('--operating_system', type=str, default='rock', choices=['rock', 'compression'], help='Operating system type for analysis.')
    parser.add_argument('--save_plots', action='store_true', help='Save plots of the analysis.')
    parser.add_argument('--Use_same_mask', action='store_true', help='Use the same mask for all videos in a subfolder.')
    parser.add_argument('--totalframes2analyze', type=int, default=271, help='Total frames to analyze per video.')
    parser.add_argument('--channel', type=int, default=1, help='Channel to analyze (0 for red, 1 for green, 2 for blue).')
    parser.add_argument('--span', type=int, default=150, help='Smoothing span for Savitzky-Golay filter.')
    parser.add_argument('--spanderivative', type=int, default=150, help='Smoothing span for derivative Savitzky-Golay filter.')
    parser.add_argument('--threshstd', type=float, default=0.025, help='Threshold for standard deviation to detect flat regions.')
    parser.add_argument('--newthresh', type=float, default=0.95, help='New threshold for PCT analysis.')
    parser.add_argument('--injs', type=int, default=10, help='Seconds between injection and agitation.') 
    parser.add_argument('--GG', type=int, default=30, help='Number of frames to average at the beginning.')
    parser.add_argument('--FF', type=int, default=900, help='Number of frames to average at the end.')
    parser.add_argument('--skip', type=int, default=100, help='Number of frames to skip between each analyzed frame.')
       
    args = parser.parse_args()
    
    main_processor(args.rawvideos_path,args.operating_system, args.save_plots ,args.Use_same_mask, args.totalframes2analyze, args.channel,
                   args.span, args.spanderivative, args.threshstd, args.newthresh, args.injs, args.GG, args.FF, args.skip)
if __name__ == "__main__":
    main()

# %%
