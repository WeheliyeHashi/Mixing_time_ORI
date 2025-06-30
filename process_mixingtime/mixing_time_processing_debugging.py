# %%
%matplotlib qt 
import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from Reader.readVideomp4 import readVideomp4
# %%

# ------- Define functions -------

def fit_circle_least_squares(pts):
    # pts: list of (x, y)
    pts = np.array(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
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
        vid = readVideomp4(video_file)
    else:
        raise Exception("Only *.mp4 and *.avi videos are supported so far")

    if vid.width == 0 or vid.height == 0:
        raise RuntimeError

    return vid
#%%
# ------- Input Parameters -------
R = 1
video = 1
channel = 1  # 0: R, 1: G, 2: B (OpenCV uses BGR)
Filter = 1  # Placeholder; define actual filter later (e.g., gaussian_filter)
span = 150
spanderivative = 150
consecutivePoints = 5
threshstd = 0.025 # threshold for mixing time calculation using standard deviation
newthresh = 0.95    # threshold for mixing time calculation using PCT
fromend = 1 if R in [1, 3] else 1  # Set manually as in original script
frombeg = 1
totalframes2analyze = 271 # total frames to analyze
FF = 30 * 30  # frames to average at the end
injs = 10  # seconds between injection and agitation

thresh = 0.95   
#%%
ID = "experiment1"
video_path = "Data/RawVideos/26062025/exp_id1/run1_250ml_rock.avi"  # Replace with actual video path
pathOG = os.getcwd()
# ------- Mask Definition -------
# v = cv2.VideoCapture(video_path)
# _, frame = v.read()
store = selectVideoReader(video_path)
frame = store.read()[-1]  # Read the first frame
fig, ax = plt.subplots()
ax.imshow(frame, cmap='gray')
ax.set_title("Click points to define mask area (Press Enter when done)")
plt.axis("off")

# Allow user to click on the image
pts = plt.ginput(n=-1, timeout=0)  # n=-1 means unlimited points
plt.close()

mask = _return_masked_image(frame, pts)
#%%
# ------- fromend and frombeg -------


# # ------- Open Video -------
# v = cv2.VideoCapture(video_path)
# fps = v.get(cv2.CAP_PROP_FPS)
# totalframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
# height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = np.ceil(store.fps)
totalframes = int(store.tot_frames)
height = int(store.height)
width = int(store.width)

if video == 1:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writerObj = cv2.VideoWriter(f"{ID}_vid.mp4", fourcc, 15, (width, height))

# ------- Average of Initial Frames -------
GG = 30
indicesi = [(frombeg + i) / fps for i in range(GG)]
initial = []

for i, t in enumerate(indicesi):
    print(f"Processing initial frame {i+1}")
    # v.set(cv2.CAP_PROP_POS_MSEC, t )
    # ret, frame = v.read()
    frame = store.read_frame(int(t * fps))[1]
    if frame is None:
        print(f"Frame {i+1} could not be read.")
        continue
    frameg = frame[:, :, channel]
    doub = np.flipud(frameg.astype(float))
    doub[mask == 0] = np.nan
    initial.append(doub)

M0 = np.nanmean(initial, axis=0)
M0 = gaussian_filter(M0, sigma=Filter)
plt.figure()
plt.title("Initial Frame Average: M0")
plt.imshow(M0, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.axis("off")
plt.tight_layout()
#%%
# ------- Average of Final Frames -------
# v = cv2.VideoCapture(video_path)
indicesfx = [((totalframes - FF - fromend + i) / fps) for i in range(FF)]
indicesf = indicesfx[::10]  # every 10th frame
final = []

for i, t in enumerate(indicesf):
    print(f"Processing final frame {i+1}")
    # v.set(cv2.CAP_PROP_POS_MSEC, t )
    # ret, frame = v.read()
    frame = store.read_frame(int(t * fps))[1]
    if frame is None:
        print(f"Frame {i+1} could not be read.")
        continue
    frameg = frame[:, :, channel]
    doub = np.flipud(frameg.astype(float))
    doub[mask == 0] = np.nan
    final.append(doub)

M1 = np.nanmean(final, axis=0)
M1 = gaussian_filter(M1, sigma=Filter)
plt.figure()
plt.title("Final Frame Average: M1")        
plt.imshow(M1, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.axis("off")
plt.tight_layout()
#%%
# ------- Mixing Calc All Frames -------
totalpx = np.sum(~np.isnan(M0))

AA = round(totalframes / totalframes2analyze)  # arbitrary choice: sample every AA-th frame
times = [(i * AA) / fps for i in range(int(totalframes / AA) - 1)]
store = selectVideoReader(video_path)
PCT, zGmean, zGmeanabs, zSTD, zSTDM = [], [], [], [], []
zred, zgreen, zblue, zGmax, zGmaxabs = [], [], [], [], []
# v = cv2.VideoCapture(video_path)
for j, t in enumerate(times):
    print(f"Processing frame at {t:.2f} s")
    # v.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    # ret, frame = v.read()
    frame = store.read_frame(int(t * fps))[1]
    if frame is None:
        break

    frame = frame.astype(float)
    M = np.flipud(frame[:, :, channel])
    N = np.flipud(frame[:, :, 2])  # red
    O = np.flipud(frame[:, :, 0])  # blue

    for arr in [M, N, O]:
        arr[mask == 0] = np.nan

    M = gaussian_filter(M, sigma=Filter)

    zred.append(np.nanmean(N))
    zgreen.append(np.nanmean(M))
    zblue.append(np.nanmean(O))

    G = 1 - (M - M1) / (M0 - M1)
    G[np.isinf(G)] = np.nan
    G[np.isneginf(G)] = np.nan

    g = np.where(np.abs(G) > thresh)
    PCT.append(len(g[0]) / totalpx)

    zGmax.append(np.nanmax(G))
    zGmaxabs.append(np.abs(np.nanmin(G)))
    zGmean.append(np.nanmean(G))
    zGmeanabs.append(np.nanmean(np.abs(G)))
    zSTD.append(np.nanstd(G))
    zSTDM.append(np.nanstd(M))

    if video == 1:
        plt.figure()
        plt.pcolormesh(G * 100, shading="auto", cmap="bone_r", vmin=0, vmax=100)
        plt.colorbar(label="PCT [%]")
        plt.title(f"G, {t:.2f}s")
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("temp_frame.png", bbox_inches="tight")
        plt.close()
        temp_frame = cv2.imread("temp_frame.png")
        temp_frame = cv2.resize(temp_frame, (width, height))
        writerObj.write(temp_frame)
#%%
if video == 1:
    writerObj.release()
    os.chdir(pathOG)
    print(f"Video saved as {ID}_vid.mp4")
# %%

# Smooth zSTDM and compute derivative
smoothSTD = savgol_filter(zSTDM, span, 3, mode='interp')
derivative = savgol_filter(np.gradient(smoothSTD), spanderivative, 3, mode='interp')

# Find flat indices in the second half of the array
flatIndices = np.where(np.abs(derivative) <= threshstd)[0]
zallindices = len(derivative)
flatIndices = flatIndices[flatIndices >= int(0.4 * zallindices)]

# Find consecutive points
flatIndex = None
for i in range(len(flatIndices) - consecutivePoints + 1):
    if np.all(np.diff(flatIndices[i:i + consecutivePoints]) == 1):
        flatIndex = flatIndices[i]
        break

empty = int(flatIndex is None)

# Mixing time estimation based on PCT threshold
smooth_zGmean = savgol_filter(zGmean, span, 2, mode='interp')
fmlpct = np.argmax(smooth_zGmean > newthresh) if np.any(smooth_zGmean > newthresh) else None
empty1 = int(fmlpct is None)

fml = flatIndex
tm = times[fml] - injs if not empty else np.nan
tmpct = times[fmlpct] - injs if not empty1 else np.nan
# %%
# Plot results for the mixing time STD analysis
times = np.array(times)
# Plot derivative
plt.figure()
plt.grid(True)
plt.box(True)
plt.title('STD Analysis')
plt.xlabel('time [s]')

plt.plot(times - injs,  zSTDM, label='std(G)')
plt.plot(times - injs, smoothSTD, label='smooth-std(G)', linewidth=1.5)

plt.ylabel('std(G)')


if not empty:
    plt.axvline(tm, linestyle='--', color='k', linewidth =3, label=f'$t_m = {tm:.1f}s$')
# plt.axhline(0, color='k', linewidth=1)
plt.legend(loc='upper right', fontsize=14)
plt.savefig(f"{ID}_derivative.png")
#plt.close()

# %%s
# Plot PCT
plt.figure()
plt.grid(True)
plt.box(True)
plt.title('PCT Analysis')
plt.xlabel('time [s]')

plt.plot(times - injs,  zGmean, label='mean(G)')
plt.plot(times - injs, smooth_zGmean, label='smooth-mean(G)', linewidth=1.5)

plt.ylabel('PCT [-]')


if not empty:
    plt.axvline(tmpct, linestyle='--', color='k', linewidth =3, label=f'$t_m = {tmpct:.1f}s$')
plt.axhline(newthresh, linestyle='--',color='k', linewidth=3, label=f'threshold = {newthresh:.2f}')
plt.legend(loc='lower right', fontsize=14)
plt.savefig(f"{ID}_derivative.png")
# %%
