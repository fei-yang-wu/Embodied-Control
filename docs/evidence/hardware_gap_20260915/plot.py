"""Render the analysis arrays with a Python environment containing matplotlib."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
arrays = np.load(OUT / "plot_data.npz")
names = arrays["joint_names"].tolist()
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
colors = {"f845_legkin_hw": "#b62732", "f845_legkin_sim": "#2363a5", "cal_baseline": "#20905d"}
for name, color in colors.items():
    fr = arrays[f"{name}/frames"]
    for ax, joint in zip(axes.flat[:3], ("left_knee_joint", "left_hip_pitch_joint", "left_ankle_pitch_joint")):
        j = names.index(joint)
        ax.plot(fr, arrays[f"{name}/q"][:, j], color=color, label=name)
        ax.plot(fr, arrays[f"{name}/cmd"][:, j], color=color, linestyle=":", alpha=.7)
        ax.set_title(joint.replace("_joint", "") + ": measured (solid), target (dotted)")
        ax.set_ylabel("rad")
    axes[1, 1].plot(fr, arrays[f"{name}/pitch_deg"], color=color)
    axes[2, 0].plot(fr, arrays[f"{name}/sole_height_difference_m"]*100, color=color)
    axes[2, 1].plot(fr, arrays[f"{name}/ankle_height_difference_m"]*100, color=color)
axes[1, 1].set_title("Pelvis pitch from recorded IMU"); axes[1, 1].set_ylabel("degrees")
axes[2, 0].set_title("Lowest left sphere surface minus right"); axes[2, 0].set_ylabel("cm; ground proxy only")
axes[2, 1].set_title("Left ankle center minus right"); axes[2, 1].set_ylabel("cm; not toe clearance")
for ax in axes.flat:
    ax.set_xlim(140, 215); ax.axvspan(176, 184, alpha=.10, color="black"); ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
for ax in axes[-1]: ax.set_xlabel("Reference frame (50 Hz)")
fig.suptitle("Hardware event vs old and new sim cadence; no physical contact sensor")
fig.tight_layout(); fig.savefig(OUT / "event.png", dpi=160); plt.close(fig)
