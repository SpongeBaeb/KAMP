"""공통 그래프 스타일."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_L, C_U, C_FAIL, C_PASS = "#2a78d6", "#eb6834", "#e34948", "#9a9892"
ALGO_COLOR = {"LogReg": "#2a78d6", "RF": "#eb6834", "HGB": "#1baf7a"}
IMB_MARKER = {"class_weight": "o", "ROS": "s", "SMOTE": "^"}
INK2 = "#52514e"

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
