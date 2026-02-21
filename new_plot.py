import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# ==============================
# CONFIG
# ==============================
OBJ = {
    "makespan": 0,
    "energy": 1,
    "cost": 2,
    "migration": 3
}

# ==============================
# LOAD DATA
# ==============================
def load_objectives(path):
    data = np.load(path)
    assert data.shape[1] == 4, "Expected 4 objectives"
    return data

# ==============================
# DOMINANCE
# ==============================
def is_dominated(a, b):
    return np.all(b <= a) and np.any(b < a)

def pareto_front_nd(points):
    mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not mask[i]:
            continue
        for j in range(len(points)):
            if i != j and is_dominated(points[i], points[j]):
                mask[i] = False
                break
    return mask

# ==============================
# 2D PARETO
# ==============================
def plot_2d_pareto(data, x, y, xlab, ylab, title):
    pts = data[:, [x, y]]
    pf_mask = pareto_front_nd(pts)

    plt.figure(figsize=(7, 5))
    plt.scatter(pts[~pf_mask, 0], pts[~pf_mask, 1],
                alpha=0.3, label="Dominated")
    plt.scatter(pts[pf_mask, 0], pts[pf_mask, 1],
                s=60, label="Pareto Front")

    # connect Pareto points
    pf = pts[pf_mask]
    pf = pf[np.argsort(pf[:, 0])]
    plt.plot(pf[:, 0], pf[:, 1], linewidth=2)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==============================
# 3D PARETO
# ==============================
def plot_3d_pareto(data, idxs, labels, title):
    pts = data[:, idxs]
    pf_mask = pareto_front_nd(pts)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pts[~pf_mask, 0],
        pts[~pf_mask, 1],
        pts[~pf_mask, 2],
        alpha=0.25,
        label="Dominated"
    )

    ax.scatter(
        pts[pf_mask, 0],
        pts[pf_mask, 1],
        pts[pf_mask, 2],
        s=80,
        label="3D Pareto Front"
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    raw = load_objectives("results/mo_nso_archive_raw_objectives.npy")

    # ---- 2D PLOTS ----
    plot_2d_pareto(raw, OBJ["makespan"], OBJ["cost"],
                   "Makespan (s)", "Cost ($)",
                   "Pareto: Makespan vs Cost")

    plot_2d_pareto(raw, OBJ["cost"], OBJ["migration"],
                   "Cost ($)", "Migration (s)",
                   "Pareto: Cost vs Migration")

    plot_2d_pareto(raw, OBJ["makespan"], OBJ["energy"],
                   "Makespan (s)", "Energy (J)",
                   "Pareto: Makespan vs Energy")

    plot_2d_pareto(raw, OBJ["makespan"], OBJ["migration"],
                   "Makespan (s)", "Migration (s)",
                   "Pareto: Makespan vs Migration")

    # ---- 3D PLOTS (ALL COMBINATIONS) ----
    names = list(OBJ.keys())
    for combo in combinations(names, 3):
        idxs = [OBJ[c] for c in combo]
        labels = [c.capitalize() for c in combo]

        plot_3d_pareto(
            raw,
            idxs,
            labels,
            f"3D Pareto: {labels[0]} vs {labels[1]} vs {labels[2]}"
        )










# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from itertools import combinations

# # ==============================
# # CONFIG
# # ==============================
# OBJ = {
#     "makespan": 0,
#     "energy": 1,
#     "cost": 2,
#     "migration": 3
# }

# # ==============================
# # LOAD DATA
# # ==============================
# def load_objectives(path):
#     data = np.load(path)
#     assert data.shape[1] == 4, "Expected 4 objectives"
#     return data

# # ==============================
# # NORMALIZATION (FOR PLOTTING)
# # ==============================
# def minmax_normalize(data):
#     """
#     Min–max normalize each objective column independently.
#     Used ONLY for visualization.
#     """
#     min_vals = data.min(axis=0, keepdims=True)
#     max_vals = data.max(axis=0, keepdims=True)
#     ranges = max_vals - min_vals
#     ranges[ranges < 1e-12] = 1.0
#     return (data - min_vals) / ranges

# # ==============================
# # DOMINANCE
# # ==============================
# def is_dominated(a, b):
#     return np.all(b <= a) and np.any(b < a)

# def pareto_front_nd(points):
#     mask = np.ones(len(points), dtype=bool)
#     for i in range(len(points)):
#         if not mask[i]:
#             continue
#         for j in range(len(points)):
#             if i != j and is_dominated(points[i], points[j]):
#                 mask[i] = False
#                 break
#     return mask

# # ==============================
# # 2D PARETO
# # ==============================
# def plot_2d_pareto(data, x, y, xlab, ylab, title):
#     pts = data[:, [x, y]]
#     pf_mask = pareto_front_nd(pts)

#     plt.figure(figsize=(7, 5))
#     plt.scatter(pts[~pf_mask, 0], pts[~pf_mask, 1],
#                 alpha=0.3, label="Dominated")
#     plt.scatter(pts[pf_mask, 0], pts[pf_mask, 1],
#                 s=60, label="Pareto Front")

#     pf = pts[pf_mask]
#     pf = pf[np.argsort(pf[:, 0])]
#     plt.plot(pf[:, 0], pf[:, 1], linewidth=2)

#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
#     plt.title(title)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # ==============================
# # 3D PARETO
# # ==============================
# def plot_3d_pareto(data, idxs, labels, title):
#     pts = data[:, idxs]
#     pf_mask = pareto_front_nd(pts)

#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")

#     ax.scatter(
#         pts[~pf_mask, 0],
#         pts[~pf_mask, 1],
#         pts[~pf_mask, 2],
#         alpha=0.25,
#         label="Dominated"
#     )

#     ax.scatter(
#         pts[pf_mask, 0],
#         pts[pf_mask, 1],
#         pts[pf_mask, 2],
#         s=80,
#         label="Pareto Front"
#     )

#     ax.set_xlabel(labels[0])
#     ax.set_ylabel(labels[1])
#     ax.set_zlabel(labels[2])
#     ax.set_title(title)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

# # ==============================
# # MAIN
# # ==============================
# if __name__ == "__main__":

#     raw = load_objectives("results/mo_nso_archive_raw_objectives.npy")

#     # ==========================
#     # RAW PLOTS (PHYSICAL)
#     # ==========================
#     plot_2d_pareto(raw, OBJ["makespan"], OBJ["cost"],
#                    "Makespan (s)", "Cost ($)",
#                    "RAW Pareto: Makespan vs Cost")

#     plot_2d_pareto(raw, OBJ["cost"], OBJ["migration"],
#                    "Cost ($)", "Migration (s)",
#                    "RAW Pareto: Cost vs Migration")

#     # ==========================
#     # NORMALIZED PLOTS
#     # ==========================
#     norm = minmax_normalize(raw)

#     plot_2d_pareto(norm, OBJ["makespan"], OBJ["cost"],
#                    "Normalized Makespan", "Normalized Cost",
#                    "NORMALIZED Pareto: Makespan vs Cost")

#     plot_2d_pareto(norm, OBJ["cost"], OBJ["migration"],
#                    "Normalized Cost", "Normalized Migration",
#                    "NORMALIZED Pareto: Cost vs Migration")

#     plot_2d_pareto(norm, OBJ["makespan"], OBJ["energy"],
#                    "Normalized Makespan", "Normalized Energy",
#                    "NORMALIZED Pareto: Makespan vs Energy")

#     plot_2d_pareto(norm, OBJ["makespan"], OBJ["migration"],
#                    "Normalized Makespan", "Normalized Migration",
#                    "NORMALIZED Pareto: Makespan vs Migration")

#     # ==========================
#     # 3D NORMALIZED PARETO
#     # ==========================
#     names = list(OBJ.keys())
#     for combo in combinations(names, 3):
#         idxs = [OBJ[c] for c in combo]
#         labels = [f"Normalized {c.capitalize()}" for c in combo]

#         plot_3d_pareto(
#             norm,
#             idxs,
#             labels,
#             f"NORMALIZED 3D Pareto: {combo[0]} vs {combo[1]} vs {combo[2]}"
#         )
