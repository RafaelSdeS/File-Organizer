"""PNG charts of one analyze_directory run (the --report flag).

matplotlib is an optional extra (`pip install -e .[viz]`), imported lazily so
the core CLI never needs it.
"""
import os

import numpy as np

from .analyzer import MISC_FOLDER

# Light-mode reference palette. One highlight hue only: past 3 categorical
# colors a scatter can't be told apart reliably (validated), so folders are
# identified by position, a shaded outline and a direct label - not by color.
HIGHLIGHT = "#2a78d6"
GROUP_FILL = "#e3ecf8"
BAND = "#ecebe7"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Sized for a README: ~9 in wide at 200 dpi is shown at ~880 px, so 11 pt
# text stays readable after the downscale.
DPI = 200
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
}


def save_report(last_run, folder_structure, out_dir):
    """Write map.png, k_selection.png (when k was searched) and similarity.png
    to out_dir; returns the paths written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("--report needs matplotlib: pip install -e .[viz]") from None
    if not last_run:
        raise ValueError("Nothing to report: analyze_directory hasn't run")

    os.makedirs(out_dir, exist_ok=True)
    paths = last_run["paths"]
    embeddings = np.asarray(last_run["embeddings"], dtype=float)
    folder_of = {path: folder for folder, files in folder_structure.items() for path in files}

    # Largest folders first, Misc last.
    folders = sorted(folder_structure, key=lambda f: (f == MISC_FOLDER, -len(folder_structure[f])))

    saved = []
    with plt.rc_context(STYLE):
        if len(paths) >= 3:
            saved.append(_document_map(plt, paths, embeddings, folder_of, folders, out_dir))
        if last_run.get("k_scores"):
            saved.append(_k_selection(plt, last_run["k_scores"], out_dir))
        saved.append(_similarity(plt, paths, embeddings, folder_of, folders, out_dir))
    return saved


def _headline(fig, title, subtitle):
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.94, subtitle, ha="left", va="top", fontsize=10.5, color=INK_2)


def _group_outline(ax, points):
    """Soft shaded blob around a group of 2-D points: the convex hull drawn
    with a thick round-joined edge in the fill color, which pads it evenly.
    1-2 points (or collinear ones, no hull) become a round-capped stroke."""
    from scipy.spatial import ConvexHull, QhullError

    style = {"color": GROUP_FILL, "linewidth": 26, "zorder": 1}
    try:
        outline = points[ConvexHull(points).vertices] if len(points) >= 3 else points
    except QhullError:
        outline = points
    if len(outline) >= 3:
        ax.fill(outline[:, 0], outline[:, 1], joinstyle="round", **style)
    else:
        # The marker covers a lone point, which has no line to cap.
        ax.plot(outline[:, 0], outline[:, 1], solid_capstyle="round",
                marker="o", markersize=26, **style)


def _declutter(fig, labels, step=5, tries=15):
    """Nudge each (annotation, direction) by `step` points along `direction`
    (+1 up, -1 down) until it overlaps no label placed before it."""
    renderer = fig.canvas.get_renderer()
    placed = []
    for label, direction in labels:
        for _ in range(tries):
            box = label.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed):
                break
            x, y = label.xyann
            label.xyann = (x, y + direction * step)
        placed.append(label.get_window_extent(renderer))


def _document_map(plt, paths, embeddings, folder_of, folders, out_dir):
    from sklearn.manifold import TSNE

    # t-SNE keeps each file's nearest neighbours near it, which is the
    # question here ("did similar files land together?"). Fixed seed, so the
    # same run always draws the same map.
    n = len(paths)
    xy = TSNE(n_components=2, metric="cosine", init="pca", random_state=0,
              perplexity=max(2, min(30, (n - 1) // 3))).fit_transform(embeddings)
    xy = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0).clip(1e-9)

    fig, ax = plt.subplots(figsize=(9, 7.2))
    folder_labels, misc_labels = [], []
    for folder in folders:
        pts = xy[[i for i, p in enumerate(paths) if folder_of.get(p) == folder]]
        if folder == MISC_FOLDER:
            ax.scatter(pts[:, 0], pts[:, 1], s=46, facecolors=SURFACE, edgecolors=MUTED,
                       linewidths=1.5, zorder=3)
            misc_names = [os.path.splitext(p)[0] for p in paths if folder_of.get(p) == folder]
            for (x, y), name in zip(pts, misc_names):
                misc_labels.append(ax.annotate(name, (x, y), xytext=(6, -3), textcoords="offset points",
                            fontsize=8.5, color=INK_2, style="italic", va="top", zorder=6,
                            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1}))
            continue
        _group_outline(ax, pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=46, c=HIGHLIGHT, edgecolors=SURFACE,
                   linewidths=1.2, zorder=3)
        folder_labels.append(ax.annotate(f"{folder} ({len(pts)})", (pts[:, 0].mean(), pts[:, 1].max()),
                    xytext=(0, 13), textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5}))

    ax.margins(0.09)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _headline(fig, "Files that read alike end up close together",
              "Each dot is a file, placed by content similarity (t-SNE of the embeddings).\n"
              "Shaded groups are the folders the tool created; hollow dots went to Misc.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    # Clashing labels move up: a Misc file name starts just below-right of
    # its dot, so moving up keeps it beside that dot.
    _declutter(fig, [(label, 1) for label in folder_labels + misc_labels])
    return _save(plt, fig, out_dir, "map.png")


def _k_selection(plt, scores, out_dir):
    ks, chosen, votes = scores["k"], scores["chosen"], scores["votes"]
    # The elbow vote is the k with the sharpest bend in the inertia curve, so
    # plot the bend itself (2nd difference): then every panel votes at its peak.
    panels = [
        ("Elbow", "how sharply inertia bends", ks[1:-1], list(np.diff(scores["inertia"], 2)), votes.get("elbow")),
        ("Silhouette", "how well separated", ks, scores["silhouette"], votes["silhouette"]),
        # Davies-Bouldin is lower-is-better; its axis is flipped below so the
        # vote is the peak here too.
        ("Davies-Bouldin", "tighter, better-separated (axis flipped)", ks, scores["davies_bouldin"], votes["davies_bouldin"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharex=True)
    for ax, (name, meaning, x, y, vote) in zip(axes, panels):
        ax.axvspan(chosen - 0.45, chosen + 0.45, color=BAND, linewidth=0, zorder=0)
        if len(x):
            ax.plot(x, y, color=HIGHLIGHT, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=2)
        if vote is None:
            ax.text(0.5, 0.5, "no vote\n(needs ≥ 5 files)", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
        else:
            vy = y[list(x).index(vote)]
            ax.scatter([vote], [vy], s=90, c=INK, edgecolors=SURFACE, linewidths=1.5, zorder=4)
            ax.annotate(f"votes {vote}", (vote, vy), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.set_title(name, pad=20)
        ax.text(0, 1.03, f"higher = {meaning}", transform=ax.transAxes, fontsize=9.5, color=MUTED)
        ax.set_xticks(ks)
        ax.set_xlabel("number of folders (k)", fontsize=10)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.margins(y=0.25)
        if name == "Davies-Bouldin":
            ax.invert_yaxis()

    vote_line = " · ".join(f"{name} {vote if vote is not None else '–'}"
                           for name, _, _, _, vote in panels)
    _headline(fig, f"Picking the number of folders: k = {chosen}",
              f"Each metric votes for its peak and the median wins (shaded column).  Votes: {vote_line}")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(plt, fig, out_dir, "k_selection.png")


def _similarity(plt, paths, embeddings, folder_of, folders, out_dir):
    from matplotlib.colors import LinearSegmentedColormap

    order = [i for folder in folders for i, p in enumerate(paths) if folder_of.get(p) == folder]
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = (unit @ unit.T)[np.ix_(order, order)]
    # A file's similarity to itself says nothing, and at 1.0 it would
    # dominate the scale: blank the diagonal.
    np.fill_diagonal(sim, np.nan)

    # Stretch the color scale over the pairs that exist instead of 0..1.
    pairs = sim[~np.isnan(sim)]
    vmin, vmax = (np.percentile(pairs, 2), pairs.max()) if pairs.size else (0, 1)

    cmap = LinearSegmentedColormap.from_list("blues", BLUES).with_extremes(bad=SURFACE)
    fig, ax = plt.subplots(figsize=(10, 7.4))
    image = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_ticks([vmin, vmax], labels=["less alike", "more alike"])
    colorbar.ax.tick_params(length=0, labelsize=10)
    colorbar.outline.set_visible(False)

    # Surface-colored gaps between folder blocks, folder names at block centres.
    sizes = [sum(1 for p in paths if folder_of.get(p) == f) for f in folders]
    edges = np.cumsum(sizes)
    for edge in edges[:-1]:
        ax.axhline(edge - 0.5, color=SURFACE, linewidth=2.5)
        ax.axvline(edge - 0.5, color=SURFACE, linewidth=2.5)
    centres = edges - np.array(sizes) / 2 - 0.5
    ax.set_yticks(centres, [f"{f} ({n})" for f, n in zip(folders, sizes)], fontsize=10.5)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _headline(fig, "Files in the same folder resemble each other",
              "Similarity of every pair of files, sorted by folder: dark blocks on the diagonal\n"
              "mean a folder's files are more alike than files across folders.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return _save(plt, fig, out_dir, "similarity.png")


def _save(plt, fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
