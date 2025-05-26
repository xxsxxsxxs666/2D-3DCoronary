import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from ..utils.vessel_tool import generate_2d_image_parreral, find_overlapping_area, vessel_bound
import seaborn as sns
import matplotlib.patheffects as path_effects
import matplotlib
from ..utils.wrapper_function import timer, select_visualize
from ..utils.vessel_projection import cta_2_dsa_2d_wbex, bbox_center_normalize

def cm_ranking(sc, ths=[512, 1024, 2048, 4096]):
    ls = sc.shape[0]
    colors = ["red", "yellow", "lime", "cyan", "blue"]
    out = ["gray"] * ls
    for i in range(ls):
        for c, th in zip(colors[: len(ths) + 1], ths + [ls]):
            if i < th:
                out[i] = c
                break
    sid = np.argsort(sc, axis=0).flip(0)
    out = np.array(out)[sid]
    return out


def cm_RdBl(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    # from IPython import embed; embed(colors="linux")
    return np.clip(c, 0, 1)


def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 1.0, 0, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)
    return out


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True, axis_off=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, axs = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        if axis_off:
            ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_image_grid(
    imgs,
    titles=None,
    cmaps="gray",
    dpi=100,
    pad=0.5,
    fig=None,
    adaptive=True,
    figs=2.0,
    return_fig=False,
    set_lim=False,
):
    """Plot a grid of images.
    Args:
        imgs: a list of lists of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    nr, n = len(imgs), len(imgs[0])
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs[0]]  # W / H
    else:
        ratios = [4 / 3] * n

    figsize = [sum(ratios) * figs, nr * figs]
    if fig is None:
        fig, axs = plt.subplots(
            nr, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
        )
    else:
        axs = fig.subplots(nr, n, gridspec_kw={"width_ratios": ratios})
        fig.figure.set_size_inches(figsize)
    if nr == 1:
        axs = [axs]

    for j in range(nr):
        for i in range(n):
            ax = axs[j][i]
            ax.imshow(imgs[j][i], cmap=plt.get_cmap(cmaps[i]))
            ax.set_axis_off()
            if set_lim:
                ax.set_xlim([0, imgs[j][i].shape[1]])
                ax.set_ylim([imgs[j][i].shape[0], 0])
            if titles:
                ax.set_title(titles[j][i])
    if isinstance(fig, plt.Figure):
        fig.tight_layout(pad=pad)
    if return_fig:
        return fig, axs
    else:
        return axs


def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0, show_legend=False, legend_name=None):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        if k is None:
            continue
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha, label=legend_name)
        if show_legend:
            ax.legend()

def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = sns.color_palette("husl", n_colors=len(kpts0))
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(
            kpts0[:, 0],
            kpts0[:, 1],
            c=color,
            s=ps,
            label=None if labels is None or len(labels) == 0 else labels[0],
        )
        ax1.scatter(
            kpts1[:, 0],
            kpts1[:, 1],
            c=color,
            s=ps,
            label=None if labels is None or len(labels) == 0 else labels[1],
        )


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
    axes=None,
    **kwargs,
):
    if axes is None:
        axes = plt.gcf().axes

    ax = axes[idx]
    t = ax.text(
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=ax.transAxes,
        **kwargs,
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )
    return t


def draw_epipolar_line(
    line, axis, imshape=None, color="b", label=None, alpha=1.0, visible=True
):
    if imshape is not None:
        h, w = imshape[:2]
    else:
        _, w = axis.get_xlim()
        h, _ = axis.get_ylim()
        imshape = (h + 0.5, w + 0.5)
    # Intersect line with lines representing image borders.
    X1 = np.cross(line, [1, 0, -1])
    X1 = X1[:2] / X1[2]
    X2 = np.cross(line, [1, 0, -w])
    X2 = X2[:2] / X2[2]
    X3 = np.cross(line, [0, 1, -1])
    X3 = X3[:2] / X3[2]
    X4 = np.cross(line, [0, 1, -h])
    X4 = X4[:2] / X4[2]

    # Find intersections which are not outside the image,
    # which will therefore be on the image border.
    Xs = [X1, X2, X3, X4]
    Ps = []
    for p in range(4):
        X = Xs[p]
        if (0 <= X[0] <= (w + 1e-6)) and (0 <= X[1] <= (h + 1e-6)):
            Ps.append(X)
            if len(Ps) == 2:
                break

    # Plot line, if it's visible in the image.
    if len(Ps) == 2:
        art = axis.plot(
            [Ps[0][0], Ps[1][0]],
            [Ps[0][1], Ps[1][1]],
            color,
            linestyle="dashed",
            label=label,
            alpha=alpha,
            visible=visible,
        )[0]
        return art
    else:
        return None


def get_line(F, kp):
    hom_kp = np.array([list(kp) + [1.0]]).transpose()
    return np.dot(F, hom_kp)


def plot_epipolar_lines(
    pts0, pts1, F, color="b", axes=None, labels=None, a=1.0, visible=True
):
    if axes is None:
        axes = plt.gcf().axes
    assert len(axes) == 2

    for ax, kps in zip(axes, [pts1, pts0]):
        _, w = ax.get_xlim()
        h, _ = ax.get_ylim()

        imshape = (h + 0.5, w + 0.5)
        for i in range(kps.shape[0]):
            if ax == axes[0]:
                line = get_line(F.transpose(0, 1), kps[i])[:, 0]
            else:
                line = get_line(F, kps[i])[:, 0]
            draw_epipolar_line(
                line,
                ax,
                imshape,
                color=color,
                label=None if labels is None else labels[i],
                alpha=a,
                visible=visible,
            )


def plot_heatmaps(heatmaps, vmin=0.0, vmax=None, cmap="Spectral", a=0.5, axes=None):
    if axes is None:
        axes = plt.gcf().axes
    artists = []
    for i in range(len(axes)):
        a_ = a if isinstance(a, float) else a[i]
        art = axes[i].imshow(
            heatmaps[i],
            alpha=(heatmaps[i] > vmin).float() * a_,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        artists.append(art)
    return artists


def plot_lines(
    lines,
    line_colors="orange",
    point_colors="cyan",
    ps=4,
    lw=2,
    alpha=1.0,
    indices=(0, 1),
):
    """Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        alpha: transparency of the points and lines.
        indices: indices of the images to draw the matches on.
    """
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 0, 0], l[i, 1, 0]),
                (l[i, 0, 1], l[i, 1, 1]),
                zorder=1,
                c=lc,
                linewidth=lw,
                alpha=alpha,
            )
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1], c=pc, s=ps, linewidths=0, zorder=2, alpha=alpha)


def plot_color_line_matches(lines, correct_matches=None, lw=2, indices=(0, 1)):
    """Plot line matches for existing images with multiple colors.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: bool array of size (N,) indicating correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    n_lines = len(lines[0])
    colors = sns.color_palette("husl", n_colors=n_lines)
    np.random.shuffle(colors)
    alphas = np.ones(n_lines)
    # If correct_matches is not None, display wrong matches with a low alpha
    if correct_matches is not None:
        alphas[~np.array(correct_matches)] = 0.2

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]

    # Plot the lines
    for a, img_lines in zip(axes, lines):
        for i, line in enumerate(img_lines):
            fig.add_artist(
                matplotlib.patches.ConnectionPatch(
                    xyA=tuple(line[0]),
                    coordsA=a.transData,
                    xyB=tuple(line[1]),
                    coordsB=a.transData,
                    zorder=1,
                    color=colors[i],
                    linewidth=lw,
                    alpha=alphas[i],
                )
            )


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)


def plot_cumulative(
    errors: dict,
    thresholds: list,
    colors=None,
    title="",
    unit="-",
    logx=False,
):
    thresholds = np.linspace(min(thresholds), max(thresholds), 100)

    plt.figure(figsize=[5, 8])
    for method in errors:
        recall = []
        errs = np.array(errors[method])
        for th in thresholds:
            recall.append(np.mean(errs <= th))
        plt.plot(
            thresholds,
            np.array(recall) * 100,
            label=method,
            c=colors[method] if colors else None,
            linewidth=3,
        )

    plt.grid()
    plt.xlabel(unit, fontsize=25)
    if logx:
        plt.semilogx()
    plt.ylim([0, 100])
    plt.yticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.ylabel(title + "Recall [%]", rotation=0, fontsize=25)
    plt.gca().yaxis.set_label_coords(x=0.45, y=1.02)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.yticks(rotation=0)

    plt.legend(
        bbox_to_anchor=(0.45, -0.12),
        ncol=2,
        loc="upper center",
        fontsize=20,
        handlelength=3,
    )
    plt.tight_layout()

    return plt.gcf()


def load_color_map(path: Path = None):
    """
    :param path: path to the color map
    :return: color map
    """
    if path is None:
        path = Path("gluefactory/utils/vessel_structure") / "color_map.json"
    with open(path, "r") as f:
        color_map = json.load(f)
    return color_map


def get_color_by_branch_id(branch_id, color_map=None):
    if color_map is None:
        color_map = load_color_map()
    color = color_map.get(str(int(branch_id)), None)
    if color is None:
        RuntimeWarning(f"Branch {branch_id} is not in the color map")
        # randomly assign a color not black or white
        color = np.random.rand(3)
        while np.all(color == 0) or np.all(color == 1):
            color = np.random.rand(3)
        color = color.tolist()
    return color

def get_color_by_branch_id_list(branch_id_list, color_map=None):
    if color_map is None:
        color_map = load_color_map()
    color_list = []
    for branch_id in branch_id_list:
        color = color_map.get(str(int(branch_id)), None)
        if color is None:
            RuntimeWarning(f"Branch {branch_id} is not in the color map")
            # randomly assign a color not black or white
            color = np.random.rand(3)
            while np.all(color == 0) or np.all(color == 1):
                color = np.random.rand(3)
            color = color.tolist()
        color_list.append(color)
    return color_list

def plot_vessel_tree(keypoints_list, label_list, axes=None, one_color=None,
                     show_legend=False, legend_name=None, linewidth=2, linestyle='-', alpha=0.5):
    """
    :param keypoints: m x 2
    :param label: label of the vessel tree
    :return: image with vessel tree
    """
    if one_color is None:
        color_map = load_color_map(Path("gluefactory/utils/vessel_structure") / "color_map.json")
    if axes is None:
        axes = plt.gcf().axes
    for ax, keypoints, label in zip(axes, keypoints_list, label_list):
        if keypoints is None:
            continue
        case_branch_id_list, branch_start_index = np.unique(label, return_index=True)
        branch_end_index = np.unique(label[::-1], return_index=True)[1]
        branch_end_index = len(label) - branch_end_index
        i_list = np.arange(len(branch_end_index))

        for i, branch_id, start_index, end_index in zip(i_list, case_branch_id_list, branch_start_index, branch_end_index):
            branch = keypoints[start_index:end_index]
            # plot the branch by line
            if one_color is not None:
                if i == len(branch_end_index) - 1:
                    ax.plot(branch[:, 0], branch[:, 1], color=one_color, linewidth=linewidth, linestyle=linestyle,
                            alpha=alpha, label=legend_name if show_legend else "Vessel Tree")
                else:
                    ax.plot(branch[:, 0], branch[:, 1], color=one_color, linewidth=linewidth, linestyle=linestyle,
                            alpha=alpha)
            else:
                color = color_map.get(str(int(branch_id)), None)
                if color is None:
                    RuntimeWarning(f"Branch {branch_id} is not in the color map")
                    # randomly assign a color not black or white
                    color = np.random.rand(3)
                    while np.all(color == 0) or np.all(color == 1):
                        color = np.random.rand(3)
                ax.plot(branch[:, 0], branch[:, 1], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                        label=branch_id)
            # show legend

            if show_legend:
                ax.legend()


def plot_vessel_percentage(keypoints_list, axes=None, c=None, cmap= None, s=2, alpha=0.5):
    """
    :param keypoints: m x 2
    :param label: label of the vessel tree
    :return: image with vessel tree
    """
    if cmap is None:
        cmap = plt.cm.get_cmap("viridis")
    if axes is None:
        axes = plt.gcf().axes
    fig = plt.gcf()
    for ax, keypoints in zip(axes, keypoints_list):
        # ax.plot(keypoints[:, 0], keypoints[:, 1], color=color, cmap=cmap, linewidth=linewidth, alpha=alpha)
        scatter = ax.scatter(keypoints[:, 0], keypoints[:, 1], c=c, cmap=cmap, alpha=alpha, s=s)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Value')


def wrap_index(data_list, index_list):
    """
    :param data_list: list of data
    :param index_list: list of index
    :return: list of data with index
    """
    return [data[index] for data, index in zip(data_list, index_list)]


def list_inverse(index_list):
    """
    :param index_list: list of index
    :return: list of index
    """
    return [~index for index in index_list]


def lax_color_generator(m0, gtm0, kp0, kp1):
    valid = (m0 > -1) & (gtm0 >= -1)
    kpm0, kpm1 = kp0[valid], kp1[m0[valid]]

    # todo: 这里会取到gt, -1的点，这个点是不应该被取到的
    correct = (gtm0[valid] == m0[valid])
    gt_kpm1 = kp1[gtm0[valid]]

    gt_unmatched_index = gtm0[valid] == -1
    gt_kpm1[gt_unmatched_index] = np.inf

    distance = np.linalg.norm(kpm1 - gt_kpm1, axis=-1)
    correct = correct

    correct_lax = correct.copy().astype(np.float32)
    correct_lax[(distance < 2) & (~correct)] = 0.5

    return cm_RdGn(correct_lax), correct_lax


@select_visualize()
def my_plot(x, y):
    plt.plot(x, y)

class ComplexVisualize:
    @staticmethod
    @select_visualize()
    def visualize_overlapping(imgs, keypoints_list, label_list, radius_list, size_list):
        plot_images(imgs)
        axes = plt.gcf().axes
        plot_vessel_tree(keypoints_list=keypoints_list, label_list=label_list, show_legend=False)
        for i, (kpt, label, radius, size) in enumerate(zip(keypoints_list, label_list, radius_list, size_list)):
            overlapping, connecting_area = find_overlapping_area(points_position=kpt, label=label,
                                                                 radius=radius,
                                                                 image_size=size, around_threshold=6)
            axes[i].imshow((overlapping - (overlapping > 1) * connecting_area) > 1, cmap="Reds", alpha=0.6)

    @staticmethod
    @select_visualize()
    def visualize_vessel_tree(imgs, keypoints_list, label_list):
        plot_images(imgs)
        plot_vessel_tree(keypoints_list=keypoints_list, label_list=label_list, show_legend=True)

    @staticmethod
    @select_visualize()
    def visualize_vessel_percentage(imgs, keypoints_list, cut_percentage):
        plot_images(imgs)
        plot_vessel_percentage(keypoints_list, c=cut_percentage, cmap="Reds")


    @staticmethod
    @select_visualize()
    def visualize_lm_position(imgs, lm_position):
        plot_images(imgs)
        ax = plt.gcf().axes
        for i, lm in enumerate(lm_position):
            ax[i].scatter([lm[0]], [lm[1]], c='r', s=10, alpha=0.5)

    @staticmethod
    @select_visualize()
    def visualize_cutting(imgs, keypoints_list, label_list, cut_index_list):
        plot_images(imgs)
        plot_vessel_tree(keypoints_list=wrap_index(keypoints_list, list_inverse(cut_index_list)),
                         label_list=wrap_index(label_list, list_inverse(cut_index_list)), show_legend=True)
        plot_vessel_tree(keypoints_list=wrap_index(keypoints_list, cut_index_list),
                         label_list=wrap_index(label_list, cut_index_list), one_color="red",
                         show_legend=True, legend_name="Cut",
                         linewidth=4, alpha=0.5)

    @staticmethod
    @select_visualize()
    def visualize_gt_matching(image0, image1, keypoints0, keypoints1, matches):
        plot_images([image0, image1])
        plot_keypoints([keypoints0, keypoints1], a=1.0, colors="k")
        valid_index = matches > -1
        kpt0 = keypoints0[valid_index]
        kpt1 = keypoints1[matches[valid_index]]
        plot_matches(kpt0, kpt1, a=0.2, color="lime")

    @staticmethod
    @select_visualize()
    def visualize_gt_matching_color(image0, image1, keypoints0, keypoints1, kpt1_label, matches):
        plot_images([image0, image1])
        plot_keypoints([keypoints0, keypoints1], a=1.0, colors="k")
        valid_index = matches > -1
        kpt0 = keypoints0[valid_index]
        kpt1 = keypoints1[matches[valid_index]]
        color = [get_color_by_branch_id(str(int(i))) for i in kpt1_label[matches[valid_index]]]
        plot_matches(kpt0, kpt1, a=0.2, color=color)

    @staticmethod
    @select_visualize()
    def visualize_pred_matching(image0, image1, keypoints0, keypoints1, matches, prune0_index=None, prune1_index=None):
        if prune0_index is None:
            prune0_index = np.zeros(len(keypoints0), dtype=bool)
        if prune1_index is None:
            prune1_index = np.zeros(len(keypoints1), dtype=bool)
        plot_images([image0, image1])
        plot_keypoints([keypoints0, keypoints1], a=1.0, colors="k")
        valid_index = matches > -1
        kpt1 = keypoints0[valid_index]
        kpt2 = keypoints1[matches[valid_index]]
        plot_matches(kpt1, kpt2, a=0.2, color="lime")
        if prune0_index.sum() > 0 or prune1_index.sum() > 0:
            plot_keypoints([keypoints0[prune0_index], keypoints1[prune1_index]], a=1.0, colors="r", ps=16)

    @staticmethod
    @select_visualize()
    def visualize_pred_matching_with_wrong(image0, image1, keypoints0, keypoints1, matches, gt_matches, prune0_index=None, prune1_index=None):
        if prune0_index is None:
            prune0_index = np.zeros(len(keypoints0), dtype=bool)
        if prune1_index is None:
            prune1_index = np.zeros(len(keypoints1), dtype=bool)
        plot_images([image0, image1])
        plot_keypoints([keypoints0, keypoints1], a=1.0, colors="k")
        valid_index = matches > -1
        kpt1 = keypoints0[valid_index]
        kpt2 = keypoints1[matches[valid_index]]
        color, correct_lax = lax_color_generator(matches, gt_matches, keypoints0, keypoints1)
        correct_lax = correct_lax > 0
        plot_matches(kpt1[correct_lax], kpt2[correct_lax], a=0.2, color=color[correct_lax].tolist())
        plot_matches(kpt1[~correct_lax], kpt2[~correct_lax], a=1, color=color[~correct_lax].tolist())
        if prune0_index.sum() > 0 or prune1_index.sum() > 0:
            plot_keypoints([keypoints0[prune0_index], keypoints1[prune1_index]], a=1.0, colors="r", ps=16)

    @staticmethod
    @select_visualize()
    def visualize_pred_matching_color(image0, image1, keypoints0, keypoints1, kpt1_label, matches, prune0_index=None, prune1_index=None):
        if prune0_index is None:
            prune0_index = np.zeros(len(keypoints0), dtype=bool)
        if prune1_index is None:
            prune1_index = np.zeros(len(keypoints1), dtype=bool)
        plot_images([image0, image1])
        plot_keypoints([keypoints0, keypoints1], a=1.0, colors="k")
        valid_index = matches > -1
        kpt1 = keypoints0[valid_index]
        kpt2 = keypoints1[matches[valid_index]]

        color = [get_color_by_branch_id(str(int(i))) for i in kpt1_label[matches[valid_index]]]
        plot_matches(kpt1, kpt2, a=0.2, color=color)
        if prune0_index.sum() > 0 or prune1_index.sum() > 0:
            plot_keypoints([keypoints0[prune0_index], keypoints1[prune1_index]], a=1.0, colors="r", ps=16)

    @staticmethod
    @select_visualize()
    def visualize_real_projection_process(dsa, grid_3d_points, grid_params, label):
        # if dsa is binary, using gray
        if dsa.max() == 1:
            plt.imshow(dsa, cmap="gray")
        else:
            plt.imshow(dsa)
        grid = cta_2_dsa_2d_wbex(grid_3d_points, grid_params)
        plot_vessel_tree([grid], [label], show_legend=True)
        ax = plt.gcf().axes[0]
        ax.patch.set_facecolor('black')
        ax.set_xlim([0, dsa.shape[1]])
        ax.set_ylim([dsa.shape[0], 0])

    @staticmethod
    @select_visualize()
    def visualize_real_projection_process_combine(dsa, grid_3d_points, grid_params, opt_params, gt_params, label, lm_delta=None):
        # if dsa is binary, using gray
        grid = cta_2_dsa_2d_wbex(grid_3d_points, grid_params)
        gt = cta_2_dsa_2d_wbex(grid_3d_points, gt_params)
        opt = cta_2_dsa_2d_wbex(grid_3d_points, opt_params)
        num = 3
        tree_list = [gt, opt, grid]
        if lm_delta is not None:
            num = 4
            tree_list.insert(-1, grid+lm_delta)
        plot_images([dsa] * num)
        plot_vessel_tree(tree_list, [label] * num, alpha=1.0, show_legend=False)
        axes = plt.gcf().axes
        for ax in axes:
            ax.patch.set_facecolor('black')
            ax.set_xlim([0, dsa.shape[1]])
            ax.set_ylim([dsa.shape[0], 0])

    @staticmethod
    @select_visualize()
    def visualize_simulated_projection_process(target_3d_points, grid_3d_points, target_params, grid_params,
                                               target_radius, grid_label):
        target = cta_2_dsa_2d_wbex(target_3d_points, target_params)
        grid = cta_2_dsa_2d_wbex(grid_3d_points, grid_params)
        target_centered, delta_coords = bbox_center_normalize(dsa_points=target, imager_size=(512, 512))
        grid_centered = grid + delta_coords
        plt.imshow(
            generate_2d_image_parreral(points_position=target_centered, radius=target_radius,
                                       image_size=(512, 512)),
            cmap="gray")
        in_index = (grid_centered[:, 0] > 0) & (grid_centered[:, 0] < 512) & (grid_centered[:, 1] > 0) & (
                grid_centered[:, 1] < 512)
        plot_vessel_tree([grid_centered[in_index]], [grid_label[in_index]], show_legend=True)

    @staticmethod
    @select_visualize()
    def visualize_simulated_projection_process_with_cutting(cta_3d_points, cut_index0, cut_index1, target_params, grid_params,
                                                            radius, label, show_full=True):
        target = cta_2_dsa_2d_wbex(cta_3d_points[~cut_index0], target_params)
        target_cut = cta_2_dsa_2d_wbex(cta_3d_points[cut_index0], target_params)
        target_full = cta_2_dsa_2d_wbex(cta_3d_points, target_params)
        grid = cta_2_dsa_2d_wbex(cta_3d_points[~cut_index1], grid_params)
        target_centered, delta_coords = bbox_center_normalize(dsa_points=target, imager_size=(512, 512))
        grid_centered = grid + delta_coords
        target_full_center = target_full + delta_coords
        target_cut_centered = target_cut + delta_coords
        plt.imshow(
            generate_2d_image_parreral(points_position=target_centered, radius=radius[~cut_index0],
                                       image_size=(512, 512)),
            cmap="gray")

        plot_vessel_tree([target_cut_centered], [label[cut_index0]],
                         one_color="whitesmoke",
                         show_legend=False, legend_name="Cut",
                         linewidth=4, linestyle="--", alpha=0.5)

        plot_vessel_tree([grid_centered], [label[~cut_index1]], show_legend=False,
                         alpha=1)
        h_min, h_max, w_min, w_max = vessel_bound(target_full_center, h=512, w=512, side=0)

        ax = plt.gcf().axes[0]
        ax.patch.set_facecolor('black')
        if show_full:
            ax.set_xlim([w_min, w_max])
            ax.set_ylim([h_max, h_min])
        else:
            ax.set_xlim([0, 512])
            ax.set_ylim([512, 0])

    @staticmethod
    @select_visualize()
    def visualize_simulated_projection_process_with_cutting_combine(cta_3d_points, cut_index0, cut_index1, target_params,
                                                                    grid_params, opt_params,
                                                                    target_radius, label, lm_delta=False, show_full=True):
        target = cta_2_dsa_2d_wbex(cta_3d_points[~cut_index0], target_params)
        target_cut = cta_2_dsa_2d_wbex(cta_3d_points[cut_index0], target_params)
        target_full = cta_2_dsa_2d_wbex(cta_3d_points, target_params)
        grid = cta_2_dsa_2d_wbex(cta_3d_points[~cut_index1], grid_params)

        opt = cta_2_dsa_2d_wbex(cta_3d_points[~cut_index1], opt_params)
        target_centered, delta_coords = bbox_center_normalize(dsa_points=target, imager_size=(512, 512))

        grid_full = cta_2_dsa_2d_wbex(cta_3d_points, grid_params)
        lm_index = int((np.where(label == 5)[0][0] + np.where(label == 5)[0][-1]) / 2)
        delta = (target + delta_coords)[lm_index] - grid_full[lm_index]

        grid_centered = grid + delta_coords
        opt_centered = opt + delta_coords
        target_full_center = target_full + delta_coords
        target_cut_centered = target_cut + delta_coords
        target_image = generate_2d_image_parreral(points_position=target_centered, radius=target_radius[~cut_index0],
                                                  image_size=(512, 512))
        num = 2
        tree_list = [grid_centered, opt]
        if lm_delta:
            num = 3
            tree_list = [grid_centered, grid+delta, opt_centered]
        plot_images([target_image] * num, axis_off=False)
        plot_vessel_tree([target_cut_centered] * num, [label[cut_index0]] * num,
                         one_color="whitesmoke",
                         show_legend=False, legend_name="Cut",
                         linewidth=4, linestyle="--", alpha=0.5)

        plot_vessel_tree(tree_list, [label[~cut_index1]]*num, show_legend=False,
                         alpha=1)
        h_min, h_max, w_min, w_max = vessel_bound(target_full_center, h=512, w=512, side=0)

        for ax in plt.gcf().axes:
            ax.patch.set_facecolor('black')
            if show_full:
                ax.set_xlim([w_min, w_max])
                ax.set_ylim([h_max, h_min])
            else:
                ax.set_xlim([0, 512])
                ax.set_ylim([512, 0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ax = plt.figure().add_subplot(projection='3d')

    # Plot a sin curve using the x and y axes.
    x = np.linspace(0, 1, 100)
    y = np.sin(x * 2 * np.pi) / 2 + 0.5
    ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

    # Plot scatterplot data (20 2D points per colour) on the x and z axes.
    colors = ('r', 'g', 'b', 'k')

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    x = np.random.sample(20 * len(colors))
    y = np.random.sample(20 * len(colors))
    c_list = []
    for c in colors:
        c_list.extend([c] * 20)
    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x, y) points are plotted on the x and z axes.
    ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35, roll=0)

    plt.show()



















