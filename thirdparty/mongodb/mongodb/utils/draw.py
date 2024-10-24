from typing import Tuple, Optional, List, Sequence, Union
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import io
from PIL import Image
import random

from .bbox import BBox
from .tr_image import get_dsize

BB_COLOR = 0, 255, 255
BB_THICKNESS = 8

CT_COLOR = 0, 0, 255
CT_THICKNESS = 8

Contours = List[List[int]]


def draw_bbox(
    image: np.ndarray,
    bbox: BBox,
    dsize: Optional[Union[float, Tuple[int]]] = None,
    color: Tuple[int] = BB_COLOR,
    text: str = None,
    thickness: int = BB_THICKNESS,
    *args,
    **kwargs,
):
    drawn_img = image.copy()
    if drawn_img.ndim == 3 and drawn_img.shape[-1] == 1:
        drawn_img = drawn_img[..., 0]
    if drawn_img.ndim == 2:
        drawn_img = np.stack([drawn_img] * 3, axis=-1)
    if text is None and bbox.score is not None:
        text = str(bbox.score)
    cv2.rectangle(
        drawn_img,
        (int(bbox.coords[0]), int(bbox.coords[1])),
        (int(bbox.coords[2]), int(bbox.coords[3])),
        color=color,
        thickness=thickness,
        *args,
        **kwargs,
    )
    if text is not None:
        cv2.putText(
            drawn_img,
            text,
            (int(bbox.coords[0]), int(bbox.coords[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (color[0], color[1], color[2]),
            2,
            cv2.LINE_AA,
        )
    if dsize is not None:
        dsize = get_dsize(drawn_img, dsize)
        drawn_img = cv2.resize(drawn_img, dsize=dsize)
    return drawn_img


def draw_bboxes(
    image: np.array,
    bboxes: Optional[List[BBox]],
    dsize: Optional[Union[float, Tuple[int]]] = None,
    color: list = BB_COLOR,
    labels: Optional[Sequence[str]] = None,
    thickness: int = BB_THICKNESS,
    *args,
    **kwargs,
):
    if bboxes is None or len(bboxes) == 0:
        return image
    if labels is None:
        labels = [None] * len(bboxes)
    for bbox, label in zip(bboxes, labels):
        image = draw_bbox(
            image,
            bbox,
            dsize=dsize,
            color=color,
            text=label,
            thickness=thickness,
            *args,
            **kwargs,
        )
    return image


def draw_contours(
    image: np.ndarray,
    contours: Contours,
    dsize: Optional[Union[float, Tuple[int]]] = None,
    color: list = CT_COLOR,
    thickness: int = CT_THICKNESS,
) -> np.ndarray:
    drawn_img = image.copy()
    for contour in contours:
        contour = np.asanyarray(contour)
        cv2.drawContours(drawn_img, contour, -1, color=color, thickness=thickness)
    if dsize is not None:
        dsize = get_dsize(drawn_img, dsize)
        drawn_img = cv2.resize(drawn_img, dsize=dsize)
    return drawn_img


def crop_bbox(
    image: np.array, bbox: BBox, dsize: Optional[Union[float, Tuple[int]]] = None
):
    x_min, y_min, x_max, y_max = bbox.coords
    image = image[y_min:y_max, x_min:x_max]
    if dsize is not None:
        dsize = get_dsize(image, dsize)
        image = cv2.resize(image, dsize=dsize)
    return image


def get_colors(seeds: Union[int, List[str]]):
    seeds_set = set(seeds)
    if isinstance(seeds, int):
        unique_num = seeds
        seeds = range(seeds)
    else:
        unique_num = len(seeds_set)
    if unique_num == 0:
        return []
    colormap = cm.get_cmap("jet")  # you can choose another colormap here
    unique_colors = [colormap(i) for i in np.linspace(0, 1, unique_num)]
    random.seed(seeds[0])
    random.shuffle(unique_colors)
    seeds_dict = {seed: color for seed, color in zip(seeds_set, unique_colors)}
    colors = [seeds_dict[seed] for seed in seeds]
    return colors


def _fig2array(fig, dpi: int = 300, format: str = "png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img_rgba = Image.open(buf)
    img_array = np.array(img_rgba)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    buf.close()
    plt.close(fig)
    return img_bgr


def draw_time_plot(
    name: str,
    names: List[str],
    durations: List[float],
    start_timestamps: List[datetime],
    end_timestamps: List[datetime],
    total_duration: Optional[float] = None,
    return_plot: bool = False,
    add_times: bool = False,
    color_seeds: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    def update_durations(start_timestamps, durations, end_timestamps, add_times=False):
        durations = [int(duration * 1000) for duration in durations]
        if add_times:
            start_str = [start.strftime("%S:%f")[:-4] for start in start_timestamps]
            end_str = [end.strftime("%S:%f")[:-4] for end in end_timestamps]
            durations = [
                f"{st} - {du} - {en}"
                for (st, du, en) in zip(start_str, durations, end_str)
            ]
        return durations

    if (
        not (
            len(names) == len(durations) == len(start_timestamps) == len(end_timestamps)
        )
        or len(names) == 0
    ):
        return
    # Extract the necessary data
    durations = update_durations(
        start_timestamps, durations, end_timestamps, add_times=add_times
    )

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(0.4 * len(names), 0.3 * len(names)))

    # Set the y-axis ticks and labels
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    # Format the x-axis to display dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    if color_seeds is None:
        colors = [None] * len(names)
    else:
        colors = get_colors(color_seeds)

    for i, (start, end, duration, color) in enumerate(
        zip(start_timestamps, end_timestamps, durations, colors)
    ):
        ax.plot([start, end], [i, i], marker="o", linewidth=2, color=color)
        midpoint = start + (end - start) / 2
        ax.text(midpoint, i, f"{duration}", horizontalalignment="center", color="black")

    # Set the plot labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{name} Names")
    title = f"{name} Timeline"
    if total_duration is not None:
        title += f" ({total_duration} secs)"
    ax.set_title(title)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add faint horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Automatically adjust the plot layout
    plt.tight_layout()

    if return_plot:
        return _fig2array(fig)

    plt.show()


def draw_stats(
    stats_dict,
    name: Optional[str] = None,
    num: Optional[int] = None,
    digits: int = 3,
    color_seeds: Optional[List[str]] = None,
    return_plot: bool = False,
):
    # Create lists for the statistics
    stats_mean = []
    stats_std = []
    keys = []

    # Loop through each key-value pair in the dictionary
    for key, values in stats_dict.items():
        # Check if values is a list and contains numbers only
        if isinstance(values, list) and all(
            isinstance(i, (int, float)) for i in values
        ):
            # Append the statistics to the lists
            stats_mean.append(np.mean(values))
            stats_std.append(np.std(values))
            keys.append(key)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    colors = get_colors(color_seeds)

    # Create bar plot with error bars
    # bars = ax.bar(keys, stats_mean, yerr=stats_std, alpha=0.7, color="blue", capsize=10)
    for i in range(len(keys)):
        ax.bar(
            keys[i],
            stats_mean[i],
            yerr=stats_std[i],
            color=colors[i % len(colors)],
            capsize=10,
        )

    ax.set_title(f"{name} ({num} items)")
    ax.set_ylabel("Values")

    # Add mean values on top of each bar
    for i, mean in enumerate(stats_mean):
        ax.text(i, mean, str(round(mean, digits)), ha="center", va="bottom")

    plt.xticks(rotation="vertical")
    plt.tight_layout()
    if return_plot:
        return _fig2array(fig)
    plt.show()
