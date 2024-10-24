import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from .array_io import load_array


# -----------------------------------------------------------------
# ---- Helper functions fro displaying images via the matplotlib
# ---- interface.
# -----------------------------------------------------------------


def _show_img_data(image, cmap=None, vmin=None, vmax=None, use_minmax=True):
    if cmap is None and (image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1)):
        cmap = "gray"
    if use_minmax:
        vmin, vmax = image.min(), image.max()
    elif vmin is None and vmax is None:
        if image.dtype in (np.uint8, np.uint16):
            pass
        elif image.min() > 0:
            vmin = 0
            vmax = 1 if image.max() <= 1 else 255
        else:
            vmin, vmax = image.min(), image.max()
    img_data = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    return img_data


def plot_pcd(
    pcd,
    initial_point_size=0.02,
    plane_model=None,
    reverse_axes: Optional[Union[List[int], int]] = None,
    *args,
    **kwargs
):
    import open3d as o3d
    from pyntcloud import PyntCloud
    import pandas as pd

    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd = PyntCloud.from_instance("open3d", pcd)

    if plane_model is not None:
        a, b, c, d = plane_model
        x, y = np.meshgrid(np.linspace(-100, 100, 50), np.linspace(-100, 100, 50))
        z = -(a * x + b * y + d) / c
        plane_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        plane_pcd = PyntCloud(pd.DataFrame(plane_points, columns=["x", "y", "z"]))

        pcd_df = pd.DataFrame(pcd.xyz, columns=["x", "y", "z"])
        plane_pcd_df = pd.DataFrame(plane_pcd.xyz, columns=["x", "y", "z"])
        merged_pcd_df = pd.concat([pcd_df, plane_pcd_df])
        pcd = PyntCloud(merged_pcd_df)

    if reverse_axes is not None:
        if isinstance(reverse_axes, int):
            reverse_axes = [reverse_axes]
        for axis in reverse_axes:
            pcd.xyz[:, axis] *= -1
            pcd.xyz[:, axis] += np.abs(pcd.xyz[:, axis].min()) + np.abs(
                pcd.xyz[:, axis].max()
            )

    pcd.plot(initial_point_size=initial_point_size, *args, **kwargs)


def plot_pcds(
    *pcds,
    initial_point_size=0.02,
    reverse_axes: Optional[Union[List[int], int]] = None,
    **kwargs
):
    import open3d as o3d
    from pyntcloud import PyntCloud
    import pandas as pd

    def get_random_color(seed=None):
        np.random.seed(seed)
        return np.random.randint(0, 256, size=3).tolist()

    if not pcds:
        return
    pcds = list(pcds)
    colored_pcds = []
    for i in range(len(pcds)):
        if isinstance(pcds[i], o3d.geometry.PointCloud):
            pcds[i] = np.asarray(pcds[i].points)
        color = np.tile(get_random_color(i), (pcds[i].shape[0], 1))
        colored_pcds.append(color)
    combined_xyz = np.vstack(pcds)
    combined_colors = np.vstack(colored_pcds)

    df_combined = pd.DataFrame(
        data={
            "x": combined_xyz[:, 0],
            "y": combined_xyz[:, 1],
            "z": combined_xyz[:, 2],
            "red": combined_colors[:, 0],
            "green": combined_colors[:, 1],
            "blue": combined_colors[:, 2],
        }
    )
    combined_cloud = PyntCloud(df_combined)
    if reverse_axes is not None:
        if isinstance(reverse_axes, int):
            reverse_axes = [reverse_axes]
        for axis in reverse_axes:
            combined_cloud.xyz[:, axis] *= -1
            combined_cloud.xyz[:, axis] += np.abs(
                combined_cloud.xyz[:, axis].min()
            ) + np.abs(combined_cloud.xyz[:, axis].max())

    combined_cloud.plot(initial_point_size=initial_point_size, **kwargs)


def show_image(
    image,
    fig_size=8,
    title=None,
    fontdict=None,
    cmap=None,
    vmin=None,
    vmax=None,
    use_minmax=True,
    bgr2rgb=True,
):
    """
    Display an image
    :param image: image to display
    :param fig_size: size of the figure
    :param cmap: colormap
    :param vmin: minimum of the data range that the colormap covers
    :param vmax: maximum of the data range that the colormap covers
    :return:
    """
    if isinstance(image, str):
        image = load_array(image)
    if isinstance(fig_size, int):
        fig_size = [fig_size] * 2
    plt.figure(figsize=fig_size)
    if bgr2rgb and image.ndim == 3:
        image = image[..., ::-1]
    if title is not None:
        if fontdict is None:
            fontdict = {"color": "gray", "fontsize": fig_size[0] + 4}
        plt.title(title, fontdict=fontdict)
    plt.imshow(
        image,
        **_show_img_data(image, cmap=cmap, vmin=vmin, vmax=vmax, use_minmax=use_minmax)
    )
    plt.show()


def show_images(
    *images: Union[Sequence[np.ndarray], Sequence[str]],
    fig_size: Union[int, Tuple[int, int]] = 8,
    titles: Optional[Sequence[str]] = None,
    fontdict: Optional[dict] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = -1,
    cmap=None,
    vmin=None,
    vmax=None,
    use_minmax=True,
    bgr2rgb=True
):
    """Display multiple images

    Args:
        *images: list of np.ndarray or image/dcm files or file paths to be displayed.
        fig_size (Union[int, Tuple[int, int]], optional): Size of each image. Defaults to 8.
        titles (Optional[Sequence[str]], optional): Optional list of images titles. Defaults to None.
        fontdict (Optional[dict], optional): dict with title plot configuration (color/fontsize etc). Defaults to None.
        rows (Optional[int], optional): How many rows will be shown. Defaults to None.
        cols (Optional[int], optional): How many columns (images per row) will be shown. If rows is defined cols will be adapted. Defaults to 1.
        cmap ([type], optional): [description]. Defaults to None.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        bgr2rgb (bool, optional): Change from BGR to RGB since images are read using cv2 and plot using plt. Defaults to True.
    """

    # ---- Determine number of rows and columns
    if cols == -1:
        cols = len(images)
        if cols > 10:
            print(
                "Too many images to be displayed in one row. Consider setting <cols> and try again."
            )
            return
    if rows is None:
        rows = np.ceil(len(images) / cols).astype(int)
    cols = max(cols, np.ceil(len(images) / rows).astype(int))

    # ---- If images are image paths, call show_thumbs
    if all([isinstance(img, str) for img in images]):
        show_thumbs(
            images,
            fig_size=fig_size,
            titles=titles,
            fontdict=fontdict,
            cols=cols,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            bgr2rgb=bgr2rgb,
        )
        return

    # ---- Determine images titles
    if titles is None:
        titles = [None] * len(images)

    # ---- Determine figure size
    if isinstance(fig_size, int):
        fig_size = [fig_size] * 2
    fig_size = (fig_size[0]) * cols, (fig_size[1]) * rows

    # ---- Plot images
    if all([image is None for image in images]):
        return
    fig = plt.figure(figsize=fig_size)
    fig.tight_layout()
    for i, (image, title) in enumerate(zip(images, titles), 1):
        if image is None:
            continue
        if isinstance(image, str):
            image = load_array(image)
        fig.add_subplot(rows, cols, i)
        if bgr2rgb and image.ndim == 3:
            image = image[..., ::-1]

        if title is not None:
            if fontdict is None:
                fontdict = {"color": "gray", "fontsize": int(fig_size[0] / cols) + 4}
            plt.title(title, fontdict=fontdict)
        plt.imshow(
            image,
            **_show_img_data(
                image, cmap=cmap, vmin=vmin, vmax=vmax, use_minmax=use_minmax
            )
        )
    plt.show()


def show_thumbs(
    img_paths: Union[str, Sequence[str]],
    only_name: bool = True,
    cols: int = 6,
    fig_size: Union[int, Tuple[int, int]] = 3,
    sep: str = "   |   ",
    titles: Optional[Sequence[str]] = None,
    names_for_titles: bool = True,
    as_title: bool = True,
    max_imgs: Optional[int] = None,
    *args,
    **kwargs
):
    """Similar to show_images() but accepts only list of image/dcm file paths.

    Args:
        img_paths (Union[str, Sequence[str]): List of image/dicom file paths or path of the directory that includes them.
        only_name (bool, optional): If no titles are passed and if it is True, the names of the files will be printed. Defaults to True.
        cols (int, optional): How many columns (images per row) will be shown. Defaults to 6.
        fig_size (int, optional): Size of each image. Defaults to 3.
        sep (str, optional): Separation string between printed image titles. Defaults to ', '.
        titles (Optional[Sequence[str]], optional): Optional list of images titles. Defaults to None.
        names_for_titles (bool, optional): If no titles are passed and this is True, the names of the files will be used as titles. Defaults to True.
        as_title (bool, optional): If True the images names will be plotted as titles. If False will be printed as text. Defaults to True.
    """

    def show_row(row, only_name, fig_size, sep, titles, as_title, *args, **kwargs):
        if len(row) == 0:
            return
        images = [load_array(img_path) for img_path in row]
        if titles is not None:
            row = titles
        elif only_name:
            row = [os.path.basename(p) for p in row]
        if as_title:
            show_images(*images, rows=1, fig_size=fig_size, titles=row, *args, **kwargs)
        else:
            print(sep.join(row))
            show_images(*images, rows=1, fig_size=fig_size, *args, **kwargs)

    if isinstance(img_paths, str):
        if "*" in img_paths:
            img_paths = glob(img_paths)
        elif os.path.isdir(img_paths):
            img_paths = [os.path.join(img_paths, p) for p in os.listdir(img_paths)]
    row, r_titles = [], []
    if titles is None:
        if names_for_titles:
            titles = [os.path.basename(p) for p in img_paths]
        else:
            titles = [None] * len(img_paths)
    titles, img_paths = titles[:max_imgs], img_paths[:max_imgs]
    for title, img_path in zip(titles, img_paths):
        row.append(img_path)
        r_titles.append(title)
        if len(row) == cols:
            show_row(row, only_name, fig_size, sep, r_titles, as_title, *args, **kwargs)
            row, r_titles = [], []
    show_row(row, only_name, fig_size, sep, r_titles, as_title, *args, **kwargs)


def visualize_depth(depth_map, replace_nan=False):
    import plotly.graph_objects as go

    def _replace_nan(array):
        mask = np.isnan(array)
        array[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask]
        )
        return array

    if replace_nan:
        depth_map = _replace_nan(depth_map)

    y, x = np.mgrid[0 : depth_map.shape[0], 0 : depth_map.shape[1]]
    fig = go.Figure(data=[go.Surface(z=depth_map, x=x, y=y)])
    fig.update_layout(scene=dict(aspectmode="auto"))
    fig.show()
