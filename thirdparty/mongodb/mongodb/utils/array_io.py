from typing import List, Optional, Tuple, Union
import os
import atexit
import hashlib
from glob import glob
from collections import defaultdict
import numpy as np
import cv2
import blosc
from multiprocessing import Pool
import signal
from .tr_image import get_dsize
from .print_colors import pdebug, pwarn
from .decorators import exception_handler

USE_ARRAY_THREADS = True

if USE_ARRAY_THREADS:
    processes = 10
    pool = None
    pending_tasks = 0


def init_pool():
    global pool
    if pool is None:
        pool = Pool(processes=processes, initializer=ignore_sigint)
        for _ in range(processes):
            pool.apply_async(lambda: None)
        atexit.register(stop_workers)
        # with open("/tmp/array_io_tasts.log", "w") as f:
        #     f.write("timestamp,func,path,array_type,pending_tasks")


def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def submit_task(func, *args, **kwargs):
    global pending_tasks
    if pool is None:
        init_pool()
    pending_tasks += 1
    result = pool.apply_async(func, args=args, kwds=kwargs, callback=on_task_complete)
    path, array = args
    # with open("/tmp/array_io_tasts.log", "+a") as f:
    #     f.write(
    #         f"\n{datetime.now()},{func.__name__},{path},{type(array).__name__},{pending_tasks}"
    #     )
    return result


def on_task_complete(_):
    global pending_tasks
    pending_tasks -= 1


def remove_duplicates(directory):
    return
    file_paths = glob(os.path.join(directory, "*"))
    mdf5_dict = defaultdict(list)
    for path in file_paths:
        with open(path, "rb") as file:
            mdf5 = hashlib.md5(file.read()).hexdigest()
        mdf5_dict[mdf5].append(path)
    for paths in mdf5_dict.values():
        if len(paths) > 1:
            rm_paths = paths[1:]
            for path in rm_paths:
                os.remove(path)


@exception_handler
def load_array(
    path, dsize: Optional[Union[float, Tuple[int]]] = None
) -> Optional[np.ndarray]:
    if not path:
        return
    if path.endswith((".png", "jpg")):
        return load_img(path, dsize=dsize)
    elif path.endswith((".np", ".npy")):
        return load_np(path)
    elif path.endswith((".npz")):
        return load_npz(path)
    elif path.endswith((".bin")):
        return load_bin(path)


def load_img(
    path, dsize: Optional[Union[float, Tuple[int]]] = None
) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    if dsize is not None:
        dsize = get_dsize(img, dsize)
        img = cv2.resize(img, dsize=dsize)
    return img


def load_np(path):
    with open(path, "rb") as f:
        array = np.load(f)
    return array


def load_bin(filename):
    with open(filename, "rb") as f:
        compressed_array = f.read()
    return blosc.unpack_array(compressed_array)


def load_npz(path: str) -> List[np.ndarray]:
    container = np.load(path)
    return [container[key] for key in container]


def is_img(array: np.ndarray) -> bool:
    if not isinstance(array, np.ndarray):
        return False
    if array.dtype not in (np.uint8, np.uint16):
        return False
    if not 2 <= array.ndim <= 3:
        return False
    if array.ndim == 3 and array.shape[-1] not in (1, 3):
        return False
    return True


def save_np(path: str, array: np.ndarray):
    with open(path, "wb") as f:
        np.save(f, array)
    remove_duplicates(os.path.dirname(path))


def save_bin(path: str, array: np.ndarray):
    compressed_array = blosc.pack_array(
        array, cname="zstd", clevel=9, shuffle=blosc.SHUFFLE
    )
    with open(path, "wb") as f:
        f.write(compressed_array)


def save_img(path: str, array: np.ndarray):
    cv2.imwrite(path, array)
    remove_duplicates(os.path.dirname(path))


def save_npz(path: str, arrays: List[np.ndarray]):
    np.savez(path, *arrays)
    remove_duplicates(os.path.dirname(path))


def save_array(
    path: str,
    array: Union[Tuple[np.ndarray], np.ndarray],
    use_threads: bool = USE_ARRAY_THREADS,
    no_compress: bool = False,
    force_np: bool = False,
    as_jpg: bool = False,
):
    if not isinstance(path, str):
        return False, path
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except Exception:
            return False, ""

    if isinstance(array, (list, tuple)):
        path += ".npz"
        if use_threads:
            submit_task(save_npz, path, array)
            # arrays_queue.put((save_npz, (path, array), {}))
        else:
            save_npz(path, array)
        return True, path

    if not isinstance(array, np.ndarray):
        return False, ""

    if not force_np and is_img(array):
        if as_jpg:
            path += ".jpg"
        else:
            path += ".png"
        if use_threads:
            submit_task(save_img, path, array)
            # arrays_queue.put((save_png, (path, array), {}))
        else:
            save_img(path, array)
        return True, path

    if array.ndim == 3 and not no_compress:
        path += ".bin"
        if use_threads:
            submit_task(save_bin, path, array)
            # arrays_queue.put((save_bin, (path, array), {}))
        else:
            save_bin(path, array)
        return True, path

    path += ".np"
    if use_threads:
        submit_task(save_np, path, array)
        # arrays_queue.put((save_np, (path, array), {}))
    else:
        save_np(path, array)
    return True, path


def stop_workers():
    if USE_ARRAY_THREADS:
        pool.close()
        pool.join()
        if pending_tasks != 0:
            pwarn(f"pending arrays tasks: {pending_tasks}")
        pdebug(f"Arrays worker terminated")
