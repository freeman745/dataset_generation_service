from typing import Callable, List, Tuple, Union, Sequence, Dict, Any, Optional, Type
import os
import numpy as np
import hashlib
import time
from datetime import datetime
import csv
import random
import string
import statistics
import git

from .decorators import exception_handler

CSV_FILE_PATH = "/tmp/mongodb_timing_{}.csv"
total_time = 0
active_time_logging = False  # True


def apply_all(
    func: Callable,
    data: Union[Sequence, Dict, Any],
    target_type: Optional[Type] = None,
    depth: Optional[int] = None,
):
    if (depth is None) or (depth > 0):
        if isinstance(data, (list, tuple)):
            new_data = [
                apply_all(
                    func,
                    value,
                    target_type=target_type,
                    depth=None if (depth is None) else depth - 1,
                )
                for value in data
            ]
            if isinstance(data, tuple):
                return tuple(new_data)
            return new_data
        if isinstance(data, Dict):
            return {
                key: apply_all(
                    func,
                    value,
                    target_type=target_type,
                    depth=None if (depth is None) else depth - 1,
                )
                for key, value in data.items()
            }
    if (target_type is None) or isinstance(data, target_type):
        return func(data)
    return data


def np2normal(data, depth: Optional[int] = None):
    data = apply_all(np.ndarray.tolist, data, target_type=np.ndarray, depth=depth)
    data = apply_all(np.number.item, data, target_type=np.number, depth=depth)
    return data


def get_nested_item(iterable, index, default_for_nonexistent=False, default=None):
    def handle(iterable, index, default_for_nonexistent, default):
        try:
            return iterable[index]
        except (IndexError, KeyError, TypeError) as e:
            if default_for_nonexistent or default is not None:
                return default
            else:
                raise e

    if isinstance(index, (int, str, type(None))):
        return handle(iterable, index, default_for_nonexistent, default)
    if isinstance(index, (list, tuple)):
        result = iterable
        for ind in index:
            result = handle(result, ind, default_for_nonexistent, default)
        return result
    raise TypeError(
        f"index must be int, str, None or list/tuple of them. <{index}> was passed"
    )


def has_index(lst, index):
    try:
        _ = lst[index]
        return True
    except IndexError:
        return False


@exception_handler
def get_agregated_duration(
    intervals: List[Tuple[Optional[datetime], Optional[datetime]]]
) -> Optional[float]:
    durations = _get_agregated_durations(intervals)
    if durations:
        total_time = sum(durations)
        return total_time


@exception_handler
def get_agregated_std(
    intervals: List[Tuple[Optional[datetime], Optional[datetime]]]
) -> Optional[float]:
    durations = _get_agregated_durations(intervals)
    std = statistics.stdev(durations)
    return std


def _get_agregated_durations(
    intervals: List[Tuple[Optional[datetime], Optional[datetime]]]
) -> Optional[List[float]]:
    if len(intervals) == 0:
        return 0
    if any([any([None in interval]) for interval in intervals]):
        return
    intervals.sort(key=lambda x: x[0])
    stack = []
    stack.append(intervals[0])
    for i in range(1, len(intervals)):
        if stack[-1][1] < intervals[i][0]:
            stack.append(intervals[i])
        elif stack[-1][1] < intervals[i][1]:
            stack[-1] = (stack[-1][0], intervals[i][1])
    durations = [(end - start).total_seconds() for start, end in stack]
    return durations


def generate_random_string(length):
    characters = string.ascii_lowercase + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def are_same_files(paths):
    if len(paths) < 2:
        return False
    sha_set = set()
    for path in paths:
        with open(path, "rb") as f:
            bytes = f.read()
            sha = hashlib.sha256(bytes).hexdigest()
            sha_set.add(sha)
    return len(sha_set) == 1


def create_csv_file_if_not_exists(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["timestamp", "func_name", "duration", "total"])


def get_cls_attributes(obj):
    return [
        attr
        for attr in dir(obj)
        if not attr.startswith("__") and not callable(getattr(obj, attr))
    ]


def log_time(func):
    def wrapper(*args, **kwargs):
        global total_time
        if not active_time_logging:
            return func(*args, **kwargs)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Log the measurements to the CSV file
        datestamp = datetime.now().strftime("%Y%m%d")
        csv_path = CSV_FILE_PATH.format(datestamp)
        create_csv_file_if_not_exists(csv_path)
        with open(csv_path, "a") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    timestamp,
                    func.__name__,
                    f"{elapsed_time:.5f}",
                    f"{total_time:.5f}",
                ]
            )
        return result

    return wrapper


def get_git_info(path: Optional[str] = None, search_parent_directories: bool = True):
    try:
        repo = git.Repo(path, search_parent_directories=search_parent_directories)
        sha = repo.head.object.hexsha
        branch = repo.active_branch.name
        return {"sha": sha, "branch": branch}
    except Exception as e:
        print(e)
        return {"sha": "", "branch": ""}
