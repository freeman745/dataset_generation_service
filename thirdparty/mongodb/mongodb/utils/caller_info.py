import inspect
from dataclasses import dataclass


@dataclass
class CallerInfo:
    cls_name: str = ""
    method_name: str = ""
    lineno: int = -1
    fname: str = ""


def get_caller_info():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    info = inspect.getframeinfo(caller_frame)

    file_name = info.filename
    lineno = info.lineno
    method_name = info.function
    cls_name = ""

    # Check if the function is a method of a class
    if "self" in caller_frame.f_locals:
        cls_name = caller_frame.f_locals["self"].__class__.__name__

    caller_info = CallerInfo(
        cls_name=cls_name, method_name=method_name, lineno=lineno, fname=file_name
    )
    return caller_info
