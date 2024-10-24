import os
import traceback
from time import time
from datetime import datetime
from dataclasses import asdict
from .log_classes import MethodLog
from .caller_info import CallerInfo

MONGODB_LOGGER_NAME = "mongodb_logger"
METHODS_LOGS_KEY = "methods_logs"
PRINT_EXCEPTIONS = "PRINT_EXCEPTIONS"


def mongodb_method_log(func):
    def wrapper(self, *args, **kwargs):
        mongodb_logger = getattr(self, MONGODB_LOGGER_NAME)
        if mongodb_logger:
            cls_name = self.__class__.__name__
            method_name = func.__name__
            caller_info = CallerInfo(cls_name=cls_name, method_name=method_name)
            mongodb_logger.info(f"method {cls_name}.{method_name} started", caller_info)
            start_time = datetime.now()
            start = time()
        result = func(self, *args, **kwargs)
        if mongodb_logger:
            duration = time() - start
            end_time = datetime.now()
            mongodb_logger.info(
                f"method {cls_name}.{method_name} completed", caller_info
            )
            method_log = MethodLog(
                cls_name, method_name, start_time, end_time, duration
            )
            mongodb_logger.append(
                METHODS_LOGS_KEY,
                asdict(method_log),
                add_metadata=False,
                add_key_prefix=False,
                add_method_in_key=False,
            )

        return result

    return wrapper


# def exception_handler(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except:
#             return None

#     return wrapper


def exception_handler(exception_value=None):
    print_exceptions = os.environ.get(PRINT_EXCEPTIONS, True)
    if exception_value is not None and not callable(exception_value):

        def _decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except:
                    if print_exceptions:
                        print(traceback.format_exc())
                    return exception_value

            return wrapper

        return _decorator
    else:

        def _decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except:
                    if print_exceptions:
                        print(traceback.format_exc())
                    return None

            return wrapper

        return _decorator(exception_value)
