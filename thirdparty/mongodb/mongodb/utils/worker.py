import threading
import multiprocessing
import queue

thread_terminate = threading.Event()
process_terminate = multiprocessing.Event()


class WorkerThread(threading.Thread):
    def __init__(self, shared_queue: queue.Queue):
        super().__init__()
        self.shared_queue = shared_queue

    def run(self):
        while not thread_terminate.is_set():
            try:
                task, args, kwargs = self.shared_queue.get(block=True, timeout=1)
                task(*args, **kwargs)
                self.shared_queue.task_done()
            except queue.Empty:
                pass

    def stop(self):
        thread_terminate.set()
        self.join()


class WorkerProcess(multiprocessing.Process):
    def __init__(self, shared_queue: multiprocessing.JoinableQueue):
        super().__init__()
        self.shared_queue = shared_queue

    def run(self):
        while not process_terminate.is_set():
            try:
                task, args, kwargs = self.shared_queue.get(block=True, timeout=1)
                task(*args, **kwargs)
                self.shared_queue.task_done()
            except multiprocessing.queues.Empty:
                pass

    def stop(self):
        process_terminate.set()
        self.join()


def worker_function(input_queue: multiprocessing.Queue):
    while True:
        task = input_queue.get()
        if task is None:
            break

        func, args, kwargs = task
        print(f"Executing task: {func.__name__}, args: {args}, kwargs: {kwargs}")
        func(*args, **kwargs)
        print(f"Task completed: {func.__name__}, args: {args}, kwargs: {kwargs}")
