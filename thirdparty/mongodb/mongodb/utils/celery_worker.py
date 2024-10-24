import subprocess
import argparse
import time
import psutil
import os
import signal


def is_celery_worker_running(worker_name: str = ""):
    pids = []
    for process in psutil.process_iter():
        try:
            cmd = " ".join(process.cmdline())
            if (
                "celery" in cmd
                and "-A" in cmd
                and "worker" in cmd
                and "hostname" in cmd
                and worker_name in cmd
            ):
                pids.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids


def start_celery_worker(
    app: str, loglevel: str = "error", workers_num: int = 1, start_flower: bool = False
):
    worker_pids = is_celery_worker_running(app)
    if not worker_pids:
        cmd = [
            "celery",
            "-A",
            app,
            "worker",
            f"--loglevel={loglevel}",
            f"--hostname={app}",
            f"--concurrency={workers_num}",
        ]
        process = subprocess.Popen(cmd)
        time.sleep(0.5)
        print(f"{app} celery worker started with PID: {process.pid}")
        if start_flower:
            cmd = [
                "celery",
                "-A",
                app,
                "flower",
                "--broker_url=redis://localhost:6379/1",
                "--port=5556",
            ]
            subprocess.Popen(cmd)
        return process.pid
    else:
        print(f"{app} celery worker is already running with PIDs: {worker_pids}.")
        return worker_pids


def stop_celery_worker(worker_name="") -> bool:
    def stop_gracefully(pids):
        for pid in pids:
            if pid:
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    print(f"Terminated worker named {worker_name} with PID: {pid}")
                except Exception as e:
                    print(f"Error stopping worker {worker_name}: {e}")
                    return False
        return True

    for _ in range(20):
        pids = is_celery_worker_running(worker_name)
        if not pids:
            print(f"No running Celery worker named {worker_name} found.")
            return True
        stopped = stop_gracefully(pids)
        if stopped is not None:
            return stopped
        pids = is_celery_worker_running(worker_name)
        if not pids:
            break
        time.sleep(0.5)
    pids = is_celery_worker_running(worker_name)
    if not pids:
        return True
    for pid in pids:
        os.kill(pid, signal.SIGKILL)


def restart_celery_worker(
    app: str, loglevel: str = "error", start_flower: bool = False
):
    stop_celery_worker(app)
    start_celery_worker(app, loglevel=loglevel, start_flower=start_flower)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart celery worker")
    parser.add_argument("func", type=str, help="Name of func")
    parser.add_argument("worker_name", type=str, help="Name of app")

    # Add a flag for start_flower. The "action='store_true'" will set the value to True if the flag is present.
    parser.add_argument(
        "--start-flower", action="store_true", help="Start Flower if set"
    )

    args = parser.parse_args()
    if args.func == "start_celery_worker":
        start_celery_worker(args.worker_name, start_flower=args.start_flower)
    elif args.func == "stop_celery_worker":
        stop_celery_worker(args.worker_name)
    elif args.func == "restart_celery_worker":
        restart_celery_worker(args.worker_name, start_flower=args.start_flower)
