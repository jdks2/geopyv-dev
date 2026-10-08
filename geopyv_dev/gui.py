"""Launcher for the geopyv desktop GUI (the ``geopyv-gui`` command)."""

import sys
import threading

from geopyv_dev._geopyv_dev import run_gui


def main():
    """Open the geopyv GUI and block until its window is closed."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("geopyv-gui must be launched from the main thread")
    try:
        run_gui()
    except RuntimeError as e:
        # e.g. no display available: report it without a traceback.
        sys.exit(str(e))


if __name__ == "__main__":
    main()
