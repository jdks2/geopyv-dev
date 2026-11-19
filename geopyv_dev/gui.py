"""Launcher for the geopyv desktop GUI (the ``geopyv-gui`` command)."""

import sys
import threading

from geopyv_dev import _geopyv_dev


def main():
    """Open the geopyv GUI and block until its window is closed."""
    run_gui = getattr(_geopyv_dev, "run_gui", None)
    if run_gui is None:
        sys.exit("geopyv-gui: this build of geopyv_dev was made without the GUI")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("geopyv-gui must be launched from the main thread")
    try:
        run_gui()
    except RuntimeError as e:
        # e.g. no display available: report it without a traceback.
        sys.exit(str(e))


if __name__ == "__main__":
    main()
