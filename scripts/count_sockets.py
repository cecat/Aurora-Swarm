#!/usr/bin/env python3
"""Monitor socket count for a process by PID.

Reads /proc/<pid>/fd, counts file descriptors that are sockets,
and prints the count periodically. Useful for observing socket
usage of tree_reduce_coli or other Aurora-Swarm clients.

Usage:
    python scripts/count_sockets.py <pid> [interval_sec]

Examples:
    python scripts/count_sockets.py 12345
    python scripts/count_sockets.py 12345 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def count_sockets(pid: int) -> int:
    """Count socket file descriptors for the given process."""
    fd_dir = Path(f"/proc/{pid}/fd")
    if not fd_dir.exists():
        return -1
    count = 0
    try:
        for fd in fd_dir.iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith("socket:"):
                    count += 1
            except (OSError, FileNotFoundError):
                pass
    except Exception:
        return -1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pid",
        type=int,
        help="Process ID to monitor",
    )
    parser.add_argument(
        "interval",
        type=float,
        nargs="?",
        default=10.0,
        help="Seconds between samples (default: 10)",
    )
    args = parser.parse_args()

    pid = args.pid
    interval = args.interval

    proc_path = Path(f"/proc/{pid}")
    if not proc_path.exists():
        print(f"Error: Process {pid} not found", file=sys.stderr)
        return 1

    try:
        while True:
            n = count_sockets(pid)
            if n < 0:
                print(f"Error: Cannot read /proc/{pid}/fd (process may have exited)")
                return 1
            print(f"Sockets in use (pid={pid}): {n}")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
