#!/usr/bin/env python3
"""Phase 2 FD/socket/shm behavior probe.

Holds an open TCP listener and a POSIX shared-memory segment, then idles, so
CRIU dump behavior for those resources can be observed. Usage:
    python p2_fd_socket.py <status_file>
"""
import mmap
import os
import socket
import sys
import time
from multiprocessing import shared_memory


def main() -> None:
    status = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p2_fd/state"
    os.makedirs(os.path.dirname(status), exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    shm = shared_memory.SharedMemory(create=True, size=mmap.PAGESIZE)
    shm.buf[0] = 42

    with open(status, "w") as f:
        f.write(f"ready pid={os.getpid()} port={srv.getsockname()[1]} "
                f"shm={shm.name}\n")

    n = 0
    while True:
        n += 1
        with open(status, "w") as f:
            f.write(f"{n} pid={os.getpid()} port={srv.getsockname()[1]} "
                    f"shm={shm.name} shm0={shm.buf[0]}\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
