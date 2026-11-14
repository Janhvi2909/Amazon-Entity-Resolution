"""One status file, refreshed every 2 minutes: stage, parts, rows, measured rate, ETA, disk, memory.

The rate is measured over the last 10 minutes of this process only (resumed work never
counts), and the ETA is remaining units / that rate. Both say how they were measured.
"""
from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from pathlib import Path


def _rss_gb() -> tuple[float, float]:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        peak = float("nan")
    cur = float("nan")
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 2**20
    except Exception:
        pass
    return cur, peak


def cgroup_mem() -> tuple[float, float]:
    """(used GB, limit GB) of this container, or host values if no cgroup limit."""
    try:
        used = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        return used, (int(lim) / 2**30 if lim != "max" else float("inf"))
    except Exception:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return (vm.total - vm.available) / 2**30, vm.total / 2**30
        except Exception:
            return float("nan"), float("nan")


class Progress:
    def __init__(self):
        self.path: Path | None = None
        self.stage = "-"
        self.unit = "parts"
        self.done = 0          # units finished in this process
        self.skipped = 0       # units found complete on disk (resumed)
        self.total = 0
        self.rows = 0
        self.note = ""
        self.t0 = time.time()
        self.stage_t0 = time.time()
        self.hist: deque = deque()
        self.lock = threading.Lock()

    def start(self, path: Path, every: int = 120) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._loop, args=(every,), daemon=True).start()

    def begin(self, stage: str, total: int = 0, unit: str = "parts", skipped: int = 0) -> None:
        with self.lock:
            self.stage, self.total, self.unit, self.skipped = stage, total, unit, skipped
            self.done, self.rows, self.note = 0, 0, ""
            self.stage_t0 = time.time()
            self.hist = deque([(time.time(), 0)])
        self.write()

    def step(self, units: int = 1, rows: int = 0, note: str = "") -> None:
        with self.lock:
            self.done += units
            self.rows += rows
            if note:
                self.note = note
            self.hist.append((time.time(), self.done))
            while len(self.hist) > 2 and self.hist[1][0] < time.time() - 600:
                self.hist.popleft()
        self.write()

    def text(self) -> str:
        now = time.time()
        with self.lock:
            (t_a, d_a), (t_b, d_b) = self.hist[0], self.hist[-1]
            rate = (d_b - d_a) / (t_b - t_a) if t_b > t_a and d_b > d_a else 0.0
            remaining = max(self.total - self.skipped - self.done, 0)
            eta = f"{remaining / rate / 60:.1f} min (remaining {remaining} {self.unit} / rate over last {(t_b - t_a) / 60:.1f} min)" \
                if rate > 0 else "unknown (no unit finished in this process yet)"
            lines = [
                f"updated      {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now))} UTC",
                f"stage        {self.stage}  ({(now - self.stage_t0) / 60:.1f} min in stage, {(now - self.t0) / 60:.1f} min total)",
                f"{self.unit:12s} {self.done} done now + {self.skipped} resumed / {self.total}",
                f"rows         {self.rows:,} processed in this stage",
                f"rate         {rate * 60:.2f} {self.unit}/min measured over recent window",
                f"eta          {eta}",
            ]
        if self.path is not None:
            du = shutil.disk_usage(self.path.parent)
            lines.append(f"disk         {du.free / 2**30:.1f} GB free of {du.total / 2**30:.1f} GB")
        cur, peak = _rss_gb()
        used, lim = cgroup_mem()
        lines.append(f"memory       rss {cur:.1f} GB, peak rss {peak:.1f} GB, container {used:.1f} / {lim:.1f} GB")
        if self.note:
            lines.append(f"note         {self.note}")
        return "\n".join(lines) + "\n"

    def write(self) -> None:
        if self.path is None:
            return
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(self.text())
            tmp.replace(self.path)
        except Exception:
            pass

    def _loop(self, every: int) -> None:
        while True:
            time.sleep(every)
            self.write()


PROGRESS = Progress()
