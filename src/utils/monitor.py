"""
GPU/CPU Utilization Monitor

Samples GPU/CPU utilization in the background during pipeline execution
and generates a timeline graph together with pipeline events (GPU batch, CPU grading).

Usage (standalone):
  python -m src.utils.monitor --output monitor_result.png --duration 60 --gpus 0

Usage (integrated):
  from src.utils.monitor import PipelineMonitor
  mon = PipelineMonitor(gpu_ids=[0], output_dir="./output")
  mon.start()
  # ... run pipeline ...
  mon.mark("gpu_batch_start", batch=0)
  mon.mark("gpu_batch_end", batch=0)
  mon.stop()
  mon.plot()
"""
import os
import time
import json
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Sample:
    timestamp: float           # time.monotonic() relative to start
    cpu_percent: float         # overall CPU utilization %
    gpu_utils: Dict[int, float] = field(default_factory=dict)  # gpu_id → util %
    gpu_mem_used: Dict[int, float] = field(default_factory=dict)  # gpu_id → MB


@dataclass
class Event:
    timestamp: float           # relative to start
    name: str                  # e.g. "gpu_batch_start", "cpu_grade_end"
    meta: Dict = field(default_factory=dict)  # e.g. {"batch": 3, "size": 512}


class PipelineMonitor:
    """Background monitor for GPU/CPU utilization + pipeline events."""

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        interval: float = 0.5,
        output_dir: str = "./output",
        prefix: str = "",
    ):
        self.gpu_ids = gpu_ids or [0]
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        self.samples: List[Sample] = []
        self.events: List[Event] = []
        self._start_time: float = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start background sampling."""
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background sampling."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def mark(self, name: str, **meta):
        """Record a pipeline event at current time."""
        t = time.monotonic() - self._start_time
        self.events.append(Event(timestamp=t, name=name, meta=meta))

    def _sample_loop(self):
        import psutil

        while not self._stop_event.is_set():
            t = time.monotonic() - self._start_time
            cpu = psutil.cpu_percent(interval=None)
            gpu_utils = {}
            gpu_mem = {}

            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0,
                )
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        idx = int(parts[0])
                        if idx in self.gpu_ids:
                            gpu_utils[idx] = float(parts[1])
                            gpu_mem[idx] = float(parts[2])
            except Exception:
                pass

            self.samples.append(Sample(
                timestamp=t, cpu_percent=cpu,
                gpu_utils=gpu_utils, gpu_mem_used=gpu_mem,
            ))
            self._stop_event.wait(self.interval)

    def save_raw(self, path: Optional[str] = None):
        """Save raw samples and events as JSON."""
        out = Path(path) if path else self.output_dir / f"{self.prefix}monitor_raw.json"
        data = {
            "samples": [
                {"t": s.timestamp, "cpu": s.cpu_percent,
                 "gpu_util": s.gpu_utils, "gpu_mem": s.gpu_mem_used}
                for s in self.samples
            ],
            "events": [
                {"t": e.timestamp, "name": e.name, "meta": e.meta}
                for e in self.events
            ],
        }
        out.write_text(json.dumps(data, indent=2))
        return str(out)

    def plot(self, path: Optional[str] = None):
        """Generate timeline plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        out = Path(path) if path else self.output_dir / f"{self.prefix}pipeline_monitor.png"

        if not self.samples:
            print("[monitor] No samples collected.")
            return

        ts = np.array([s.timestamp for s in self.samples])
        cpus = np.array([s.cpu_percent for s in self.samples])

        # Collect per-GPU data
        gpu_data = {}
        for gid in self.gpu_ids:
            gpu_data[gid] = np.array([
                s.gpu_utils.get(gid, 0) for s in self.samples
            ])

        n_gpus = len(self.gpu_ids)
        n_plots = 1 + n_gpus  # CPU + each GPU
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots),
                                 sharex=True, gridspec_kw={"hspace": 0.08})
        if n_plots == 1:
            axes = [axes]

        # ── Color map for events ──
        event_colors = {
            "gpu_batch_start": "#2196F3",
            "gpu_batch_end":   "#1565C0",
            "cpu_grade_start": "#FF9800",
            "cpu_grade_end":   "#E65100",
        }

        def _draw_events(ax):
            for ev in self.events:
                color = event_colors.get(ev.name, "gray")
                alpha = 0.6
                ax.axvline(ev.timestamp, color=color, alpha=alpha,
                           linewidth=0.8, linestyle="--")

        def _draw_spans(ax, start_name, end_name, color, label):
            """Draw shaded spans between paired start/end events."""
            starts = [e for e in self.events if e.name == start_name]
            ends = [e for e in self.events if e.name == end_name]
            drawn_label = False
            for s, e in zip(starts, ends):
                lbl = label if not drawn_label else None
                ax.axvspan(s.timestamp, e.timestamp, alpha=0.15,
                           color=color, label=lbl)
                drawn_label = True

        # ── GPU plots ──
        for i, gid in enumerate(self.gpu_ids):
            ax = axes[i]
            ax.fill_between(ts, gpu_data[gid], alpha=0.4, color="#2196F3")
            ax.plot(ts, gpu_data[gid], color="#1565C0", linewidth=0.8)
            ax.set_ylabel(f"GPU {gid}\nUtil %", fontsize=10)
            ax.set_ylim(-5, 105)
            ax.grid(axis="y", alpha=0.3)
            _draw_spans(ax, "gpu_batch_start", "gpu_batch_end", "#2196F3", "GPU batch")
            _draw_spans(ax, "cpu_grade_start", "cpu_grade_end", "#FF9800", "CPU grading")
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        # ── CPU plot ──
        ax_cpu = axes[-1]
        ax_cpu.fill_between(ts, cpus, alpha=0.4, color="#FF9800")
        ax_cpu.plot(ts, cpus, color="#E65100", linewidth=0.8)
        ax_cpu.set_ylabel("CPU\nUtil %", fontsize=10)
        ax_cpu.set_ylim(-5, 105)
        ax_cpu.set_xlabel("Time (seconds)", fontsize=11)
        ax_cpu.grid(axis="y", alpha=0.3)
        _draw_spans(ax_cpu, "gpu_batch_start", "gpu_batch_end", "#2196F3", "GPU batch")
        _draw_spans(ax_cpu, "cpu_grade_start", "cpu_grade_end", "#FF9800", "CPU grading")
        ax_cpu.legend(loc="upper right", fontsize=8)

        # ── Title ──
        total_time = ts[-1] if len(ts) > 0 else 0
        n_gpu_batches = sum(1 for e in self.events if e.name == "gpu_batch_end")
        n_cpu_batches = sum(1 for e in self.events if e.name == "cpu_grade_end")
        fig.suptitle(
            f"Pipeline Monitor — {total_time:.1f}s total, "
            f"{n_gpu_batches} GPU batches, {n_cpu_batches} CPU grading batches",
            fontsize=13, y=0.98,
        )

        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[monitor] Saved plot: {out}")
        return str(out)
