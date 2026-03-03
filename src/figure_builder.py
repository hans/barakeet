"""
FigureBuilder — staged figure construction for talk slide builds.

Build a matplotlib figure incrementally, call .stage() to snapshot,
then .render() to export all frames with **identical geometry**
(axes positions, figure size) locked to the final frame's layout.

Design
------
Instead of replaying draw commands, each .stage() call pickles the
entire figure state.  At render time we:
1. Unpickle the *final* stage, run the layout engine, and record
   every axes' position as the reference geometry.
2. For each earlier stage, unpickle its figure, force the same
   axes positions, and save.

This means you can use the normal imperative matplotlib API freely —
ax.plot(), ax.legend(), fig.colorbar(), etc. — with no wrappers.

Usage
-----
    from figure_builder import FigureBuilder

    fb = FigureBuilder(figsize=(8, 5))
    ax = fb.ax

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    fb.stage("axes")

    ax.plot(x, y1, label="Signal A")
    fb.stage("line_a")

    ax.plot(x, y2, label="Signal B")
    ax.legend(loc="upper right")
    ax.set_title("Two signals")
    fb.stage("final")

    paths = fb.render("./build_frames/", fmt="pdf", dpi=300)
"""

from __future__ import annotations

import io
import pickle
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


class FigureBuilder:
    """Incremental figure builder with geometry-locked frame export.

    Parameters
    ----------
    figsize : tuple[float, float]
        Figure size in inches.
    nrows, ncols : int
        Subplot grid dimensions.
    gridspec_kw : dict | None
        Forwarded to ``Figure.subplots``.
    constrained_layout : bool
        Use constrained layout (recommended).
    subplot_kw : dict | None
        Forwarded to subplot creation.
    sharex, sharey : bool | str
        Shared axes for multi-panel figures. Forwarded to ``Figure.subplots``.
    style : str | None
        Matplotlib style name to apply.
    fig_kw : dict | None
        Extra kwargs for ``plt.figure()``.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (8, 5),
        nrows: int = 1,
        ncols: int = 1,
        gridspec_kw: dict | None = None,
        constrained_layout: bool = True,
        subplot_kw: dict | None = None,
        sharex: bool | str = False,
        sharey: bool | str = False,
        style: str | None = None,
        fig_kw: dict | None = None,
    ):
        self._figsize = figsize
        self._nrows = nrows
        self._ncols = ncols
        self._gridspec_kw = gridspec_kw or {}
        self._constrained_layout = constrained_layout
        self._subplot_kw = subplot_kw or {}
        self._sharex = sharex
        self._sharey = sharey
        self._style = style
        self._fig_kw = fig_kw or {}

        self._stage_names: list[str] = []
        self._stage_snapshots: list[bytes] = []  # pickled figures
        self._finalized = False

        # Create the live figure
        self._fig, self._axes = self._make_figure()

    def _make_figure(self) -> tuple[Figure, np.ndarray | Axes]:
        fig_kw = dict(figsize=self._figsize, **self._fig_kw)
        if self._constrained_layout:
            fig_kw["layout"] = "constrained"

        fig = plt.figure(**fig_kw)

        if self._nrows == 1 and self._ncols == 1:
            axes = fig.add_subplot(111, **self._subplot_kw)
        else:
            axes = fig.subplots(
                self._nrows, self._ncols,
                gridspec_kw=self._gridspec_kw,
                subplot_kw=self._subplot_kw,
                sharex=self._sharex,
                sharey=self._sharey,
            )
        return fig, axes

    # ---- Public properties ----

    @property
    def fig(self) -> Figure:
        return self._fig

    @property
    def ax(self) -> Axes:
        if isinstance(self._axes, np.ndarray):
            return self._axes.flat[0]
        return self._axes

    @property
    def axes(self) -> np.ndarray | Axes:
        return self._axes

    @property
    def stages(self) -> list[str]:
        return list(self._stage_names)

    # ---- Core API ----

    def stage(self, name: str) -> "FigureBuilder":
        """Snapshot the current figure state as a named build stage.

        Call this after each logical group of drawing operations.
        """
        if self._finalized:
            raise RuntimeError("Cannot add stages after render().")
        if name in self._stage_names:
            raise ValueError(f"Duplicate stage name: {name!r}")

        # Pickle the current figure
        buf = io.BytesIO()
        pickle.dump(self._fig, buf)
        self._stage_snapshots.append(buf.getvalue())
        self._stage_names.append(name)
        return self

    def render(
        self,
        output_dir: str | Path = "./build_frames",
        fmt: str = "pdf",
        dpi: int = 300,
        prefix: str = "",
        pad_index: bool = True,
        close: bool = True,
        transparent: bool = False,
    ) -> list[Path]:
        """Render all stages with axes geometry locked to the final frame.

        Parameters
        ----------
        output_dir : path-like
            Output directory (created if needed).
        fmt : str
            File format: ``"pdf"``, ``"png"``, ``"svg"``, etc.
        dpi : int
            Resolution for raster formats.
        prefix : str
            Filename prefix.
        pad_index : bool
            Zero-pad indices in filenames.
        close : bool
            Close figures after saving.
        transparent : bool
            Transparent figure background.

        Returns
        -------
        list[Path]
            Paths to saved files in stage order.
        """
        if not self._stage_snapshots:
            raise RuntimeError("No stages recorded.")

        self._finalized = True
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(self._stage_snapshots)
        idx_w = len(str(n - 1)) if pad_index else 0

        # --- 1. Restore final figure, resolve layout, capture geometry ---
        ref_fig = pickle.loads(self._stage_snapshots[-1])
        ref_fig.canvas.draw()
        ref_positions = _capture_positions(ref_fig)
        ref_size = ref_fig.get_size_inches().copy()
        plt.close(ref_fig)

        # --- 2. Render each stage with locked geometry ---
        saved: list[Path] = []
        for i, (name, snap) in enumerate(
            zip(self._stage_names, self._stage_snapshots)
        ):
            fig_i = pickle.loads(snap)
            fig_i.set_size_inches(ref_size)

            # Let layout engine run first
            fig_i.canvas.draw()

            # Override with reference positions
            _apply_positions(fig_i, ref_positions)

            # Second draw to render with corrected positions
            fig_i.canvas.draw()

            fname = f"{prefix}{i:0{idx_w}d}_{name}.{fmt}"
            fpath = output_dir / fname
            fig_i.savefig(fpath, dpi=dpi, transparent=transparent)
            saved.append(fpath)
            if close:
                plt.close(fig_i)

        if close:
            plt.close(self._fig)

        return saved

    def preview(self, stage: int | str | None = None) -> Figure:
        """Show a stage interactively. None = last stage."""
        idx = self._resolve_stage(stage)
        fig = pickle.loads(self._stage_snapshots[idx])
        fig.canvas.draw()
        plt.show()
        return fig

    def _resolve_stage(self, stage) -> int:
        if stage is None:
            return len(self._stage_names) - 1
        if isinstance(stage, int):
            return stage
        return self._stage_names.index(stage)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _capture_positions(fig: Figure) -> list[tuple]:
    """Record (x0, y0, width, height) for every axes."""
    positions = []
    for ax in fig.get_axes():
        pos = ax.get_position()
        positions.append((pos.x0, pos.y0, pos.width, pos.height))
    return positions


def _apply_positions(fig: Figure, positions: list[tuple]):
    """Lock every axes to the reference positions."""
    fig.set_layout_engine("none")
    for ax, (x0, y0, w, h) in zip(fig.get_axes(), positions):
        ax.set_position(mpl.transforms.Bbox.from_bounds(x0, y0, w, h))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Generate a demo build and return the saved paths."""
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 200)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-x / 10)

    fb = FigureBuilder(figsize=(8, 4.5))
    ax = fb.ax

    # Stage 0: empty axes
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.3, 1.3)
    fb.stage("axes_only")

    # Stage 1: first line
    ax.plot(x, y1, color="#1f77b4", lw=1.8, label="sin(t)")
    fb.stage("sin")

    # Stage 2: + second line
    ax.plot(x, y2, color="#ff7f0e", lw=1.8, label="cos(t)")
    fb.stage("sin_cos")

    # Stage 3: + third line, legend, title
    ax.plot(x, y3, color="#2ca02c", lw=1.8, label=r"sin(t)$\cdot e^{-t/10}$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Three waveforms", fontsize=13)
    fb.stage("full")

    paths = fb.render("./demo_build", fmt="png", dpi=200)
    print("Saved frames:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    demo()
