"""Render ASCII architecture diagrams from docs/matrix_shapes.md to PNG."""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("analytics", "presentation")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join("docs", "matrix_shapes.md"), encoding="utf-8") as f:
    md = f.read()

blocks = re.findall(r"```\n(.*?)```", md, re.DOTALL)
VILLAIN_DIAGRAM = blocks[0].strip()
HERO_DIAGRAM = blocks[1].strip()
DISCOVERY_DIAGRAM = blocks[2].strip()


def render(text, title, filename, fontsize=9):
    lines = text.split("\n")
    num_lines = len(lines)
    max_width = max(len(l) for l in lines)
    fig_w = max(max_width * 0.088, 8)
    fig_h = max(num_lines * 0.28 + 1.4, 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#1e1e2e")

    ax.text(0.5, 0.97, title,
            transform=ax.transAxes, fontsize=fontsize + 4,
            fontweight="bold", color="#cdd6f4",
            ha="center", va="top", fontfamily="monospace")

    ax.text(0.02, 0.90, text,
            transform=ax.transAxes, fontsize=fontsize,
            color="#a6e3a1", fontfamily="monospace",
            va="top", ha="left", linespacing=1.35)

    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=300,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ok {out}")


if __name__ == "__main__":
    print(f"Rendering architecture diagrams -> {OUT_DIR}/")
    render(VILLAIN_DIAGRAM,
           "ACT 1 - VILLAIN: SASRec + Position Bias",
           "arch_villain.png", fontsize=8)
    render(HERO_DIAGRAM,
           "ACT 2 - HERO: BST + ResNet50 Visual Fusion + Contrastive",
           "arch_hero.png", fontsize=7)
    render(DISCOVERY_DIAGRAM,
           "ACT 3 - BRAIN: Multi-Objective Discovery Loss",
           "arch_discovery_loss.png", fontsize=9)
    print("Done.")
