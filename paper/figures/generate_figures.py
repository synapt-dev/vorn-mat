"""Generate Figures 1-5 for vorn-mat paper.

Outputs PDF figures next to this script for \\includegraphics in the LaTeX build.

Data refresh 2026-05-26: numbers verified against site/src/_data/vorn_mat_results.json
(playground schema vorn-mat-viz/v0.4) at site main 335c83c. Figure 5 (Memory
Pareto) added per Layne Option A ratification. Figure 1 extended to 7 families
(Qwen 3-NT restored 2026-05-25). Figure 2 extended to 5-budget Gemma 4 + H2O
lines. Figure 3 Mistral late-budget recovery filled. Figure 4 Gemma 2 + Qwen
2.5 H2O within-channel cells added per Option C ratification.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ===== Figure 1: Cross-family channel-favoritism axis (bar chart) =====
# Seven no-eviction-discriminative families with both sentence-vorn and sentence-TOVA at b=1024.
# Sorted by (sentence-vorn - sentence-TOVA) to show the spectrum: from
# Gemma 4 (attention-favoring extreme) to Qwen 2.5 (vorn-favoring extreme).
# Qwen 3-NT (no-thinking) added 2026-05-26 per Layne Option A ratification.

families = [
    ("Gemma 4 E4B-it", 0.24, 0.88),
    ("Qwen 3-NT 8B", 0.40, 1.00),
    ("Mistral 7B v0.3", 0.98, 1.00),
    ("Llama 3.1 8B", 1.00, 1.00),
    ("Ministral 8B", 1.00, 0.98),
    ("Gemma 2 9B", 0.86, 0.78),
    ("Qwen 2.5 7B", 0.76, 0.70),  # post-bf16-fix 2026-05-26 (was 0.96/0.08 pre-fix; see Appendix C.2)
]
# Sort by gap (sentence_vorn - sentence_tova) — left = attention-favoring, right = vorn-tolerant
families.sort(key=lambda f: f[1] - f[2])

labels = [f[0] for f in families]
vorn_vals = [f[1] for f in families]
tova_vals = [f[2] for f in families]

x = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5))
# synapt brand palette — colors matched to the active logo (brighter than the muted style-guide swatch)
SYNAPT_PURPLE = "#a855f7"  # logo outer magenta-purple
SYNAPT_BLUE_CHANNEL = "#22d3ee"  # logo inner cyan-turquoise
SYNAPT_TEAL = "#00c4ae"
SYNAPT_AMBER = "#f59e0b"
SYNAPT_BLUE = "#3b82f6"

bars_vorn = ax.bar(x - width/2, vorn_vals, width, label="sentence-vorn", color=SYNAPT_PURPLE, edgecolor="#3d2c5f", linewidth=0.5)
bars_tova = ax.bar(x + width/2, tova_vals, width, label="sentence-TOVA", color=SYNAPT_BLUE_CHANNEL, edgecolor="#1e40af", linewidth=0.5)

# Annotate the attention-favoring outliers (left side); post-bf16-fix the right side
# clusters as channel-tolerant majority rather than a vorn-favoring extreme.
ax.annotate(
    "attention-favoring\noutliers",
    xy=(0, 0.88), xytext=(0.5, 1.18),
    ha="center", fontsize=9, color=SYNAPT_BLUE_CHANNEL, weight="bold",
    arrowprops=dict(arrowstyle="->", color=SYNAPT_BLUE_CHANNEL, lw=1),
)
ax.annotate(
    "channel-tolerant majority\n(spread ≤ 0.08)",
    xy=(5, 0.95), xytext=(4.5, 1.18),
    ha="center", fontsize=9, color="#374151", weight="bold",
    arrowprops=dict(arrowstyle="->", color="#374151", lw=1),
)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Hit rate at b=1024", fontsize=10)
ax.set_ylim(0, 1.30)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.axhline(0, color="black", lw=0.5)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.40), fontsize=9, framealpha=0.95)
ax.set_title(
    "Cross-family channel-favoritism axis: sentence-vorn vs sentence-TOVA at b=1024",
    fontsize=11,
)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels above bars
for bars, vals in [(bars_vorn, vorn_vals), (bars_tova, tova_vals)]:
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

plt.tight_layout()
plt.savefig("figure1-cross-family-channel-axis.pdf", bbox_inches="tight")
plt.close()
print("Figure 1 saved: figure1-cross-family-channel-axis.pdf")


# ===== Figure 2: Gemma 4 degradation curve (line chart) — 5 budgets + 6 methods =====
# Extended to b=256 + b=2048 per Layne Option A ratification; H2O lines added.
# Source: playground vorn-mat-viz/v0.4 at site main 335c83c.
budgets = [256, 512, 1024, 1536, 2048]
token_vorn          = [0.00, 0.00, 0.02, 0.22, 0.56]
sentence_vorn       = [0.00, 0.04, 0.24, 0.68, 0.96]
tova_style          = [0.48, 0.34, 0.94, 0.98, 1.00]
sentence_tova_style = [1.00, 0.72, 0.88, 0.98, 1.00]
h2o_style           = [0.56, 0.48, 0.94, 0.98, 1.00]
sentence_h2o_style  = [0.98, 0.68, 0.86, 0.98, 1.00]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(budgets, token_vorn, marker="o", label="token-vorn", color=SYNAPT_PURPLE, linewidth=2, linestyle="--")
ax.plot(budgets, sentence_vorn, marker="s", label="sentence-vorn", color=SYNAPT_PURPLE, linewidth=2)
ax.plot(budgets, tova_style, marker="o", label="TOVA", color=SYNAPT_BLUE_CHANNEL, linewidth=2, linestyle="--")
ax.plot(budgets, sentence_tova_style, marker="s", label="sentence-TOVA", color=SYNAPT_BLUE_CHANNEL, linewidth=2)
ax.plot(budgets, h2o_style, marker="o", label="H2O", color="#0e7490", linewidth=2, linestyle="--")
ax.plot(budgets, sentence_h2o_style, marker="s", label="sentence-H2O", color="#0e7490", linewidth=2)

ax.set_xlabel("Cache budget B (tokens)", fontsize=10)
ax.set_ylabel("Hit rate on niah_multikey_1_4k", fontsize=10)
ax.set_xticks(budgets)
ax.set_xticklabels([str(b) for b in budgets])
ax.set_ylim(-0.05, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.set_title(
    "Gemma 4 E4B-it: budget-sweep degradation curve at 4k context",
    fontsize=11,
)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate the regime transition
ax.annotate(
    "constrained-budget regime:\nchannel-favoritism persists\n(attention-weight >> vorn)",
    xy=(1024, 0.5), xytext=(640, 0.55),
    fontsize=8, color="#8C1F1F",
    arrowprops=dict(arrowstyle="->", color="#8C1F1F", lw=0.7),
)
ax.annotate(
    "ceiling-regime budget:\nconvergence across\nscoring channels",
    xy=(2048, 0.98), xytext=(1500, 0.40),
    fontsize=8, color="#1F4E8C",
    arrowprops=dict(arrowstyle="->", color="#1F4E8C", lw=0.7),
)

plt.tight_layout()
plt.savefig("figure2-gemma4-budget-sweep.pdf", bbox_inches="tight")
plt.close()
print("Figure 2 saved: figure2-gemma4-budget-sweep.pdf")


# ===== Figure 3: Three-panel rescue spectrum (Mistral / Llama / Gemma 4) =====
# Vertical comparison of the three family-anchored sentence-attention 2x2 surfaces.
# Each panel: 4 lines (token-vorn, sentence-vorn, TOVA, sentence-TOVA) vs budget.
# Vorn lines in SYNAPT_PURPLE; TOVA lines in SYNAPT_BLUE_CHANNEL.
# Solid = sentence-level, dashed = token-level.
# Panel annotations: rescue-resilient / threshold-bounded / rescue-resistant.

# Full uniform-budget coverage at b={256, 512, 1024, 1536, 2048} on all three families.
# Mistral attention-weight cells at b=1536, 2048 are now COVERED via Atlas's 2026-05-26
# B=1536/B=2048 fill (site#116). The original H100 OOM was recovered via 3-case
# micro-recovery + 1-case recovery; rows carry completed_recovered status. No more Nones.
# Gemma 4 token-vorn at b=2048 updated 0.52 → 0.56 per current playground.
# Mistral sentence-vorn at b=256 updated 0.50 → 0.56 per current playground.
# Source: playground vorn-mat-viz/v0.4 at site main 335c83c.

mistral_budgets = [256, 512, 1024, 1536, 2048]
mistral_token_vorn     = [0.90, 0.96, 0.96, 0.96, 0.98]
mistral_sentence_vorn  = [0.56, 1.00, 0.98, 1.00, 1.00]
mistral_token_tova     = [0.04, 0.30, 0.86, 0.98, 0.98]
mistral_sentence_tova  = [0.74, 0.96, 1.00, 0.98, 1.00]

llama_budgets = [256, 512, 1024, 1536, 2048]
llama_token_vorn     = [0.54, 0.96, 1.00, 1.00, 1.00]
llama_sentence_vorn  = [1.00, 1.00, 1.00, 1.00, 1.00]
llama_token_tova     = [0.12, 0.56, 0.90, 0.96, 1.00]
llama_sentence_tova  = [0.74, 0.94, 1.00, 1.00, 1.00]

gemma4_budgets = [256, 512, 1024, 1536, 2048]
gemma4_token_vorn     = [0.00, 0.00, 0.02, 0.22, 0.56]
gemma4_sentence_vorn  = [0.00, 0.04, 0.24, 0.68, 0.96]
gemma4_token_tova     = [0.48, 0.34, 0.94, 0.98, 1.00]
gemma4_sentence_tova  = [1.00, 0.72, 0.88, 0.98, 1.00]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

panels = [
    (axes[0], "Mistral 7B v0.3", "rescue-resilient",
     mistral_budgets, mistral_token_vorn, mistral_sentence_vorn,
     mistral_token_tova, mistral_sentence_tova),
    (axes[1], "Llama 3.1 8B", "threshold-bounded recovery",
     llama_budgets, llama_token_vorn, llama_sentence_vorn,
     llama_token_tova, llama_sentence_tova),
    (axes[2], "Gemma 4 E4B-it", "attention-favoring across spectrum",
     gemma4_budgets, gemma4_token_vorn, gemma4_sentence_vorn,
     gemma4_token_tova, gemma4_sentence_tova),
]

def plot_with_gaps(ax, budgets, values, **kwargs):
    """Plot only where values are not None; treats None as runtime-unsupported."""
    xs = [b for b, v in zip(budgets, values) if v is not None]
    ys = [v for v in values if v is not None]
    if xs:
        ax.plot(xs, ys, **kwargs)
    # Mark unsupported cells with an X marker
    miss_xs = [b for b, v in zip(budgets, values) if v is None]
    if miss_xs:
        miss_color = kwargs.get("color", "#999999")
        ax.scatter(miss_xs, [0.04] * len(miss_xs),
                   marker="x", s=60, color=miss_color, linewidth=2.2, alpha=0.8)

for ax, family, profile, budgets, tv, sv, tt, st in panels:
    plot_with_gaps(ax, budgets, tv, marker="o", color=SYNAPT_PURPLE, linewidth=1.8,
                   linestyle="--", label="token-vorn", alpha=0.9)
    plot_with_gaps(ax, budgets, sv, marker="s", color=SYNAPT_PURPLE, linewidth=2.2,
                   linestyle="-", label="sentence-vorn")
    plot_with_gaps(ax, budgets, tt, marker="o", color=SYNAPT_BLUE_CHANNEL, linewidth=1.8,
                   linestyle="--", label="token-TOVA", alpha=0.9)
    plot_with_gaps(ax, budgets, st, marker="s", color=SYNAPT_BLUE_CHANNEL, linewidth=2.2,
                   linestyle="-", label="sentence-TOVA")

    ax.set_xlabel("Cache budget B (tokens)", fontsize=9.5)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{family}\n({profile})", fontsize=10.5)

axes[0].set_ylabel("Hit rate on niah_multikey_1_4k", fontsize=10)

# Shared legend at the bottom of the figure
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels_,
    loc="lower center", bbox_to_anchor=(0.5, -0.04),
    ncol=4, fontsize=9.5, framealpha=0.95,
)

# Per-panel annotation arrows. Each arrow lands on a specific data point,
# and the descriptor names the structural signature the point illustrates.

# Mistral: arrow lands on the sentence-TOVA point at b=512 (0.96 — the top of the rescue).
# The token-TOVA point at the same budget (0.30) is the floor; the rescue is the vertical
# distance between them. We label the *endpoint* the arrow targets.
axes[0].annotate(
    "token-TOVA 0.30 →\nsentence-TOVA 0.96\n(full rescue)",
    xy=(512, 0.96), xytext=(620, 0.55),
    fontsize=8.5, color="#0e7490",
    arrowprops=dict(arrowstyle="->", color="#0e7490", lw=0.9,
                    connectionstyle="arc3,rad=-0.15"),
)

# Llama: arrow lands on sentence-vorn at b=256 (1.00). The structural claim is
# "sentence-vorn still beats sentence-TOVA at the threshold lane" — the gap is
# 1.00 vs 0.74 at b=256.
axes[1].annotate(
    "sentence-vorn\nstill leads at b=256\n(1.00 vs sentence-TOVA 0.74)",
    xy=(256, 1.00), xytext=(330, 0.45),
    fontsize=8.5, color="#5b21b6",
    arrowprops=dict(arrowstyle="->", color="#5b21b6", lw=0.9,
                    connectionstyle="arc3,rad=-0.15"),
)

# Gemma 4: arrow lands on sentence-vorn at b=1024 (0.24). The descriptor names the
# structural failure — sentence-grouping did not rescue this point into the
# attention-weight band where sentence-TOVA sits (0.88) at the same budget.
axes[2].annotate(
    "sentence-vorn\nstays at 0.24\n(sentence-TOVA at 0.88)",
    xy=(1024, 0.24), xytext=(1120, 0.55),
    fontsize=8.5, color="#5b21b6",
    arrowprops=dict(arrowstyle="->", color="#5b21b6", lw=0.9,
                    connectionstyle="arc3,rad=0.15"),
)

fig.suptitle(
    "Three-family rescue spectrum: sentence-grouping rescues attention-weight channel on Mistral; "
    "rescue is threshold-bounded on Llama; Gemma 4 is attention-favoring across the spectrum",
    fontsize=10.5, y=1.02,
)

plt.tight_layout()
plt.savefig("figure3-rescue-spectrum.pdf", bbox_inches="tight")
plt.close()
print("Figure 3 saved: figure3-rescue-spectrum.pdf")


# ===== Figure 4: Cost vs Hit-Rate — paired bar comparison on Gemma 4 qa_2_4k at b=1024 =====
# Three completed cost-comparison rows from the Gemma 4 qa_2_4k slice (n=200 fixtures, b=1024).
# Two side-by-side panels: absolute cost (left) and absolute hit rate (right).
# Same method ordering + color coding across both panels so the trade-off is read directly
# off bar-length contrasts: vorn is much shorter in the cost panel and only slightly shorter
# in the hit-rate panel.

N_FIXTURES = 200

methods = [
    # (name, batch_cost_usd, hit_rate, color)
    ("sentence-vorn", 0.4620, 0.260, SYNAPT_PURPLE),
    ("TOVA",    0.8995, 0.310, SYNAPT_BLUE_CHANNEL),
    ("H2O",     1.1577, 0.305, "#0e7490"),
]

names   = [m[0] for m in methods]
costs   = [m[1] for m in methods]
rates   = [m[2] for m in methods]
colors  = [m[3] for m in methods]
rates_pct = [r * 100 for r in rates]
correct_per_dollar = [(r * N_FIXTURES) / c for r, c in zip(rates, costs)]
cost_per_correct_cents = [100.0 * c / (r * N_FIXTURES) for r, c in zip(rates, costs)]
y_pos   = np.arange(len(methods))

# ===== MULTI-FAMILY DATA (qa_2_4k cross-task, n=200, b=1024) =====
# Source: prototypes/vorn-mat/results/qa2-cross-task-multifamily-2026-05-16.json
# Rows: (family, method, correct, cost_usd)
MULTIFAM = [
    # Gemma 4
    ("Gemma 4 E4B-it",  "no-eviction",   90, 0.1771),
    ("Gemma 4 E4B-it",  "sentence-vorn", 52, 0.4620),
    ("Gemma 4 E4B-it",  "TOVA",    62, 0.8995),
    ("Gemma 4 E4B-it",  "H2O",     61, 1.1577),
    # Llama 3.1
    ("Llama 3.1 8B",    "no-eviction",   77, 0.1837),
    ("Llama 3.1 8B",    "sentence-vorn", 92, 0.2976),
    ("Llama 3.1 8B",    "TOVA",    66, 0.3954),
    ("Llama 3.1 8B",    "H2O",     66, 0.3738),
    # Ministral 8B
    ("Ministral 8B",    "no-eviction",  115, 0.1959),
    ("Ministral 8B",    "sentence-vorn",102, 0.3424),
]
METHOD_COLORS = {
    "no-eviction":   "#9ca3af",   # neutral gray — context baseline
    "sentence-vorn": SYNAPT_PURPLE,
    "TOVA":    SYNAPT_BLUE_CHANNEL,
    "H2O":     "#0e7490",
}
FAMILIES = ["Gemma 4 E4B-it", "Llama 3.1 8B", "Ministral 8B"]
METHODS  = ["no-eviction", "sentence-vorn", "TOVA", "H2O"]
N_TOTAL = 200

def _get(family, method):
    """Return (correct, cost) for a (family, method) pair, or None if missing."""
    for r in MULTIFAM:
        if r[0] == family and r[1] == method:
            return r[2], r[3]
    return None


# =============================================================================
# Variant A — Family-grouped raw bars
# For each family, two bar groups side by side: correct-out-of-200 (left) +
# cost (right). Anchored absolute numbers, no derived metrics in primary view.
# Tells the multi-family story directly: vorn cheapest with competitive quality
# on Llama and Ministral; only Gemma 4 attention-weight methods buy more correct
# answers, and pay roughly double for it.
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharey="row")

for col, fam in enumerate(FAMILIES):
    ax_q = axes[0, col]
    ax_c = axes[1, col]
    methods_present = [m for m in METHODS if _get(fam, m) is not None]
    colors_p = [METHOD_COLORS[m] for m in methods_present]
    correct  = [_get(fam, m)[0] for m in methods_present]
    cost_p   = [_get(fam, m)[1] for m in methods_present]
    x = np.arange(len(methods_present))

    # Top row: correct out of 200
    bars_q = ax_q.bar(x, correct, color=colors_p, edgecolor="black", linewidth=0.7)
    ax_q.set_ylim(0, 200)
    ax_q.set_yticks([0, 50, 100, 150, 200])
    ax_q.axhline(200, color="#374151", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_q.set_title(fam, fontsize=11, weight="bold")
    if col == 0:
        ax_q.set_ylabel("Correct out of 200", fontsize=10)
    ax_q.spines["top"].set_visible(False)
    ax_q.spines["right"].set_visible(False)
    ax_q.set_xticks(x)
    ax_q.set_xticklabels([])
    for bar, c in zip(bars_q, correct):
        ax_q.text(bar.get_x() + bar.get_width() / 2, c + 4,
                  f"{c}", ha="center", fontsize=10, weight="bold")

    # Bottom row: cost
    bars_c = ax_c.bar(x, cost_p, color=colors_p, edgecolor="black", linewidth=0.7,
                       alpha=0.55, hatch="//")
    ax_c.set_ylim(0, 1.35)
    ax_c.set_yticks(np.arange(0, 1.31, 0.25))
    if col == 0:
        ax_c.set_ylabel("Cost per 200-fixture\nbatch (USD)", fontsize=10)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(methods_present, rotation=20, ha="right", fontsize=9)
    for bar, c in zip(bars_c, cost_p):
        ax_c.text(bar.get_x() + bar.get_width() / 2, c + 0.03,
                  f"\\${c:.2f}", ha="center", fontsize=9.5, weight="bold")

fig.suptitle(
    "Variant A — Anchored bars: raw correct-out-of-200 on top, raw cost on bottom, by family",
    fontsize=10.5, y=1.02,
)
plt.tight_layout()
plt.savefig("figure4-variant-a-family-grouped-raw.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 Variant A saved: figure4-variant-a-family-grouped-raw.pdf")


# =============================================================================
# Variant B — Cost vs correct scatter (multi-family)
# X = cost ($), Y = correct out of 200. One scatter, all 10 points, colored by
# method, shaped by family. Reference line at "perfect = 200". Reader sees
# the trade-off geometrically across families.
# =============================================================================
FAMILY_MARKERS = {"Gemma 4 E4B-it": "o", "Llama 3.1 8B": "s", "Ministral 8B": "D"}

fig, ax = plt.subplots(figsize=(11, 6))
for fam, met, correct, cost in MULTIFAM:
    ax.scatter([cost], [correct],
               s=240, color=METHOD_COLORS[met],
               marker=FAMILY_MARKERS[fam],
               edgecolor="black", linewidth=1.2, zorder=5)
    # Inline label "fam · method" near each point
    label = f"{fam.split()[0]} · {met}"
    ax.annotate(f"  {met}\n  {correct}/200 correct\n  \\${cost:.2f}",
                xy=(cost, correct), xytext=(8, 6),
                textcoords="offset points",
                fontsize=8.0, color=METHOD_COLORS[met], weight="bold",
                alpha=0.95)

ax.axhline(200, color="#374151", linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(1.45, 200.5, "perfect = 200 correct", fontsize=8.5,
        color="#374151", alpha=0.7, ha="right")

ax.set_xlabel("Cost per 200-fixture batch (USD)", fontsize=10)
ax.set_ylabel("Correct answers (out of 200)", fontsize=10)
ax.set_xlim(0.10, 1.45)
ax.set_ylim(40, 215)
ax.set_xticks(np.arange(0.2, 1.41, 0.2))
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Family legend (shape)
from matplotlib.lines import Line2D
fam_handles = [Line2D([0], [0], marker=FAMILY_MARKERS[f], color="w",
                       markerfacecolor="#9ca3af", markeredgecolor="black",
                       markersize=11, label=f) for f in FAMILIES]
# Method legend (color)
met_handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=METHOD_COLORS[m], markeredgecolor="black",
                       markersize=11, label=m) for m in METHODS]
leg1 = ax.legend(handles=fam_handles, loc="upper left", title="Family (shape)",
                 fontsize=8.5, title_fontsize=9, framealpha=0.95)
ax.add_artist(leg1)
ax.legend(handles=met_handles, loc="lower right", title="Method (color)",
          fontsize=8.5, title_fontsize=9, framealpha=0.95)

ax.set_title(
    "Variant B — Cost vs correct scatter: every (family, method) on one chart",
    fontsize=10.5,
)
plt.tight_layout()
plt.savefig("figure4-variant-b-multifamily-scatter.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 Variant B saved: figure4-variant-b-multifamily-scatter.pdf")


# =============================================================================
# Variant C — Per-family marginal-cost waterfall
# For each family, start at no-eviction (baseline), then each compressed method
# is a step showing "+$X buys +/- Y correct vs the baseline." Visualizes what
# the dollars actually purchase relative to the natural baseline.
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

for col, fam in enumerate(FAMILIES):
    ax = axes[col]
    methods_present = [m for m in METHODS if _get(fam, m) is not None]
    base_correct, base_cost = _get(fam, "no-eviction")

    # Bar 1: no-eviction baseline absolute correct
    x_positions = np.arange(len(methods_present))
    base_bar_h = base_correct
    ax.bar([0], [base_bar_h], color=METHOD_COLORS["no-eviction"],
           edgecolor="black", linewidth=0.7, label="absolute correct")
    ax.text(0, base_bar_h + 3, f"{base_correct}/200\n\\${base_cost:.2f}",
            ha="center", fontsize=9, weight="bold")

    # Subsequent bars: each compressed method shown as delta-from-baseline
    for i, met in enumerate(methods_present[1:], start=1):
        cc, cost = _get(fam, met)
        delta_correct = cc - base_correct
        delta_cost = cost - base_cost
        # Draw bar at absolute height + connecting segment from base
        ax.bar([i], [cc], color=METHOD_COLORS[met],
               edgecolor="black", linewidth=0.7, alpha=0.95)
        # Connecting horizontal line from baseline top to this bar top
        ax.plot([0.4, i - 0.4], [base_bar_h, base_bar_h],
                color="#9ca3af", linestyle=":", linewidth=0.6)
        # Delta arrow
        arrow_color = "#15803d" if delta_correct >= 0 else "#b91c1c"
        ax.annotate("",
                    xy=(i, cc), xytext=(i, base_bar_h),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.3))
        sign = "+" if delta_correct >= 0 else ""
        ax.text(i, max(cc, base_bar_h) + 6,
                f"{sign}{delta_correct} correct\n+\\${delta_cost:.2f}",
                ha="center", fontsize=8.8, color=arrow_color, weight="bold")
        # Absolute label below
        ax.text(i, cc / 2, f"{cc}/200\n\\${cost:.2f}",
                ha="center", va="center", fontsize=8.5,
                color="white", weight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods_present, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 200)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_title(fam, fontsize=11, weight="bold")
    if col == 0:
        ax.set_ylabel("Correct out of 200", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Variant C — Marginal-cost: \"what does spending more vs no-eviction buy you?\" by family",
    fontsize=10.5, y=1.03,
)
plt.tight_layout()
plt.savefig("figure4-variant-c-marginal-waterfall.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 Variant C saved: figure4-variant-c-marginal-waterfall.pdf")


# =============================================================================
# Variant D — Correct-per-dollar leaderboard (multi-family)
# One horizontal bar per (family, method), bar length = correct answers per $1
# spent. Sorted from most efficient to least. Raw correct + raw cost shown to
# the right of each bar so the derivation is visible. Single ranked view.
# =============================================================================
rows = []
for fam, met, correct, cost in MULTIFAM:
    cpd = correct / cost
    rows.append((fam, met, correct, cost, cpd))
rows.sort(key=lambda r: -r[4])  # descending efficiency

fig, ax = plt.subplots(figsize=(12.5, 6))
labels = [f"{r[0].split()[0]} · {r[1]}" for r in rows]
values = [r[4] for r in rows]
colors_d = [METHOD_COLORS[r[1]] for r in rows]
edge_per_family = {"Gemma 4 E4B-it": "#374151", "Llama 3.1 8B": "#1e3a8a", "Ministral 8B": "#581c87"}
edges = [edge_per_family[r[0]] for r in rows]

y_pos = np.arange(len(rows))
bars = ax.barh(y_pos, values, color=colors_d, edgecolor=edges, linewidth=1.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Correct answers per \\$1 spent  (higher is better)", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)

for bar, (fam, met, correct, cost, cpd) in zip(bars, rows):
    detail = f"  {cpd:.0f}   ({correct}/200 correct, \\${cost:.2f} cost)"
    ax.text(cpd, bar.get_y() + bar.get_height() / 2, detail,
            va="center", fontsize=9.5, weight="bold", color="#1f2937")

ax.set_xlim(0, max(values) * 1.45)
ax.set_title(
    "Variant D — Correct-per-\\$1 leaderboard (rank-ordered); raw correct \\& cost shown to the right",
    fontsize=10.5,
)
plt.tight_layout()
plt.savefig("figure4-variant-d-leaderboard.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 Variant D saved: figure4-variant-d-leaderboard.pdf")


# =============================================================================
# B-two-task and D-two-task: re-build the two layouts Layne preferred (B scatter,
# D leaderboard) with TWO panels each — primary task niah_multikey_1_4k vs
# cross-task validation qa_2_4k. Both panels at B=1024. Budget + task spelled out.
# =============================================================================

NIAH_B1024 = [
    ("Mistral 7B v0.3", "sentence-vorn",       49, 0.194),
    ("Mistral 7B v0.3", "sentence-TOVA", 50, 0.204),
    ("Mistral 7B v0.3", "sentence-H2O",  50, 0.165),
    ("Llama 3.1 8B",    "sentence-vorn",       50, 0.195),
    ("Llama 3.1 8B",    "sentence-TOVA", 50, 0.157),
    ("Llama 3.1 8B",    "sentence-H2O",  50, 0.183),
    ("Gemma 4 E4B-it",  "sentence-vorn",       12, 0.218),
    ("Gemma 4 E4B-it",  "sentence-TOVA", 44, 0.204),
    ("Gemma 4 E4B-it",  "sentence-H2O",  43, 0.224),
]
NIAH_N = 50

QA2_B1024 = [
    ("Gemma 4 E4B-it", "sentence-vorn", 52,  0.4620),
    ("Gemma 4 E4B-it", "TOVA",    62,  0.8995),
    ("Gemma 4 E4B-it", "H2O",     61,  1.1577),
    ("Llama 3.1 8B",   "sentence-vorn", 92,  0.2976),
    ("Llama 3.1 8B",   "TOVA",    66,  0.3954),
    ("Llama 3.1 8B",   "H2O",     66,  0.3738),
    ("Ministral 8B",   "sentence-vorn",102,  0.3424),
]
QA2_N = 200

TT_METHOD_COLORS = {
    "sentence-vorn":       SYNAPT_PURPLE,
    "sentence-TOVA": SYNAPT_BLUE_CHANNEL,
    "sentence-H2O":  "#0e7490",
    "TOVA":          SYNAPT_BLUE_CHANNEL,
    "H2O":           "#0e7490",
}
TT_FAM_MARKERS = {
    "Mistral 7B v0.3": "^",
    "Llama 3.1 8B":    "s",
    "Gemma 4 E4B-it":  "o",
    "Ministral 8B":    "D",
}
FAMILY_EDGE = {"Gemma 4 E4B-it": "#374151", "Llama 3.1 8B": "#1e3a8a",
               "Ministral 8B": "#581c87", "Mistral 7B v0.3": "#7c2d12"}

# ----- Variant B-two-task: two scatter panels side by side -----
def plot_scatter_panel(ax, data, n_total, title, xmax):
    for fam, met, correct, cost in data:
        ax.scatter([cost], [correct],
                   s=200, color=TT_METHOD_COLORS[met],
                   marker=TT_FAM_MARKERS[fam],
                   edgecolor="black", linewidth=1.0, zorder=5)
        ax.annotate(f"  {fam.split()[0]}\n  {met}\n  {correct}/{n_total} \\${cost:.2f}",
                    xy=(cost, correct), xytext=(7, 5),
                    textcoords="offset points",
                    fontsize=7.5, color=TT_METHOD_COLORS[met], weight="bold")
    ax.axhline(n_total, color="#374151", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(xmax * 0.97, n_total + n_total * 0.025,
            f"perfect = {n_total}", fontsize=8.5, color="#374151",
            ha="right", alpha=0.7)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, n_total * 1.15)
    ax.set_xlabel("Cost per batch (USD)", fontsize=10)
    ax.set_ylabel(f"Correct out of {n_total}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=11, weight="bold")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
plot_scatter_panel(
    axes[0], NIAH_B1024, NIAH_N,
    "Primary task: niah_multikey_1_4k  (B=1024 tokens, n=50)", xmax=0.30,
)
plot_scatter_panel(
    axes[1], QA2_B1024, QA2_N,
    "Cross-task validation: qa_2_4k  (B=1024 tokens, n=200)", xmax=1.35,
)

from matplotlib.lines import Line2D
fam_handles = [Line2D([0], [0], marker=TT_FAM_MARKERS[f], color="w",
                       markerfacecolor="#9ca3af", markeredgecolor="black",
                       markersize=10, label=f)
               for f in ["Mistral 7B v0.3", "Llama 3.1 8B", "Gemma 4 E4B-it", "Ministral 8B"]]
met_keys = ["sentence-vorn", "sentence-TOVA", "sentence-H2O",
            "TOVA", "H2O"]
met_handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=TT_METHOD_COLORS[m],
                       markeredgecolor="black", markersize=10, label=m)
               for m in met_keys]
fig.legend(handles=fam_handles, loc="lower left", bbox_to_anchor=(0.02, -0.05),
           title="Family (shape)", ncol=4, fontsize=8.5, title_fontsize=9,
           framealpha=0.95)
fig.legend(handles=met_handles, loc="lower right", bbox_to_anchor=(0.98, -0.05),
           title="Method (color)", ncol=5, fontsize=8.5, title_fontsize=9,
           framealpha=0.95)

fig.suptitle(
    "Variant B (two-task) — Cost vs correct, same budget B=1024 tokens, both tasks.  "
    "Left: primary task (saturates near perfect).  Right: cross-task validation (harder, no saturation).",
    fontsize=10.5, y=1.02,
)
plt.tight_layout()
plt.savefig("figure4-B-two-task.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 B-two-task saved: figure4-B-two-task.pdf")


# ----- Variant D-two-task: two leaderboards side by side -----
def plot_leaderboard_panel(ax, data, n_total, title):
    rows = [(f, m, c, cost, c / cost) for (f, m, c, cost) in data]
    rows.sort(key=lambda r: -r[4])
    labels = [f"{r[0].split()[0]} · {r[1]}" for r in rows]
    values = [r[4] for r in rows]
    colors_d = [TT_METHOD_COLORS[r[1]] for r in rows]
    edges = [FAMILY_EDGE[r[0]] for r in rows]

    y_pos = np.arange(len(rows))
    bars = ax.barh(y_pos, values, color=colors_d, edgecolor=edges, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Correct answers per \\$1 spent  (n={n_total})", fontsize=10)
    ax.set_xlim(0, max(values) * 1.55)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, (fam, met, correct, cost, cpd) in zip(bars, rows):
        ax.text(cpd, bar.get_y() + bar.get_height() / 2,
                f"  {cpd:.0f}   ({correct}/{n_total} correct, \\${cost:.2f} cost)",
                va="center", fontsize=8.4, weight="bold", color="#1f2937")
    ax.set_title(title, fontsize=11, weight="bold")

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
plot_leaderboard_panel(
    axes[0], NIAH_B1024, NIAH_N,
    "Primary task: niah_multikey_1_4k  (B=1024, n=50)",
)
plot_leaderboard_panel(
    axes[1], QA2_B1024, QA2_N,
    "Cross-task validation: qa_2_4k  (B=1024, n=200)",
)
fig.suptitle(
    "Variant D (two-task) — Correct-per-\\$1 leaderboards, same budget B=1024 tokens, both tasks.",
    fontsize=10.5, y=1.00,
)
plt.tight_layout()
plt.savefig("figure4-D-two-task.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 D-two-task saved: figure4-D-two-task.pdf")


# =============================================================================
# Figure 4 (final) — within-channel sentence-vs-token cost-per-correct.
# Headline: granularity (sentence vs token) is the cost-dominant variable;
# scoring channel (vorn vs TOVA/H2O) is not. Each (family, channel) cell shows
# paired bars for token-level and sentence-level correct-per-$1; sentence wins
# in every cell across both tasks.
# =============================================================================

# niah_multikey_1_4k at b=1024, n=50
# Sentinel review 2026-05-20: excluded the Gemma 4 vorn cell (floor case — 1/50
# token-vorn correct, ratio unstable near zero) and the Gemma 4 token-H2O cell
# (suspected cost-recording anomaly — $2.18 vs ~$0.30 on adjacent cells). Both
# are reported as outliers in Appendix A footnote, not in the within-channel
# ratio claim. Seven non-floor non-anomalous cells remain.
NIAH_CELLS = [
    # (family, channel, token_correct, token_cost, sentence_correct, sentence_cost)
    ("Mistral 7B v0.3", "vorn",  48, 0.300, 49, 0.194),
    ("Mistral 7B v0.3", "TOVA",  43, 0.292, 50, 0.204),
    ("Mistral 7B v0.3", "H2O",   42, 0.291, 50, 0.165),
    ("Llama 3.1 8B",    "vorn",  50, 0.249, 50, 0.195),
    ("Llama 3.1 8B",    "TOVA",  45, 0.263, 50, 0.157),
    ("Llama 3.1 8B",    "H2O",   47, 0.216, 50, 0.183),
    ("Gemma 4 E4B-it",  "TOVA",  47, 0.315, 44, 0.204),
    # Gemma 2 + Qwen 2.5 H2O cells added 2026-05-26 per Layne Option C ratification.
    # Both extend the within-channel sentence-vs-token finding to two additional
    # families. Numbers verified against playground vorn-mat-viz/v0.4 at site 335c83c.
    ("Gemma 2 9B",      "H2O",   30, 0.5166, 41, 0.2592),
    ("Qwen 2.5 7B",     "H2O",   30, 0.3169, 35, 0.1989),
    # Excluded outliers (reported in Appendix A footnote):
    #   Gemma 4 vorn  — token-vorn floor 1/50, ratio unstable near zero
    #   Gemma 4 H2O   — token-H2O cost $2.18 likely Modal cold-start anomaly
]
# qa_2_4k at b=1024, n=200 (cross-task)
QA2_CELLS = [
    # Llama 3.1 vorn cross-task: 52 → 92, ratio 4.03× (the strongest non-floor
    # sentence advantage in the figure; reported as the cross-task vorn anchor).
    # Ratio recomputed precisely: (92/0.2976) / (52/0.6778) = 309.14 / 76.72 = 4.03×.
    # Excluded from chart (floor case, footnoted): Gemma 4 vorn cross-task
    # token-vorn 10/200 @ $1.32 vs sentence-vorn 52/200 @ $0.46. Ratio 14.1×
    # unstable near zero, parallel to the niah Gemma 4 vorn floor exclusion.
    ("Llama 3.1 8B",    "vorn",  52, 0.6778, 92, 0.2976),
    ("Gemma 4 E4B-it",  "TOVA",  62, 0.8995, 74, 0.5375),
    ("Gemma 4 E4B-it",  "H2O",   61, 1.1577, 74, 0.5997),
    ("Llama 3.1 8B",    "TOVA",  66, 0.3954, 85, 0.2789),
    ("Llama 3.1 8B",    "H2O",   66, 0.3738, 85, 0.2732),
]

CHANNEL_COLORS = {
    "vorn": SYNAPT_PURPLE,
    "TOVA": SYNAPT_BLUE_CHANNEL,
    "H2O":  "#0e7490",
}

def cpd(correct, cost):
    return correct / cost

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5),
                          gridspec_kw={"width_ratios": [9, 4.5]})

# ----- Left panel: niah primary task -----
ax = axes[0]
y_positions = []
y_labels = []
# Build all niah rows across all families present in NIAH_CELLS (5 families post Option C)
y = 0
gap = 0.5
bar_w = 0.38
for fam in ["Mistral 7B v0.3", "Llama 3.1 8B", "Gemma 4 E4B-it", "Gemma 2 9B", "Qwen 2.5 7B"]:
    fam_rows = [r for r in NIAH_CELLS if r[0] == fam]
    fam_start_y = y
    for (f, ch, tc, tcost, sc, scost) in fam_rows:
        t_cpd = cpd(tc, tcost)
        s_cpd = cpd(sc, scost)
        ax.barh([y - bar_w/2], [t_cpd], bar_w,
                color=CHANNEL_COLORS[ch], alpha=0.4, hatch="///",
                edgecolor="black", linewidth=0.5,
                label="token-level" if (fam == "Mistral 7B v0.3" and ch == "vorn") else None)
        ax.barh([y + bar_w/2], [s_cpd], bar_w,
                color=CHANNEL_COLORS[ch], alpha=0.95,
                edgecolor="black", linewidth=0.5,
                label="sentence-level" if (fam == "Mistral 7B v0.3" and ch == "vorn") else None)
        # Annotate values + raw numbers
        ax.text(t_cpd + 5, y - bar_w/2,
                f"  token: {tc}/50, \\${tcost:.2f} → {t_cpd:.0f}",
                va="center", fontsize=7.5, color="#374151")
        ax.text(s_cpd + 5, y + bar_w/2,
                f"  sentence: {sc}/50, \\${scost:.2f} → {s_cpd:.0f}",
                va="center", fontsize=7.5, weight="bold", color="#1f2937")
        y_positions.append(y)
        y_labels.append(f"{ch} · {fam.split()[0]}")
        y += 1
    y += gap

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Correct answers per \\$1 spent  (higher = more efficient)", fontsize=10)
ax.set_xlim(0, 450)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Primary task: niah_multikey_1_4k  (B=1024, n=50)",
             fontsize=11, weight="bold")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

# ----- Right panel: qa_2 cross-task -----
ax = axes[1]
y_positions = []
y_labels = []
y = 0
for fam in ["Gemma 4 E4B-it", "Llama 3.1 8B"]:
    fam_rows = [r for r in QA2_CELLS if r[0] == fam]
    for (f, ch, tc, tcost, sc, scost) in fam_rows:
        t_cpd = cpd(tc, tcost)
        s_cpd = cpd(sc, scost)
        ax.barh([y - bar_w/2], [t_cpd], bar_w,
                color=CHANNEL_COLORS[ch], alpha=0.4, hatch="///",
                edgecolor="black", linewidth=0.5)
        ax.barh([y + bar_w/2], [s_cpd], bar_w,
                color=CHANNEL_COLORS[ch], alpha=0.95,
                edgecolor="black", linewidth=0.5)
        ax.text(t_cpd + 4, y - bar_w/2,
                f"  {tc}/200, \\${tcost:.2f} → {t_cpd:.0f}",
                va="center", fontsize=7.5, color="#374151")
        ax.text(s_cpd + 4, y + bar_w/2,
                f"  {sc}/200, \\${scost:.2f} → {s_cpd:.0f}",
                va="center", fontsize=7.5, weight="bold", color="#1f2937")
        y_positions.append(y)
        y_labels.append(f"{ch} · {fam.split()[0]}")
        y += 1
    y += gap

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Correct answers per \\$1 spent", fontsize=10)
ax.set_xlim(0, 500)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Cross-task validation: qa_2_4k  (B=1024, n=200)",
             fontsize=11, weight="bold")

# In-chart floor-exclusion annotation for Gemma 4 vorn qa_2 (token 10/200 floor).
# Per Layne 2026-05-26: the reader's first question on this panel is "why no
# vorn row for Gemma?" Answer it visibly without adding a misleading 14.9× bar.
ax.text(250, -1.0,
        "Excluded from cost-ratio claim: Gemma 4 vorn (token 10/200 floor; sentence 52/200; see §A.7)",
        fontsize=8, color="#7c2d12", style="italic",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7",
                  edgecolor="#92400e", linewidth=0.8, alpha=0.85))

fig.suptitle(
    "Figure 4 — Within-channel sentence-vs-token cost-per-correct on non-floor non-anomalous cells.\n"
    "Sentence-grouping is cheaper per correct within every plotted cell: ~1.25-2.72× on niah primary, ~1.76-4.03× on qa_2 cross-task.\n"
    "Excluded: Gemma 4 vorn niah (token floor 1/50, ratio unstable); Gemma 4 token-H2O niah (cost anomaly \\$2.18, suspected cold-start);\n"
    "Gemma 4 vorn qa_2 (token floor 10/200, parallel to niah floor exclusion).",
    fontsize=9.5, y=1.06,
)
plt.tight_layout()
plt.savefig("figure4-cost-quality-scatter.pdf", bbox_inches="tight")
plt.close()
print("Figure 4 (final, within-channel layout) saved: figure4-cost-quality-scatter.pdf")


# =============================================================================
# Figure 5 — Memory Pareto frontier (scatter)
# Layne Option A ratification 2026-05-26: visualize the vorn-vs-attention-weight
# memory split that §5.10 quantifies. Vorn cluster ~21GB; attention-weight
# cluster ~52GB. Same hit-rate ceiling, dramatically different memory cost.
# Source: playground vorn-mat-viz/v0.4 at site main 335c83c. 81 cells with
# peak_memory_allocated_mb + hit_rate (memory rerun coverage). Method-class
# clusters from playground summary:
#   vorn (token + sentence):  n=27, median 20.8 GB, range 19.5-22.1 GB
#   attention-weight (TOVA + H2O, token + sentence): n=53, median 52.0 GB,
#                                                    range 29.2-58.0 GB
# =============================================================================

# Data points: (family, method_class, peak_memory_GB, hit_rate)
# Values represent the median of each (family, method) cell across budgets in
# the memory-rerun coverage set. Aggregated from playground for readability;
# raw per-cell scatter would overplot at family/method clusters since memory
# is largely budget-invariant for the methods that emitted telemetry.
F5_DATA = [
    # vorn cluster (~20-22 GB), all families with vorn memory data
    ("Mistral 7B v0.3",       "token-vorn",    20.0, 0.96),
    ("Mistral 7B v0.3",       "sentence-vorn", 20.0, 1.00),
    ("Llama 3.1 8B",          "token-vorn",    20.5, 1.00),
    ("Llama 3.1 8B",          "sentence-vorn", 20.5, 1.00),
    ("Ministral 8B",          "token-vorn",    20.8, 0.98),
    ("Ministral 8B",          "sentence-vorn", 20.8, 1.00),
    ("Gemma 2 9B",            "token-vorn",    22.1, 0.50),
    ("Gemma 2 9B",            "sentence-vorn", 22.1, 0.86),
    ("Gemma 4 E4B-it",        "token-vorn",    19.5, 0.02),
    ("Qwen 3-NT 8B",          "token-vorn",    20.9, 0.04),
    ("Qwen 3-NT 8B",          "sentence-vorn", 20.9, 0.40),
    # attention-weight cluster (~29-58 GB)
    ("Mistral 7B v0.3",       "token-TOVA",    57.7, 0.86),
    ("Mistral 7B v0.3",       "sentence-TOVA", 57.7, 1.00),
    ("Mistral 7B v0.3",       "token-H2O",     58.0, 0.84),
    ("Mistral 7B v0.3",       "sentence-H2O",  57.9, 1.00),
    ("Llama 3.1 8B",          "token-H2O",     50.7, 0.94),
    ("Llama 3.1 8B",          "sentence-H2O",  50.7, 1.00),
    ("Ministral 8B",          "token-TOVA",    52.5, 0.96),
    ("Ministral 8B",          "token-H2O",     52.5, 0.94),
    ("Ministral 8B",          "sentence-H2O",  52.5, 1.00),
    ("Gemma 2 9B",            "token-TOVA",    41.6, 0.66),
    ("Gemma 2 9B",            "token-H2O",     41.6, 0.60),
    ("Gemma 2 9B",            "sentence-H2O",  41.6, 0.82),
    ("Gemma 4 E4B-it",        "token-H2O",     29.2, 0.94),
    ("Gemma 4 E4B-it",        "sentence-H2O",  29.2, 0.86),
    ("Qwen 2.5 7B Instruct",  "token-H2O",     40.2, 0.60),
    ("Qwen 2.5 7B Instruct",  "sentence-H2O",  40.2, 0.70),
    ("Qwen 3-NT 8B",          "token-TOVA",    52.5, 1.00),
    ("Qwen 3-NT 8B",          "sentence-TOVA", 52.5, 1.00),
    ("Qwen 3-NT 8B",          "token-H2O",     52.5, 0.96),
    ("Qwen 3-NT 8B",          "sentence-H2O",  52.5, 1.00),
]

# Color by family, shape by method-class (vorn vs attention-weight)
F5_FAMILY_COLORS = {
    "Mistral 7B v0.3":       "#7c2d12",   # brown
    "Llama 3.1 8B":          "#1e3a8a",   # navy
    "Ministral 8B":          "#581c87",   # deep purple
    "Gemma 2 9B":            "#15803d",   # green
    "Gemma 4 E4B-it":        "#0f766e",   # teal
    "Qwen 2.5 7B Instruct":  "#9a3412",   # rust
    "Qwen 3-NT 8B":          "#a16207",   # gold
}
F5_METHOD_MARKERS = {
    # vorn methods: circle / square (filled, no hatch)
    "token-vorn":    "o",
    "sentence-vorn": "s",
    # attention-weight methods: triangle / diamond / inverted-triangle / pentagon
    "token-TOVA":    "^",
    "sentence-TOVA": "v",
    "token-H2O":     "D",
    "sentence-H2O":  "P",
}
F5_VORN_METHODS = {"token-vorn", "sentence-vorn"}

fig, ax = plt.subplots(figsize=(11, 6.5))

# Plot points
for fam, met, mem_gb, hr in F5_DATA:
    color = F5_FAMILY_COLORS[fam]
    marker = F5_METHOD_MARKERS[met]
    # Highlight vorn cluster with extra edge weight; attention-weight cluster lighter edge
    edge_lw = 1.6 if met in F5_VORN_METHODS else 0.9
    edge_color = "black" if met in F5_VORN_METHODS else "#374151"
    ax.scatter([mem_gb], [hr], s=170, marker=marker,
               color=color, edgecolor=edge_color, linewidth=edge_lw,
               zorder=5, alpha=0.92)

# Shaded cluster bands
ax.axvspan(19, 23, color=SYNAPT_PURPLE, alpha=0.08, zorder=1)
ax.axvspan(28, 60, color=SYNAPT_BLUE_CHANNEL, alpha=0.06, zorder=1)

# Cluster labels (positioned outside the data region)
ax.text(21, 0.07, "vorn cluster\n~20-22 GB",
        ha="center", fontsize=10, weight="bold", color=SYNAPT_PURPLE, alpha=0.85)
ax.text(44, 0.07, "attention-weight cluster\n~29-58 GB",
        ha="center", fontsize=10, weight="bold", color=SYNAPT_BLUE_CHANNEL, alpha=0.85)

# Pareto-frontier arrow (vorn → attention-weight memory cost)
ax.annotate(
    "Pareto-frontier delta: ~30 GB additional\nmemory for similar/lower quality on most cells",
    xy=(35, 0.92), xytext=(28, 0.50),
    fontsize=9, color="#374151", weight="bold",
    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2,
                    connectionstyle="arc3,rad=0.2"),
)

ax.set_xlim(15, 62)
ax.set_ylim(-0.02, 1.10)
ax.set_xticks(np.arange(20, 61, 5))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("Peak GPU memory allocated (GB) — A100 80GB profile", fontsize=10.5)
ax.set_ylabel("Hit rate on niah_multikey_1_4k", fontsize=10.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Family legend (color)
from matplotlib.lines import Line2D
fam_handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=F5_FAMILY_COLORS[f],
                       markeredgecolor="black", markersize=10, label=f)
               for f in F5_FAMILY_COLORS]
# Method legend (shape)
met_handles = [Line2D([0], [0], marker=F5_METHOD_MARKERS[m], color="w",
                       markerfacecolor="#9ca3af",
                       markeredgecolor="black", markersize=11, label=m)
               for m in ["token-vorn", "sentence-vorn", "token-TOVA",
                         "sentence-TOVA", "token-H2O", "sentence-H2O"]]
leg1 = ax.legend(handles=fam_handles, loc="lower right",
                 title="Family (color)",
                 fontsize=8, title_fontsize=9, framealpha=0.95,
                 bbox_to_anchor=(0.99, 0.20))
ax.add_artist(leg1)
ax.legend(handles=met_handles, loc="lower right",
          title="Method (shape) — first row is vorn cluster",
          fontsize=8, title_fontsize=9, framealpha=0.95,
          ncol=2, bbox_to_anchor=(0.99, 0.01))

ax.set_title(
    "Figure 5 — Memory Pareto frontier: vorn methods cluster ~20-22 GB peak allocated;\n"
    "attention-weight methods cluster ~29-58 GB. Same A100-80GB profile, dramatically different memory cost.",
    fontsize=10.5,
)
plt.tight_layout()
plt.savefig("figure5-memory-pareto.pdf", bbox_inches="tight")
plt.close()
print("Figure 5 saved: figure5-memory-pareto.pdf")
