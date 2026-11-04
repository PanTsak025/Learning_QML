"""Paper figures and tables, built directly from the result files (no training).

    python make_figures.py            # writes figures/*.pdf, figures/*.png and figures/*.tex

Figures (IEEE sizes: single column 3.5 in, double column 7.16 in; vector PDF + 300 dpi PNG):
  fig_pipeline          the method at a glance (double column)
  fig_best              best VQC vs best classical vs RF on all 33 features, per scenario   RQ1, RQ2
  fig_learning_curves   test macro-F1 vs training size, four representative scenarios       RQ3, RQ4
  fig_learning_curves_all   the same for all nine scenarios (supplementary)
  fig_qubits            F1 vs qubits: VQC (training and test) vs classical, binary / 8-class RQ5
  fig_vqc_mlp           VQC minus parameter-matched MLP per cell, by qubits and by size     RQ2
  fig_realistic         detection, false-alarm rate, precision on realistic class mix
  fig_cost              CPU seconds per training vs training size (single column)
Tables (LaTeX, booktabs):
  tab_cleaning          rows removed per attack family
  tab_main              best quantum vs best classical per scenario
  tab_data_requirement  n_A and n_B per scenario and sampling method

Selection rules are the study's own: every "best" model or setting is chosen on VALIDATION
macro-F1 (mean over seeds) and reported on TEST macro-F1 +- standard error over the 3 seeds.
Colors: validated colorblind-safe categorical palette (dataviz reference palette, slots 1-4);
because two slots are below 3:1 contrast on white, every series also has its own marker.
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from run_experiments import tasks_for, task_id

OUT = "results/main"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, MUTED, GRID = "#8a8986", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"VQC": BLUE, "classical": ORANGE, "MLP": AQUA, "ceiling": GREY}
NAMES = {"DDoS": "DDoS", "DoS": "DoS", "Mirai": "Mirai", "Recon": "Recon", "Spoofing": "Spoofing",
         "BruteForce": "Brute force", "Web": "Web", "benign_vs_attack": "All attacks", "multiclass": "8-class"}
ORDER = list(NAMES)
MODEL = {"VQC-RY": "VQC", "VQC-ZZ": "VQC (ZZ)", "LogReg": "LogReg", "SVM-RBF": "SVM", "RF": "RF",
         "MLP-matched": "MLP (matched)", "RF-all33": "RF (33 features)"}
CLASSICAL = ["LogReg", "SVM-RBF", "RF", "MLP-matched"]
SINGLE, DOUBLE = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "lines.markersize": 4, "grid.color": GRID, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG}/{name}.pdf / .png")


def load_runs():
    runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_*.csv")], ignore_index=True)
    runs = runs.drop_duplicates(["task", "model"])
    runs["setting"] = runs["reducer"].str.upper() + "-" + runs["n_qubits"].astype(str) + "q"
    return runs


def stage(runs, *names):
    ids = set()
    for n in names:
        ids |= {task_id(t) for t in tasks_for(n)[0]}
    return runs[runs.task.isin(ids)]


def mean_se(s):
    return s.mean(), (s.std() / np.sqrt(s.count()) if s.count() > 1 else np.nan)


# ------------------------------------------------------------------ best per scenario
def best_overall(res, models):
    """Best (setting, model) on mean VALIDATION F1; its TEST F1 mean and SE."""
    d = res[res.model.isin(models)]
    val = d.groupby(["setting", "model"])["val_macro_f1"].mean()
    if val.empty:
        return None
    st, m = val.idxmax()
    t = d[(d.setting == st) & (d.model == m)]["test_macro_f1"]
    mu, se = mean_se(t)
    return {"setting": st, "model": m, "f1": mu, "se": se}


def table_best(runs):
    res = stage(runs, "first_look", "qubits_features")
    rows = []
    for sc in ORDER:
        d = res[res.scenario == sc]
        if d.empty:
            continue
        q = best_overall(d, ["VQC-RY", "VQC-ZZ"])
        c = best_overall(d, CLASSICAL)
        ceil = d[d.model == "RF-all33"]["test_macro_f1"].mean()
        rows.append({"scenario": sc, "q": q, "c": c, "ceiling": ceil})
    return rows


def bar_label(ax, x, y, v, fmt="{:.3f}"):
    ax.text(x, y, fmt.format(v), va="center", ha="left", fontsize=5.5, color=MUTED)


def fig_best(rows):
    """Per scenario: best VQC, best classical on the SAME features, RF on all 33 features.
    'Best' = highest mean validation macro-F1 over all settings; bars = test macro-F1."""
    series = [("q", "VQC (best setting)", COL["VQC"]),
              ("c", "Best classical, same features", COL["classical"]),
              ("ceiling", "RF, all 33 features", COL["ceiling"])]
    fig, ax = plt.subplots(figsize=(SINGLE, 4.0))
    h = 0.26
    y = np.arange(len(rows))[::-1].astype(float)
    for j, (key, lab, col) in enumerate(series):
        yy = y + (1 - j) * (h + 0.02)
        vals = [r[key]["f1"] if key != "ceiling" else r["ceiling"] for r in rows]
        ax.barh(yy, vals, height=h, color=col, label=lab)
        for yi, v in zip(yy, vals):
            bar_label(ax, v + 0.01, yi, v)
    ax.set_yticks(y, [NAMES[r["scenario"]] for r in rows])
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Test macro-F1")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_best")


def tex_main(rows):
    lines = [r"\begin{tabular}{lllr}", r"\toprule",
             r"Scenario & Best VQC (setting) & Best classical (model, setting) & $\Delta$ \\", r"\midrule"]
    for r in rows:
        q, c = r["q"], r["c"]
        lines.append(f"{NAMES[r['scenario']]} & {q['f1']:.3f} $\\pm$ {q['se']:.3f} ({q['setting']}) & "
                     f"{c['f1']:.3f} $\\pm$ {c['se']:.3f} ({MODEL[c['model']]}, {c['setting']}) & "
                     f"{q['f1'] - c['f1']:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{FIG}/tab_main.tex", "w").write("\n".join(lines) + "\n")
    print(f"wrote {FIG}/tab_main.tex")


# ------------------------------------------------------------------ learning curves
def curve_rows(runs):
    lc = stage(runs, "learning_curves")
    out = []
    for (sc, samp, n), d in lc.groupby(["scenario", "sampling", "n_train"]):
        row = {"scenario": sc, "sampling": samp, "n": n}
        for m, key in [("VQC-RY", "vqc"), ("MLP-matched", "mlp"), ("RF-all33", "ceil")]:
            row[key], row[key + "_se"] = mean_se(d[d.model == m]["test_macro_f1"])
        cls = d[d.model.isin(["LogReg", "SVM-RBF", "RF"])].groupby("model")["val_macro_f1"].mean()
        best = cls.idxmax()                                   # chosen on validation at this size
        row["cls"], row["cls_se"] = mean_se(d[d.model == best]["test_macro_f1"])
        out.append(row)
    return pd.DataFrame(out)


def draw_curves(ax, c, sc):
    r = c[(c.scenario == sc) & (c.sampling == "random")].sort_values("n")
    k = c[(c.scenario == sc) & (c.sampling == "kmeans")].sort_values("n")
    ax.plot(r.n, r.ceil, ls=":", color=GREY, lw=1.0, label="RF, all 33 features")
    for d, key, col, mk, ls, lab in [(r, "cls", COL["classical"], "D", "-", "Best classical"),
                                     (r, "mlp", COL["MLP"], "^", "-", "MLP (matched)"),
                                     (k, "vqc", COL["VQC"], "s", "--", "VQC, k-means"),
                                     (r, "vqc", COL["VQC"], "o", "-", "VQC, random")]:
        ax.errorbar(d.n, d[key], yerr=d[key + "_se"], color=col, marker=mk, ls=ls, ms=3.2, mec="white",
                    mew=0.4, elinewidth=0.6, capsize=0, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([25, 50, 100, 200, 400, 800], ["25", "50", "100", "200", "400", "800"])
    ax.minorticks_off()
    ax.set_title(NAMES[sc], loc="left", fontweight="bold")
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def fig_learning_curves(c):
    # Four representative scenarios, one per behaviour seen in the results: a flood (all models
    # near perfect), Recon (information-limited, VQC ties), Web (gap widens with data), 8-class.
    pick = [s for s in ["DDoS", "Recon", "Web", "multiclass"] if s in set(c.scenario)]
    fig, axes = plt.subplots(1, len(pick), figsize=(DOUBLE, 1.9))
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, pick):
        draw_curves(ax, c, sc)
        ax.set_xlabel("Training samples per class")
    axes[0].set_ylabel("Test macro-F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_learning_curves")
    present = [s for s in ORDER if s in set(c.scenario)]
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE, 5.6))
    for ax, sc in zip(axes.flat, present):
        draw_curves(ax, c, sc)
    for ax in axes.flat[len(present):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Training samples per class")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test macro-F1")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, "fig_learning_curves_all")


# ------------------------------------------------------------------ qubits
def qubit_rows(runs):
    """Per (scenario, reducer, qubits) cell of the qubits stage: test F1 of the VQC, its
    training F1, the best same-feature classical model (chosen on validation), the matched
    MLP and RF on all 33 features. Averaged over seeds, then over cells per qubit count."""
    qf = stage(runs, "qubits_features")
    out = []
    for (sc, red, nq), g in qf.groupby(["scenario", "reducer", "n_qubits"]):
        cls = g[g.model.isin(["LogReg", "SVM-RBF", "RF"])].groupby("model")["val_macro_f1"].mean()
        f = lambda m, col="test_macro_f1": g[g.model == m][col].mean()
        out.append({"type": "8-class" if sc == "multiclass" else "binary", "n": nq,
                    "vqc": f("VQC-RY"), "vqc_train": f("VQC-RY", "train_macro_f1"),
                    "cls": f(cls.idxmax()), "mlp": f("MLP-matched"), "ceil": f("RF-all33")})
    return pd.DataFrame(out).groupby(["type", "n"]).mean().reset_index()


def fig_qubits(runs):
    d = qubit_rows(runs)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
    spec = [("ceil", "RF, all 33 features", GREY, "", ":"),
            ("cls", "Best classical, same features", COL["classical"], "D", "-"),
            ("mlp", "MLP, same size as VQC", COL["MLP"], "^", "-"),
            ("vqc_train", "VQC, training data", COL["VQC"], "o", "--"),
            ("vqc", "VQC, test data", COL["VQC"], "o", "-")]
    for ax, kind, title, ylim in [(axes[0], "binary", "(a) Binary detection, mean of 8 scenarios", (0.76, 0.9)),
                                  (axes[1], "8-class", "(b) 8-class", (0.15, 0.75))]:
        g = d[d.type == kind]
        for key, lab, col, mk, ls in spec:
            ax.plot(g.n, g[key], color=col, marker=mk or None, ls=ls, ms=3.6, mec="white", mew=0.4,
                    mfc="white" if key == "vqc_train" else col, label=lab, lw=1.4 if mk else 1.0)
        for n, v in zip(g.n, g.vqc):                       # VQC test values, the main series
            ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points", xytext=(0, -9),
                        ha="center", fontsize=5.5, color=MUTED)
        ax.set_xticks(sorted(g.n))
        ax.set_xlabel("Qubits (= input features)")
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro-F1")
    axes[1].text(4.0, 0.17, "VQC fits even its training data worse with more qubits",
                 ha="left", va="bottom", fontsize=6, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_qubits")


# ------------------------------------------------------------------ realistic traffic
def fig_realistic():
    path = f"{OUT}/eval_extra_all.csv"
    if not os.path.exists(path):
        print("skip fig_realistic: run `python evaluate_extra.py --summary` first")
        return
    d = pd.read_csv(path)
    d = d[(d.eval_set == "realistic") & (d.n_train == d.n_train.max()) & (d.scenario != "multiclass")]
    test = pd.read_csv("prepared/cleaning_report_by_family.csv", index_col=0)["test"]
    share = {s: test[s] / (test[s] + test["Benign"]) for s in ORDER if s in test}
    share["benign_vs_attack"] = 1 - test["Benign"] / test["TOTAL"]
    models = [("VQC-RY", COL["VQC"], "VQC"), ("RF", COL["classical"], "RF, same features"),
              ("RF-all33", GREY, "RF, all 33 features")]
    scen = [s for s in ORDER if s in set(d.scenario)][::-1]
    labels = [f"{NAMES[s]}\n{100 * share[s]:.1f}% attacks" for s in scen]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.3), sharey=True)
    h = 0.26
    for ax, col, title in [(axes[0], "detection_rate", "Detection rate (higher is better)"),
                           (axes[1], "false_alarm_rate", "False-alarm rate (lower is better)"),
                           (axes[2], "precision", "Precision (higher is better)")]:
        g = d.groupby(["scenario", "model"])[col].mean()
        for j, (m, c, lab) in enumerate(models):
            yy = np.arange(len(scen)) + (1 - j) * (h + 0.02)
            vals = [g.get((s, m), np.nan) for s in scen]
            ax.barh(yy, vals, height=h, color=c, label=lab)
            for yi, v in zip(yy, vals):
                bar_label(ax, v + 0.015, yi, v, "{:.2f}")
        ax.set_xlim(0, 1.2)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_title(title, loc="left")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(scen)), labels)
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_realistic")


# ------------------------------------------------------------------ VQC vs equal-size MLP
def paired_gaps(runs, stage_name, keys):
    d = stage(runs, stage_name)
    out = []
    for key, g in d.groupby(keys):
        p = g.pivot_table(index="seed", columns="model", values="test_macro_f1")
        if "VQC-RY" in p and "MLP-matched" in p:
            out.append({**dict(zip(keys, key)), "gap": float((p["VQC-RY"] - p["MLP-matched"]).mean())})
    return pd.DataFrame(out)


def fig_vqc_mlp(runs):
    """One dot per comparison cell (mean over 3 seeds): VQC minus parameter-matched MLP, test
    macro-F1. p-values: Wilcoxon signed-rank over cells, Holm-corrected (analyze_claims.py)."""
    sig = pd.read_csv(f"{OUT}/claims_significance.csv").set_index("comparison")["p_holm"]
    p = lambda name: sig.get(f"VQC vs MLP-matched: {name}", np.nan)
    qf = paired_gaps(runs, "qubits_features", ["scenario", "reducer", "n_qubits"])
    lc = paired_gaps(runs, "learning_curves", ["scenario", "sampling", "n_train"])
    rng = np.random.default_rng(0)                         # jitter only; same picture every run
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, d, xcol, xs, title in [(axes[0], qf, "n_qubits", [2, 4, 6, 8], "(a) By qubits (200 per class)"),
                                   (axes[1], lc, "n_train", [25, 50, 100, 200, 400, 800],
                                    "(b) By training size (learning curves)")]:
        pos = {v: i for i, v in enumerate(xs)}
        for kind, mk, col in [("binary", "o", COL["VQC"]), ("8-class", "D", COL["classical"])]:
            s = d[(d.scenario == "multiclass") == (kind == "8-class")]
            x = s[xcol].map(pos) + rng.uniform(-0.18, 0.18, len(s))
            ax.scatter(x, s.gap, s=9, marker=mk, color=col, edgecolor="white", linewidth=0.3,
                       label=f"{kind} cell", zorder=3)
        med = d.groupby(xcol)["gap"].median()
        for v, m in med.items():
            ax.plot([pos[v] - 0.3, pos[v] + 0.3], [m, m], color=INK, lw=1.2, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.7, zorder=2)
        ax.set_xticks(range(len(xs)), [str(v) for v in xs])
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_xlabel("Qubits")
    axes[1].set_xlabel("Training samples per class")
    axes[0].set_ylabel("VQC $-$ MLP, test macro-F1")
    lo = min(qf.gap.min(), lc.gap.min())
    axes[0].text(0.5, 0.97, f"2\u20134 qubits: p = {p('qubits stage, 2-4 qubits'):.2f} (no difference)",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=6, color=INK)
    axes[0].text(0.5, 0.06, f"6\u20138 qubits: p = {p('qubits stage, 6-8 qubits'):.1e} (MLP better)",
                 transform=axes[0].transAxes, ha="center", va="bottom", fontsize=6, color=INK)
    axes[1].text(0.5, 0.06, f"All sizes: p = {p('learning curves, all'):.2f} (no difference)",
                 transform=axes[1].transAxes, ha="center", va="bottom", fontsize=6, color=INK)
    axes[0].set_ylim(lo - 0.05, max(qf.gap.max(), lc.gap.max()) + 0.05)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK, lw=1.2))
    l.append("median")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_vqc_mlp")


# ------------------------------------------------------------------ cost
def fig_cost():
    path = f"{OUT}/claims_cost.csv"
    if not os.path.exists(path):
        print("skip fig_cost: run `python analyze_claims.py --part cost` first")
        return
    c = pd.read_csv(path)
    b = c[c.type == "binary"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    for nq, shade, mk in [(2, "#9ec5f4", "o"), (4, "#2a78d6", "o"), (6, "#0f3f80", "o")]:
        g = b[b.n_qubits == nq]
        ax.plot(g.n_train, g["VQC-RY"], color=shade, marker=mk, ms=3.2, mec="white", mew=0.4,
                label=f"VQC, {nq} qubits")
    g4 = b[b.n_qubits == 4]
    for m, col, mk, lab in [("RF", COL["classical"], "D", "RF"), ("MLP-matched", COL["MLP"], "^", "MLP (matched)"),
                            ("SVM-RBF", YELLOW, "s", "SVM")]:
        ax.plot(g4.n_train, g4[m].clip(lower=0.05), color=col, marker=mk, ms=3.2, mec="white", mew=0.4, label=lab)
    v, r = g4["VQC-RY"].iloc[-1], g4["RF"].iloc[-1]
    ax.text(25, 12, f"At 800 per class, 4 qubits: VQC {v:,.0f} s vs RF {r:.1f} s ({v / r:,.0f}$\\times$ slower)",
            ha="left", va="center", fontsize=6, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([25, 50, 100, 200, 400, 800], ["25", "50", "100", "200", "400", "800"])
    ax.minorticks_off()
    ax.set_xlabel("Training samples per class")
    ax.set_ylabel("CPU seconds per training (median)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", frameon=False, fontsize=6, ncol=3, bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.8, handlelength=1.6)
    ax.text(0.0, -0.27, "Binary scenarios. Classical times include 3-value tuning; values below 0.05 s are drawn at 0.05 s.",
            transform=ax.transAxes, fontsize=5.5, color=MUTED)
    save(fig, "fig_cost")


# ------------------------------------------------------------------ pipeline diagram
def fig_pipeline():
    rep = pd.read_csv("prepared/cleaning_report_by_family.csv", index_col=0)
    raw, clean = rep.loc["TOTAL", "raw"], rep.loc["TOTAL", "clean"]
    m = lambda v: f"{v / 1e6:.1f}M"
    steps = [("CIC-IoT-2023", f"63 merged CSVs\n{m(raw)} rows\n39 features"),
             ("Cleaning", f"invalid values\nduplicates, conflicts\n$\\rightarrow$ {m(clean)} rows"),
             ("Split", "hash split 80/20\nbalanced\nscenarios"),
             ("Features", "log1p, standardize\nPCA or MI\n$\\rightarrow$ $n$ = qubits"),
             ("VQC", "RY encoding\nEfficientSU2\n2 repetitions"),
             ("Training", "last-qubit readout\ncross-entropy\nCOBYLA")]
    fig, ax = plt.subplots(figsize=(DOUBLE, 1.35))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1.35)
    ax.axis("off")
    for i, (t, sub) in enumerate(steps):
        quantum = t in ("VQC", "Training")
        ax.add_patch(FancyBboxPatch((i + 0.08, 0.5), 0.84, 0.8, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc="#eaf2fb" if quantum else "#f4f3ef", ec=BLUE if quantum else MUTED, lw=0.7))
        ax.text(i + 0.5, 1.17, t, ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
        ax.text(i + 0.5, 0.82, sub, ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=1.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.08, 0.9), xytext=(i + 0.92, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7))
    ax.annotate("", xy=(3.5, 0.3), xytext=(3.5, 0.5), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.7))
    ax.add_patch(FancyBboxPatch((1.6, 0.02), 3.8, 0.27, boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc="#fdf0ea", ec=ORANGE, lw=0.7))
    ax.text(3.5, 0.155, "Same features $\\rightarrow$ LogReg, SVM, RF, parameter-matched MLP;  RF on all 33 features",
            ha="center", va="center", fontsize=5.8, color=INK)
    save(fig, "fig_pipeline")


# ------------------------------------------------------------------ tables
def tex_cleaning():
    rep = pd.read_csv("prepared/cleaning_report_by_family.csv", index_col=0)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Family & Raw & Duplicates & Conflicts & Clean & Clean (\%) \\", r"\midrule"]
    fmt = lambda v: f"{int(v):,}".replace(",", r"\,")
    for fam, r in rep.iterrows():
        if fam == "TOTAL":
            lines.append(r"\midrule")
        name = {"BruteForce": "Brute force", "TOTAL": "Total"}.get(fam, fam)
        lines.append(f"{name} & {fmt(r.raw)} & {fmt(r.duplicate)} & {fmt(r.conflict)} & {fmt(r.clean)} & "
                     f"{100 * r.clean / r.raw:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{FIG}/tab_cleaning.tex", "w").write("\n".join(lines) + "\n")
    print(f"wrote {FIG}/tab_cleaning.tex")


def tex_data_requirement():
    a_path, b_path = f"{OUT}/learning_curves_A_match_classical.csv", f"{OUT}/learning_curves_B_plateau.csv"
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        print("skip tab_data_requirement: run `python analyze_learning_curves.py` first")
        return
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    short = lambda v: (r"$>$800" if str(v).startswith("not within") else
                       ("--" if str(v).startswith("no stable") else str(v).replace(".0", "")))
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r" & & $n_A$ & \multicolumn{4}{c}{$n_B$ (plateau)} \\ \cmidrule(lr){4-7}",
             r"Scenario & Sampling & VQC = classical & VQC & MLP & SVM & RF \\", r"\midrule"]
    for sc in ORDER:
        for samp in ["random", "kmeans"]:
            ra, rb = a[(a.scenario == sc) & (a.sampling == samp)], b[(b.scenario == sc) & (b.sampling == samp)]
            if ra.empty or rb.empty:
                continue
            rb = rb.iloc[0]
            lines.append(f"{NAMES[sc] if samp == 'random' else ''} & {'k-means' if samp == 'kmeans' else 'random'} & "
                         f"{short(ra.iloc[0].n_A_matches_classical)} & {short(rb['n_B_VQC-RY'])} & "
                         f"{short(rb['n_B_MLP-matched'])} & {short(rb['n_B_SVM-RBF'])} & {short(rb['n_B_RF'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{FIG}/tab_data_requirement.tex", "w").write("\n".join(lines) + "\n")
    print(f"wrote {FIG}/tab_data_requirement.tex  (-- = no stable plateau)")


if __name__ == "__main__":
    runs = load_runs()
    fig_pipeline()
    rows = table_best(runs)
    fig_best(rows)
    tex_main(rows)
    fig_learning_curves(curve_rows(runs))
    fig_qubits(runs)
    fig_realistic()
    fig_vqc_mlp(runs)
    fig_cost()
    tex_cleaning()
    tex_data_requirement()