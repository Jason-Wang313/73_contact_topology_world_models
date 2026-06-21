from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

METHOD_ALIASES = {
    "last_contact_persistence": "persist",
    "distance_threshold_graph": "dist-graph",
    "state_only_dynamics_model": "state-dyn",
    "pairwise_contact_classifier": "pairwise",
    "random_forest_topology_planner": "rf-topo",
    "hist_gradient_topology_planner": "hgb-topo",
    "ensemble_uncertainty_planner": "ensemble",
    "conformal_graph_guard": "conformal",
    "risk_averse_graph_planner": "risk-averse",
    "robust_contact_mpc": "robust-mpc",
    "contact_implicit_mpc_baseline": "ci-mpc",
    "topology_world_model_v4": "topo-v4",
    "topology_world_model_v5": "topo-v5",
    "topology_no_memory_ablation": "no-memory",
    "oracle_topology_planner": "oracle",
    "topology_full_v5": "full-v5",
    "ablate_no_birth_death": "no-birth-death",
    "ablate_no_component_head": "no-component",
    "ablate_no_jam_slip": "no-jam-slip",
    "ablate_no_topology_planner": "no-topo-plan",
    "ablate_no_uncertainty_penalty": "no-uncertainty",
    "ablate_no_graph_memory": "no-memory",
    "ablate_no_transition_bonus": "no-transition",
    "ablate_no_fixture_guard": "no-fixture-guard",
    "ablate_no_tail_risk_objective": "no-tail-risk",
    "learned_only_topology_replacement": "learned-only",
}

SPLIT_ALIASES = {
    "nominal_push_to_pocket": "nominal",
    "contact_chain_transfer": "chain",
    "fixture_topology_shift": "fixture",
    "friction_mass_shift": "friction",
    "pocket_relocation": "pocket",
    "fixture_near_wall_jam": "wall-jam",
    "distractor_contact": "distractor",
    "contact_sensor_noise_burst": "noise",
    "actuator_limit_chain": "actuator",
    "delayed_topology_transition": "delayed",
    "combined_stress": "combined",
    "combined_extreme_stress": "extreme",
}


def read_csv(name: str) -> List[Dict[str, str]]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def method_tex(name: str) -> str:
    return r"\texttt{" + tex_escape(METHOD_ALIASES.get(name, name)) + "}"


def split_tex(name: str) -> str:
    return r"\texttt{" + tex_escape(SPLIT_ALIASES.get(name, name)) + "}"


def f(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return tex_escape(value)


def pm(row: Dict[str, str], mean_key: str, ci_key: str, digits: int = 3) -> str:
    return f"${f(row[mean_key], digits)} \\pm {f(row[ci_key], digits)}$"


def take(rows: Sequence[Dict[str, str]], key: str, value: str) -> List[Dict[str, str]]:
    return [row for row in rows if row.get(key) == value]


def sorted_by_float(rows: Iterable[Dict[str, str]], key: str, reverse: bool = True) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)


def summary_fields() -> Dict[str, str]:
    text = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    fields: Dict[str, str] = {"summary_text": text}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower().replace(" ", "_")] = value.strip()
    return fields


def longtable(
    caption: str,
    label: str,
    headers: Sequence[str],
    align: str,
    rows: Sequence[Sequence[str]],
    size: str = r"\scriptsize",
) -> str:
    chunk_size = 38 if size == r"\tiny" else 18
    body = []
    body.append(size)
    for chunk_idx in range(0, len(rows), chunk_size):
        chunk = rows[chunk_idx : chunk_idx + chunk_size]
        body.append(r"\begin{center}")
        if chunk_idx == 0:
            body.append(r"\refstepcounter{table}\label{" + label + r"}\textbf{Table \thetable: " + tex_escape(caption) + r"}\\[0.4ex]")
        else:
            body.append(r"\textbf{Table \ref{" + label + r"} continued}\\[0.4ex]")
        body.append(r"\begin{tabular}{" + align + "}")
        body.append(r"\toprule")
        body.append(" & ".join(headers) + r" \\")
        body.append(r"\midrule")
        for row in chunk:
            body.append(" & ".join(row) + r" \\")
        body.append(r"\bottomrule")
        body.append(r"\end{tabular}")
        body.append(r"\end{center}")
    body.append(r"\normalsize")
    return "\n".join(body)


def figure(path: str, caption: str, width: str = "0.96") -> str:
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width={width}\linewidth]{{../figures/{path}}}",
            r"\caption{" + caption + r"}",
            r"\end{figure}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Paper 73 manuscript from evidence CSVs.")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--figures-dir", default=str(FIGURES))
    parser.add_argument("--paper-dir", default=str(PAPER))
    return parser.parse_args()


def write_references() -> None:
    refs = r"""@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  doi={10.1109/IROS.2012.6386109}
}

@article{posa2014direct,
  title={A direct method for trajectory optimization of rigid bodies through contact},
  author={Posa, Michael and Cantu, Cecilia and Tedrake, Russ},
  journal={The International Journal of Robotics Research},
  volume={33},
  number={1},
  pages={69--81},
  year={2014},
  doi={10.1177/0278364913506757}
}

@inproceedings{battaglia2016interaction,
  title={Interaction networks for learning about objects, relations and physics},
  author={Battaglia, Peter W. and Pascanu, Razvan and Lai, Matthew and Rezende, Danilo and Kavukcuoglu, Koray},
  booktitle={Advances in Neural Information Processing Systems},
  year={2016},
  url={https://arxiv.org/abs/1612.00222}
}

@article{battaglia2018relational,
  title={Relational inductive biases, deep learning, and graph networks},
  author={Battaglia, Peter W. and Hamrick, Jessica B. and Bapst, Victor and Sanchez-Gonzalez, Alvaro and Zambaldi, Vinicius and others},
  journal={arXiv preprint arXiv:1806.01261},
  year={2018},
  url={https://arxiv.org/abs/1806.01261}
}

@inproceedings{sanchezgonzalez2020learning,
  title={Learning to simulate complex physics with graph networks},
  author={Sanchez-Gonzalez, Alvaro and Godwin, Jonathan and Pfaff, Tobias and Ying, Rex and Leskovec, Jure and Battaglia, Peter W.},
  booktitle={International Conference on Machine Learning},
  year={2020},
  url={https://arxiv.org/abs/2002.09405}
}

@article{ha2018world,
  title={World Models},
  author={Ha, David and Schmidhuber, Jurgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018},
  url={https://arxiv.org/abs/1803.10122}
}

@inproceedings{hafner2019planet,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle={International Conference on Machine Learning},
  year={2019},
  url={https://arxiv.org/abs/1811.04551}
}

@inproceedings{hafner2020dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/abs/1912.01603}
}

@article{ebert2018visual,
  title={Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control},
  author={Ebert, Frederik and Finn, Chelsea and Dasari, Sudeep and Xie, Annie and Lee, Alex and Levine, Sergey},
  journal={arXiv preprint arXiv:1812.00568},
  year={2018},
  url={https://arxiv.org/abs/1812.00568}
}

@article{breiman2001random,
  title={Random Forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001},
  doi={10.1023/A:1010933404324}
}

@article{friedman2001greedy,
  title={Greedy function approximation: A gradient boosting machine},
  author={Friedman, Jerome H.},
  journal={The Annals of Statistics},
  volume={29},
  number={5},
  pages={1189--1232},
  year={2001}
}

@article{angelopoulos2021gentle,
  title={A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  author={Angelopoulos, Anastasios N. and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2021},
  url={https://arxiv.org/abs/2107.07511}
}

@article{pedregosa2011scikit,
  title={Scikit-learn: Machine Learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Gael and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011},
  url={https://www.jmlr.org/papers/v12/pedregosa11a.html}
}
"""
    (PAPER / "references.bib").write_text(refs, encoding="utf-8")


def make_main_tex() -> str:
    summary = summary_fields()
    metrics = read_csv("metrics.csv")
    aggregate = read_csv("aggregate_metrics.csv")
    pairwise = read_csv("pairwise_stats.csv")
    aggregate_pairwise = read_csv("aggregate_pairwise_stats.csv")
    fixed_risk = read_csv("fixed_risk_metrics.csv")
    ablation = read_csv("ablation_metrics.csv")
    ablation_aggregate = read_csv("ablation_aggregate_metrics.csv")
    stress = read_csv("stress_sweep.csv")
    seed_metrics = read_csv("raw_seed_metrics.csv")
    negative = read_csv("negative_cases.csv")
    training = read_csv("training_summary.csv")[0]
    config = read_csv("run_config.csv")[0]

    decision = summary.get("terminal_decision", "UNKNOWN")
    reason = summary.get("terminal_reason", "missing terminal reason")
    main_rows = summary.get("main_eval_rows", str(len(read_csv("topology_raw.csv"))))
    ablation_rows = summary.get("ablation_rows", str(len(read_csv("topology_ablation_raw.csv"))))
    stress_rows = summary.get("stress_rows", str(len(read_csv("stress_sweep_raw.csv"))))

    combined = sorted_by_float(take(aggregate, "group", "combined_and_extreme"), "success_rate")
    hard = sorted_by_float(take(aggregate, "group", "hard_splits"), "success_rate")
    all_group = sorted_by_float(take(aggregate, "group", "all_splits"), "success_rate")
    v5_combined = [row for row in combined if row["method"] == "topology_world_model_v5"][0]
    best_combined = [row for row in combined if row["method"] not in {"topology_world_model_v5", "oracle_topology_planner"}][0]
    v5_hard = [row for row in hard if row["method"] == "topology_world_model_v5"][0]
    best_hard = [row for row in hard if row["method"] not in {"topology_world_model_v5", "oracle_topology_planner"}][0]
    fixed_010 = sorted_by_float([row for row in fixed_risk if row["budget"] == "0.10"], "success_at_budget")
    v5_fixed = [row for row in fixed_010 if row["method"] == "topology_world_model_v5"][0]
    best_fixed = [row for row in fixed_010 if row["method"] not in {"topology_world_model_v5", "oracle_topology_planner"}][0]
    abl_full = [row for row in ablation_aggregate if row["method"] == "topology_full_v5"][0]

    method_rows = []
    for row in combined:
        method_rows.append(
            [
                method_tex(row["method"]),
                f(row["success_rate"]),
                f(row["ci95_success_rate"]),
                f(row["edge_f1"]),
                f(row["graph_edit"]),
                f(row["safety_violation_rate"]),
                f(row["final_progress"]),
            ]
        )

    hard_rows = []
    for row in hard:
        hard_rows.append(
            [
                method_tex(row["method"]),
                f(row["success_rate"]),
                f(row["ci95_success_rate"]),
                f(row["edge_f1"]),
                f(row["graph_edit"]),
                f(row["fixture_contact_rate"]),
                f(row["wall_contact_rate"]),
                f(row["jam_f1"]),
            ]
        )

    fixed_rows = []
    for row in fixed_010:
        fixed_rows.append(
            [
                method_tex(row["method"]),
                f(row["success_at_budget"]),
                f(row["mean_diagnostic_risk"]),
                f(row["mean_safety_violation_rate"]),
                f(row["mean_fixture_contact_rate"]),
                f(row["mean_wall_contact_rate"]),
                f(row["mean_jam_f1"]),
            ]
        )

    ablation_rows_tex = []
    for row in sorted_by_float(ablation_aggregate, "success"):
        ablation_rows_tex.append(
            [
                method_tex(row["method"]),
                f(row["success"]),
                f(row["edge_f1"]),
                f(row["graph_edit"]),
                f(row["safety_violation_rate"]),
                f(row["fixture_contact_rate"]),
                f(row["wall_contact_rate"]),
                f(row["jam_f1"]),
            ]
        )

    neg_rows = []
    for row in negative:
        neg_rows.append(
            [
                tex_escape(row["case"]),
                split_tex(row["split"]),
                tex_escape(row["seed"] + "/" + row["episode"]),
                f(row["edge_f1"]),
                f(row["fixture_contact_rate"]),
                f(row["final_progress"]),
                tex_escape(row["lesson"]),
            ]
        )

    full_metric_rows = []
    for row in metrics:
        full_metric_rows.append(
            [
                split_tex(row["split"]),
                method_tex(row["method"]),
                pm(row, "mean_success_rate", "ci95_success_rate"),
                pm(row, "mean_mean_edge_f1", "ci95_mean_edge_f1"),
                pm(row, "mean_mean_graph_edit", "ci95_mean_graph_edit"),
                pm(row, "mean_mean_safety_violation_rate", "ci95_mean_safety_violation_rate"),
                f(row["mean_mean_final_progress"]),
            ]
        )

    aggregate_pair_rows = []
    for row in aggregate_pairwise:
        aggregate_pair_rows.append(
            [
                tex_escape(row["group"]),
                method_tex(row["comparison"]),
                f(row["success_diff"]),
                f(row["edge_f1_diff"]),
                f(row["graph_edit_reduction"]),
                f(row["safety_reduction"]),
            ]
        )

    pair_rows = []
    for row in pairwise:
        pair_rows.append(
            [
                split_tex(row["split"]),
                method_tex(row["comparison"]),
                pm(row, "paired_success_diff", "ci95_success_diff"),
                f(row["paired_edge_f1_diff"]),
                f(row["paired_graph_edit_reduction"]),
                f(row["paired_safety_reduction"]),
                tex_escape(row["reference_better_seeds"] + "/" + row["seeds"]),
            ]
        )

    stress_rows_tex = []
    for row in stress:
        stress_rows_tex.append(
            [
                split_tex(row["split"]),
                method_tex(row["method"]),
                tex_escape(row["stress_level"]),
                pm(row, "mean_success_rate", "ci95_success_rate"),
                pm(row, "mean_edge_f1", "ci95_edge_f1"),
                pm(row, "mean_safety_violation_rate", "ci95_safety_violation_rate"),
                f(row["mean_jam_f1"]),
            ]
        )

    seed_rows = []
    for row in seed_metrics:
        seed_rows.append(
            [
                split_tex(row["split"]),
                method_tex(row["method"]),
                tex_escape(row["seed"]),
                tex_escape(row["episodes"]),
                f(row["success_rate"]),
                f(row["mean_edge_f1"]),
                f(row["mean_graph_edit"]),
                f(row["mean_safety_violation_rate"]),
                f(row["mean_final_progress"]),
            ]
        )

    ablation_full_rows = []
    for row in ablation:
        ablation_full_rows.append(
            [
                split_tex(row["split"]),
                method_tex(row["method"]),
                pm(row, "mean_success_rate", "ci95_success_rate"),
                pm(row, "mean_mean_edge_f1", "ci95_mean_edge_f1"),
                pm(row, "mean_mean_graph_edit", "ci95_mean_graph_edit"),
                pm(row, "mean_mean_safety_violation_rate", "ci95_mean_safety_violation_rate"),
                f(row["mean_mean_jam_f1"]),
            ]
        )

    alias_rows = [[r"\texttt{" + tex_escape(alias) + "}", r"\texttt{" + tex_escape(full) + "}"] for full, alias in sorted(METHOD_ALIASES.items(), key=lambda item: item[1])]

    tiny = r"\tiny"
    alias_table = longtable(
        "Method aliases used in generated tables; CSV files retain the exact full names.",
        "tab:aliases",
        ["Alias", "Full CSV method name"],
        r"lp{0.62\linewidth}",
        alias_rows,
        tiny,
    )
    combined_table = longtable(
        "Combined/extreme aggregate metrics. Success is primary; graph metrics are diagnostic.",
        "tab:combined",
        ["Method", "Succ.", "CI", "Edge F1", "GEdit", "Safety", "Progress"],
        "lcccccc",
        method_rows,
    )
    hard_table = longtable(
        "Hard-split aggregate metrics over all non-nominal regimes.",
        "tab:hard",
        ["Method", "Succ.", "CI", "Edge F1", "GEdit", "Fixture", "Wall", "Jam F1"],
        "lccccccc",
        hard_rows,
    )
    fixed_table = longtable(
        "Fixed-risk success at diagnostic-risk budget 0.10.",
        "tab:fixedrisk",
        ["Method", "Succ@Risk", "Risk", "Safety", "Fixture", "Wall", "Jam F1"],
        "lcccccc",
        fixed_rows,
    )
    ablation_aggregate_table = longtable(
        "Ablation aggregate over frozen ablation splits.",
        "tab:ablationagg",
        ["Ablation", "Succ.", "Edge F1", "GEdit", "Safety", "Fixture", "Wall", "Jam F1"],
        "lccccccc",
        ablation_rows_tex,
    )
    negative_table = longtable(
        "Mechanically selected negative cases for topology_world_model_v5.",
        "tab:negative",
        ["Case", "Split", "Seed/Ep.", "Edge F1", "Fixture", "Prog.", "Lesson"],
        r"lllcccp{0.36\linewidth}",
        neg_rows,
        tiny,
    )
    full_metrics_table = longtable(
        "Full split-method metrics generated from results/metrics.csv.",
        "tab:fullmetrics",
        ["Split", "Method", "Success", "Edge F1", "GEdit", "Safety", "Prog."],
        "llccccc",
        full_metric_rows,
        tiny,
    )
    aggregate_pair_table = longtable(
        "Aggregate topology_world_model_v5 minus comparison method by group.",
        "tab:aggregatepair",
        ["Group", "Comparison", "Succ Diff", "Edge Diff", "GEdit Red.", "Safety Red."],
        "llcccc",
        aggregate_pair_rows,
        tiny,
    )
    pairwise_table = longtable(
        "Per-split paired seed comparisons against topology_world_model_v5.",
        "tab:pairwise",
        ["Split", "Comparison", "Succ Diff", "Edge Diff", "GEdit Red.", "Safety Red.", "Better Seeds"],
        "llccccc",
        pair_rows,
        tiny,
    )
    ablation_full_table = longtable(
        "Full ablation metrics by split and method.",
        "tab:ablationfull",
        ["Split", "Method", "Success", "Edge F1", "GEdit", "Safety", "Jam F1"],
        "llccccc",
        ablation_full_rows,
        tiny,
    )
    stress_full_table = longtable(
        "Stress sweep metrics by split, method, and stress level.",
        "tab:stressfull",
        ["Split", "Method", "Level", "Success", "Edge F1", "Safety", "Jam F1"],
        "lllcccc",
        stress_rows_tex,
        tiny,
    )
    seed_ledger_table = longtable(
        "Seed-level metrics for reproducibility and paired-statistic auditing.",
        "tab:seedledger",
        ["Split", "Method", "Seed", "Eps", "Succ.", "Edge F1", "GEdit", "Safety", "Prog."],
        "lllcccccc",
        seed_rows,
        tiny,
    )

    return rf"""\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{microtype}}
\hypersetup{{colorlinks=false,citebordercolor={{0 1 0}},linkbordercolor={{1 .45 0}},urlbordercolor={{0 .65 1}},pdfborder={{0 0 1.6}}}}
\newtheorem{{proposition}}{{Proposition}}
\newtheorem{{theorem}}{{Theorem}}
\title{{Contact Topology World Models Fail a Frozen MuJoCo Contact-Chain Submission Audit}}
\author{{Anonymous Authors}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We rebuild \emph{{Contact Topology World Models}} under a deliberately adversarial ICLR-style protocol. The scientific question is not whether a model can predict contact graph edges, but whether explicit contact-topology prediction improves downstream contact-rich control once strong learned, geometric, uncertainty-aware, conformal, robust, and contact-implicit baselines are included. The expanded v5 audit evaluates {tex_escape(main_rows)} main rollouts, {tex_escape(ablation_rows)} ablation rollouts, {tex_escape(stress_rows)} stress rollouts, {tex_escape(training["training_examples"])} training scenes, 12 distribution splits, 15 main methods, fixed-risk metrics, paired seed statistics, and maximum-stress gates. The result is terminally negative. The frozen decision is \textbf{{{tex_escape(decision)}}}. On the combined/extreme aggregate, \texttt{{topology\_world\_model\_v5}} reaches {f(v5_combined["success_rate"])} success while the strongest non-oracle baseline, {method_tex(best_combined["method"])}, reaches {f(best_combined["success_rate"])}. On the hard-split aggregate, v5 reaches {f(v5_hard["success_rate"])} while {method_tex(best_hard["method"])} reaches {f(best_hard["success_rate"])}. We report the negative archive rather than optimizing for pretty results.
\end{{abstract}}

\section{{Decision Discipline}}
This manuscript is a submission-readiness audit, not a success narrative. The central claim would be: a contact-topology world model gives a practically meaningful downstream-control advantage over non-topological, weakly topological, robust, and uncertainty-aware alternatives. That claim is allowed only if all frozen gates pass. The recorded terminal reason is:

\begin{{quote}}\small
{tex_escape(reason)}
\end{{quote}}

The archive therefore remains valuable as a negative result: graph-structured prediction can improve diagnostics while failing to improve the control objective that reviewers would care about.

\section{{Problem Setup}}
We study planar contact-chain manipulation in MuJoCo \citep{{todorov2012mujoco}}. A pusher must drive block A into block B and ultimately into a pocket while avoiding fixture and wall jams. At time $t$, the simulator state is $x_t \in \mathbb{{R}}^n$, the action is a bounded pusher command $u_t \in \mathcal{{U}}$, and the observed contact graph is $G_t=(V,E_t)$ over pusher, blocks, walls, fixture, and pocket. The model predicts edge probabilities
\[
  \hat p_\theta(e \in E_{{t+1}} \mid x_t,G_t,u_t), \quad e \in \mathcal{{E}},
\]
and the planner chooses
\[
  u_t^\star = \arg\max_{{u\in\mathcal{{U}}}} S(x_t,G_t,u,\hat p_\theta),
\]
where $S$ mixes predicted chain formation, target progress, lateral alignment, connected-component structure, and penalties for fixture/wall contacts.

The downstream success event $Y$ is intentionally stricter than graph prediction: final pocket progress must be high, lateral target error must be small, the contact chain and pocket contact must occur, and safety/fixture rates must stay below thresholds. This prevents the representation from winning by predicting the graph but selecting bad actions.

\section{{Why Graph Metrics Are Not Enough}}
Interaction networks and graph networks provide strong inductive bias for physical prediction \citep{{battaglia2016interaction,battaglia2018relational,sanchezgonzalez2020learning}}, and world-model work shows that learned latent dynamics can support planning \citep{{ha2018world,hafner2019planet,hafner2020dreamer,ebert2018visual}}. Contact-rich manipulation, however, has a sharper requirement: the representation must change the selected action in a way that improves task success under contact discontinuities.

\begin{{proposition}}[Diagnostic-control non-identifiability]
For a finite action set $\mathcal{{U}}$, there exist two contact-topology predictors $p$ and $q$ such that $p$ has strictly lower edge error than $q$ for every candidate action, yet the induced planner $\arg\max_u S(u,p)$ has lower task success than $\arg\max_u S(u,q)$.
\end{{proposition}}
\begin{{proof}}[Sketch]
Let the task success depend on avoiding one rare but catastrophic fixture edge. Predictor $p$ is more accurate on the majority of benign edges but underestimates the catastrophic edge for the action that appears most progressive. Predictor $q$ is worse on benign edges but overestimates the catastrophic edge and selects a safer action. Edge F1 ranks $p$ higher; downstream success ranks $q$ higher. Thus diagnostic edge metrics do not identify the control-optimal predictor.
\end{{proof}}

\begin{{proposition}}[Topology memory is insufficient under aliased contacts]
If two latent friction/geometry modes share the same observed graph $G_t$ and state projection but have different contact birth probabilities under the same action, then any planner that conditions only on $(x_t,G_t)$ has irreducible action-selection error.
\end{{proposition}}
\begin{{proof}}[Sketch]
The two modes induce the same planner input but different optimal actions. A deterministic policy must choose the same action in both modes, so it is wrong in at least one mode unless the mode-optimal actions coincide. This motivates the stress splits and explains why a topology representation alone cannot rescue hidden friction, actuator-limit, and fixture-shift failures.
\end{{proof}}

\begin{{theorem}}[Ablation necessity gate]
If removing a proposed component leaves success within $\epsilon$ of the full model and does not worsen safety beyond $\delta$, then the experiment does not identify that component as necessary for the claimed mechanism.
\end{{theorem}}
\begin{{proof}}[Sketch]
The full and ablated systems are evaluated under the same randomized seed/split protocol. If the ablation is statistically and practically indistinguishable on the primary objective and safety diagnostics, the observed evidence is compatible with the component being irrelevant. A submission claim of mechanism necessity would therefore overstate what the data identify.
\end{{proof}}

\section{{Frozen Experimental Protocol}}
The protocol was frozen before the full run. It uses seeds {tex_escape(config["seeds"])}, {tex_escape(config["episodes"])} main episodes per seed/split, {tex_escape(config["ablation_episodes"])} ablation episodes, {tex_escape(config["stress_episodes"])} stress episodes, stress levels {tex_escape(config["stress_levels"])}, and single-process CPU execution. The training summary reports state-delta MAE {f(training["state_delta_train_mae"])} and pairwise edge training F1 {f(training["pairwise_edge_train_f1"])}. Implemented learned baselines use scikit-learn \citep{{pedregosa2011scikit}}, random forests \citep{{breiman2001random}}, gradient boosting \citep{{friedman2001greedy}}, and conformal-style residual guarding \citep{{angelopoulos2021gentle}}. Contact-implicit baselines are motivated by trajectory optimization through contact \citep{{posa2014direct}}.

The frozen gates are: hard-regime margin at least 0.030, positive paired lower bound, combined/extreme margin at least 0.030, no diagnostic regression above 0.020, fixed-risk success at budget 0.10 not below the best non-oracle method, maximum-stress success within 0.030 of the best non-oracle method, ablation necessity, and oracle sanity.

\section{{Methods and Baselines}}
The audit includes intentionally uncomfortable baselines: last-contact persistence, distance-threshold graphs, state-only dynamics, pairwise contact classification, random forests, histogram gradient boosting, ensemble uncertainty, conformal graph guarding, risk-averse graph planning, robust contact MPC, contact-implicit MPC, the frozen v4 topology world model, the expanded v5 topology world model, a no-memory topology ablation, and an oracle topology planner. The oracle is not a competitor; it checks whether the benchmark itself permits success when next-step topology is available. Tables use compact aliases to keep the appendix readable; Table~\ref{{tab:aliases}} maps aliases to exact CSV method names.

{alias_table}

\section{{Main Results}}
Table~\ref{{tab:combined}} is the decisive combined/extreme aggregate. The proposed v5 method does not clear the success margin and does not dominate the strongest non-oracle alternative.

{combined_table}

Table~\ref{{tab:hard}} reports the broader hard-split aggregate. The same pattern holds: graph-aware models may improve some intermediate metrics, but the strongest downstream method is not decisively beaten.

{hard_table}

{figure("topology_success_by_split.png", "Task success across frozen splits. The visual pattern is reported as generated, including cases where the proposed model does not separate from baselines.")}

{figure("topology_edge_f1_by_split.png", "Contact-edge F1 across splits. Better edge diagnostics are not treated as sufficient for submission readiness.")}

\section{{Fixed-Risk and Stress Evidence}}
Submission review would ask whether the method can trade small amounts of risk for robust success. Table~\ref{{tab:fixedrisk}} answers at the frozen diagnostic-risk budget 0.10. The v5 success-at-budget is {f(v5_fixed["success_at_budget"])}, while the strongest non-oracle fixed-risk baseline {method_tex(best_fixed["method"])} reaches {f(best_fixed["success_at_budget"])}.

{fixed_table}

The stress sweep varies hidden contact difficulty, friction/mass, sensor noise, and actuation pressure. It is deliberately not tuned to make v5 look clean; its purpose is to expose brittle cases.

{figure("topology_stress_sweep.png", "Stress sweep at frozen stress levels. The maximum-stress gate is computed from CSV values, not from visual inspection.")}

\section{{Ablations}}
The full v5 mechanism is only credible if removed components hurt success or safety. The full ablation aggregate has success {f(abl_full["success"])}. Table~\ref{{tab:ablationagg}} shows that several ablations remain too close to full v5, which is a direct mechanism-identification failure.

{ablation_aggregate_table}

{figure("topology_ablation_success.png", "Ablation success across frozen hard splits. Near-ties are counted against the mechanism claim.")}

\section{{Negative Cases}}
Table~\ref{{tab:negative}} lists representative v5 failures selected mechanically from the CSV. These cases are not anecdotes chosen after the fact; they are part of the predefined failure analysis.

{negative_table}

\section{{Related Work Pressure}}
The result should be interpreted against strong prior work. MuJoCo makes local contact simulation convenient but not equivalent to hardware validation \citep{{todorov2012mujoco}}. Contact-implicit optimization directly models mode-free contact planning pressure \citep{{posa2014direct}}. Graph and interaction networks justify relational representations \citep{{battaglia2016interaction,battaglia2018relational,sanchezgonzalez2020learning}}, but they do not guarantee downstream control gains. World-model and visual foresight papers demonstrate planning through learned latent models \citep{{ha2018world,hafner2019planet,hafner2020dreamer,ebert2018visual}}. The correct bar for this paper is therefore not novelty of a contact graph, but evidence that the graph improves robust manipulation decisions.

\section{{Limitations and Terminal State}}
This is still a local MuJoCo audit: no hardware, no public benchmark validation, no vision/tactile data, and no large-scale learned contact corpus. Those limitations would remain even if v5 passed. Since v5 fails frozen gates, the honest terminal state is \textbf{{{tex_escape(decision)}}}. The contribution is a reproducible negative archive showing why contact-topology diagnostics should not be confused with control success.

\clearpage
\appendix
\section{{Full Split-Method Metrics}}
{full_metrics_table}

\section{{Aggregate Pairwise Comparisons}}
{aggregate_pair_table}

\section{{Paired Seed Comparisons}}
{pairwise_table}

\section{{Full Ablation Metrics}}
{ablation_full_table}

\section{{Full Stress Sweep}}
{stress_full_table}

\section{{Seed-Level Evidence Ledger}}
{seed_ledger_table}

\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}

\end{{document}}
"""


def main() -> None:
    global RESULTS, FIGURES, PAPER

    args = parse_args()
    RESULTS = Path(args.results_dir)
    FIGURES = Path(args.figures_dir)
    PAPER = Path(args.paper_dir)
    if not RESULTS.is_absolute():
        RESULTS = ROOT / RESULTS
    if not FIGURES.is_absolute():
        FIGURES = ROOT / FIGURES
    if not PAPER.is_absolute():
        PAPER = ROOT / PAPER
    PAPER.mkdir(parents=True, exist_ok=True)
    write_references()
    tex = make_main_tex()
    (PAPER / "main.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {PAPER / 'main.tex'}")
    print(f"wrote {PAPER / 'references.bib'}")


if __name__ == "__main__":
    main()
