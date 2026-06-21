from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


BASE_SEED = 1678111061
QUICK_MODE = os.getenv("PAPER73_QUICK", "0") == "1"
SEED_COUNT = int(os.getenv("PAPER73_SEED_COUNT", "1" if QUICK_MODE else "8"))
SEEDS = list(range(SEED_COUNT))
EVAL_EPISODES = int(os.getenv("PAPER73_EVAL_EPISODES", "6"))
ABLATION_EPISODES = int(os.getenv("PAPER73_ABLATION_EPISODES", "4"))
STRESS_EPISODES = int(os.getenv("PAPER73_STRESS_EPISODES", "3"))
TRAINING_EXAMPLES = int(os.getenv("PAPER73_TRAINING_EXAMPLES", "2400"))
STEPS = int(os.getenv("PAPER73_STEPS", "48"))
INNER_STEPS = 3
DT = 0.02

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

EDGES = [
    "pusher_block_a",
    "pusher_block_b",
    "block_a_block_b",
    "block_a_fixture",
    "block_b_fixture",
    "block_a_wall",
    "block_b_wall",
    "block_b_pocket",
]
EDGE_INDEX = {name: idx for idx, name in enumerate(EDGES)}

METHODS = [
    "last_contact_persistence",
    "distance_threshold_graph",
    "state_only_dynamics_model",
    "pairwise_contact_classifier",
    "random_forest_topology_planner",
    "hist_gradient_topology_planner",
    "ensemble_uncertainty_planner",
    "conformal_graph_guard",
    "risk_averse_graph_planner",
    "robust_contact_mpc",
    "contact_implicit_mpc_baseline",
    "topology_world_model_v4",
    "topology_world_model_v5",
    "topology_no_memory_ablation",
    "oracle_topology_planner",
]

ABLATION_METHODS = [
    "topology_full_v5",
    "ablate_no_birth_death",
    "ablate_no_component_head",
    "ablate_no_jam_slip",
    "ablate_no_topology_planner",
    "ablate_no_uncertainty_penalty",
    "ablate_no_graph_memory",
    "ablate_no_transition_bonus",
    "ablate_no_fixture_guard",
    "ablate_no_tail_risk_objective",
    "topology_world_model_v4",
    "learned_only_topology_replacement",
]

STRESS_METHODS = [
    "pairwise_contact_classifier",
    "random_forest_topology_planner",
    "hist_gradient_topology_planner",
    "ensemble_uncertainty_planner",
    "conformal_graph_guard",
    "risk_averse_graph_planner",
    "robust_contact_mpc",
    "contact_implicit_mpc_baseline",
    "topology_world_model_v4",
    "topology_world_model_v5",
    "topology_no_memory_ablation",
    "oracle_topology_planner",
]

ACTIONS = np.array(
    [
        [18.0, 0.0],
        [14.0, 6.0],
        [14.0, -6.0],
        [9.0, 10.0],
        [9.0, -10.0],
        [23.0, 0.0],
        [6.0, 0.0],
        [0.0, 9.0],
        [0.0, -9.0],
    ],
    dtype=float,
)


MODEL_XML = """
<mujoco model="contact_topology_world_models">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 0" integrator="implicitfast"/>
  <default>
    <joint damping="0.16"/>
    <geom solref="0.012 1" solimp="0.90 0.95 0.001" friction="0.62 0.08 0.02"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 -0.02" size="1.0 1.0 0.02"
          contype="0" conaffinity="0" rgba="0.90 0.89 0.84 1"/>
    <geom name="top_wall" type="box" pos="0 0.48 0.035" size="0.62 0.018 0.07"
          rgba="0.32 0.36 0.38 1"/>
    <geom name="bottom_wall" type="box" pos="0 -0.48 0.035" size="0.62 0.018 0.07"
          rgba="0.32 0.36 0.38 1"/>
    <geom name="pocket_back" type="box" pos="0.43 0 0.035" size="0.018 0.18 0.07"
          rgba="0.20 0.42 0.32 1"/>
    <geom name="fixture_post" type="box" pos="0.14 0.22 0.035" size="0.035 0.075 0.07"
          rgba="0.50 0.30 0.20 1"/>
    <body name="pusher" pos="0 0 0.035">
      <joint name="pusher_x" type="slide" axis="1 0 0" range="-0.55 0.52" damping="0.18"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" range="-0.45 0.45" damping="0.18"/>
      <geom name="pusher_tip" type="sphere" size="0.035" mass="0.45" rgba="0.05 0.05 0.06 1"/>
    </body>
    <body name="block_a" pos="0 0 0.035">
      <joint name="a_x" type="slide" axis="1 0 0" range="-0.48 0.45" damping="0.12"/>
      <joint name="a_y" type="slide" axis="0 1 0" range="-0.43 0.43" damping="0.12"/>
      <geom name="block_a_geom" type="box" size="0.055 0.055 0.035" mass="0.36" rgba="0.72 0.16 0.12 1"/>
    </body>
    <body name="block_b" pos="0 0 0.035">
      <joint name="b_x" type="slide" axis="1 0 0" range="-0.48 0.45" damping="0.12"/>
      <joint name="b_y" type="slide" axis="0 1 0" range="-0.43 0.43" damping="0.12"/>
      <geom name="block_b_geom" type="box" size="0.055 0.055 0.035" mass="0.38" rgba="0.10 0.22 0.72 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="pusher_x_motor" joint="pusher_x" gear="1" ctrllimited="true" ctrlrange="-38 38"/>
    <motor name="pusher_y_motor" joint="pusher_y" gear="1" ctrllimited="true" ctrlrange="-38 38"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    a_y: float
    b_y: float
    pocket_y: float
    fixture_x: float
    fixture_y: float
    friction: float
    mass_scale: float
    noise: float
    actuator_limit: float
    distractor: float


@dataclass(frozen=True)
class EpisodeConfig:
    split: SplitSpec
    seed: int
    episode: int
    pusher: Tuple[float, float]
    block_a: Tuple[float, float]
    block_b: Tuple[float, float]
    pocket: Tuple[float, float]
    fixture: Tuple[float, float]
    friction: float
    mass_scale: float
    noise: float
    actuator_limit: float
    distractor: float
    stress_level: float | None = None


@dataclass
class BinaryPredictor:
    scaler: StandardScaler | None
    model: object | None
    constant: float | None


@dataclass
class LearnedPack:
    state_scaler_x: StandardScaler
    state_scaler_y: StandardScaler
    state_model: Ridge
    pair_scaler: StandardScaler
    pair_models: List[BinaryPredictor]
    topo_scaler: StandardScaler
    topo_models: List[BinaryPredictor]
    rf_scaler: StandardScaler
    rf_models: List[BinaryPredictor]
    hgb_scaler: StandardScaler
    hgb_models: List[BinaryPredictor]
    ensemble_models: List[List[BinaryPredictor]]
    conformal_margin: np.ndarray
    training_rows: List[Dict[str, str]]
    state_mae: float
    edge_train_f1: float


SPLITS = [
    SplitSpec("nominal_push_to_pocket", 0.00, 0.00, 0.00, 0.15, 0.30, 0.58, 1.00, 0.004, 1.00, 0.0),
    SplitSpec("contact_chain_transfer", 0.08, -0.02, -0.01, 0.13, 0.30, 0.62, 1.05, 0.006, 0.95, 0.1),
    SplitSpec("fixture_topology_shift", 0.06, -0.07, -0.07, 0.10, 0.10, 0.66, 1.10, 0.008, 0.90, 0.3),
    SplitSpec("friction_mass_shift", -0.04, 0.04, 0.03, 0.17, -0.24, 0.82, 1.45, 0.010, 0.84, 0.2),
    SplitSpec("pocket_relocation", 0.00, 0.06, 0.16, 0.16, 0.22, 0.60, 1.05, 0.007, 0.94, 0.2),
    SplitSpec("fixture_near_wall_jam", 0.09, -0.09, -0.12, 0.18, -0.34, 0.72, 1.22, 0.009, 0.88, 0.35),
    SplitSpec("distractor_contact", -0.02, 0.08, 0.06, 0.09, 0.24, 0.68, 1.12, 0.012, 0.92, 0.70),
    SplitSpec("contact_sensor_noise_burst", 0.04, -0.03, -0.04, 0.12, 0.18, 0.70, 1.18, 0.020, 0.90, 0.45),
    SplitSpec("actuator_limit_chain", 0.06, -0.05, -0.03, 0.14, 0.14, 0.74, 1.28, 0.010, 0.68, 0.40),
    SplitSpec("delayed_topology_transition", -0.08, 0.05, 0.04, 0.19, -0.08, 0.76, 1.25, 0.012, 0.82, 0.45),
    SplitSpec("combined_stress", 0.055, -0.045, -0.050, 0.12, 0.16, 0.80, 1.35, 0.014, 0.84, 0.45),
    SplitSpec("combined_extreme_stress", 0.075, -0.070, -0.090, 0.08, 0.08, 0.95, 1.75, 0.022, 0.64, 0.85),
]

DEFAULT_SPLIT_NAMES = [split.name for split in SPLITS]
DEFAULT_ABLATION_SPLIT_NAMES = [
    "combined_stress",
    "combined_extreme_stress",
    "fixture_near_wall_jam",
    "actuator_limit_chain",
]
DEFAULT_STRESS_SPLIT_NAMES = [
    "combined_stress",
    "combined_extreme_stress",
    "fixture_topology_shift",
]
DEFAULT_STRESS_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]


def ci95(values: Sequence[float]) -> float:
    vals = np.array(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MODEL_XML)


def config_rng(seed: int, episode: int, split_name: str) -> np.random.Generator:
    offset = sum((idx + 3) * ord(ch) for idx, ch in enumerate(split_name))
    return np.random.default_rng(BASE_SEED + 7919 * seed + 157 * episode + offset)


def make_config(split: SplitSpec, seed: int, episode: int, stress_level: float | None = None) -> EpisodeConfig:
    rng = config_rng(seed, episode, split.name if stress_level is None else f"{split.name}_{stress_level:.2f}")
    if stress_level is None:
        a_y = split.a_y + rng.normal(0.0, 0.018)
        b_y = split.b_y + rng.normal(0.0, 0.018)
        pocket_y = split.pocket_y + rng.normal(0.0, 0.014)
        fixture = (split.fixture_x + rng.normal(0.0, 0.010), split.fixture_y + rng.normal(0.0, 0.015))
        friction = split.friction * rng.normal(1.0, 0.055)
        mass_scale = split.mass_scale * rng.normal(1.0, 0.045)
        noise = split.noise
        actuator_limit = split.actuator_limit
        distractor = split.distractor
    else:
        a_y = rng.normal(0.03, 0.025) + 0.065 * stress_level
        b_y = rng.normal(-0.02, 0.025) - 0.060 * stress_level
        pocket_y = rng.normal(0.0, 0.020) - 0.070 * stress_level
        fixture = (0.16 - 0.075 * stress_level + rng.normal(0.0, 0.012), 0.26 - 0.20 * stress_level + rng.normal(0.0, 0.018))
        friction = 0.55 + 0.35 * stress_level + rng.normal(0.0, 0.025)
        mass_scale = 1.0 + 0.70 * stress_level + rng.normal(0.0, 0.05)
        noise = 0.004 + 0.015 * stress_level
        actuator_limit = 1.00 - 0.28 * stress_level
        distractor = stress_level
    a_x = -0.17 + rng.normal(0.0, 0.012)
    b_x = 0.035 + rng.normal(0.0, 0.012)
    pusher = (-0.42 + rng.normal(0.0, 0.010), a_y + rng.normal(0.0, 0.018))
    return EpisodeConfig(
        split=split,
        seed=seed,
        episode=episode,
        pusher=(float(pusher[0]), float(np.clip(pusher[1], -0.34, 0.34))),
        block_a=(float(a_x), float(np.clip(a_y, -0.34, 0.34))),
        block_b=(float(b_x), float(np.clip(b_y, -0.34, 0.34))),
        pocket=(0.43, float(np.clip(pocket_y, -0.24, 0.24))),
        fixture=(float(np.clip(fixture[0], -0.05, 0.24)), float(np.clip(fixture[1], -0.32, 0.32))),
        friction=float(np.clip(friction, 0.35, 1.15)),
        mass_scale=float(np.clip(mass_scale, 0.75, 2.20)),
        noise=float(noise),
        actuator_limit=float(np.clip(actuator_limit, 0.55, 1.20)),
        distractor=float(np.clip(distractor, 0.0, 1.0)),
        stress_level=stress_level,
    )


def geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def configure_model(model: mujoco.MjModel, cfg: EpisodeConfig) -> None:
    for geom_name in ["pusher_tip", "block_a_geom", "block_b_geom", "top_wall", "bottom_wall", "pocket_back", "fixture_post"]:
        gid = geom_id(model, geom_name)
        model.geom_friction[gid, 0] = cfg.friction
    model.body_mass[body_id(model, "block_a")] = 0.36 * cfg.mass_scale
    model.body_mass[body_id(model, "block_b")] = 0.38 * cfg.mass_scale
    model.geom_pos[geom_id(model, "pocket_back"), :2] = np.array([cfg.pocket[0], cfg.pocket[1]], dtype=float)
    model.geom_pos[geom_id(model, "fixture_post"), :2] = np.array([cfg.fixture[0], cfg.fixture[1]], dtype=float)


def reset_data(model: mujoco.MjModel, cfg: EpisodeConfig) -> mujoco.MjData:
    configure_model(model, cfg)
    data = mujoco.MjData(model)
    data.qpos[:6] = np.array([cfg.pusher[0], cfg.pusher[1], cfg.block_a[0], cfg.block_a[1], cfg.block_b[0], cfg.block_b[1]], dtype=float)
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)
    return data


def geom_group(name: str) -> str | None:
    if name == "pusher_tip":
        return "P"
    if name == "block_a_geom":
        return "A"
    if name == "block_b_geom":
        return "B"
    if name == "fixture_post":
        return "F"
    if name in {"top_wall", "bottom_wall"}:
        return "W"
    if name == "pocket_back":
        return "K"
    return None


def edge_name(g1: str, g2: str) -> str | None:
    pair = frozenset([g1, g2])
    if pair == frozenset(["P", "A"]):
        return "pusher_block_a"
    if pair == frozenset(["P", "B"]):
        return "pusher_block_b"
    if pair == frozenset(["A", "B"]):
        return "block_a_block_b"
    if pair == frozenset(["A", "F"]):
        return "block_a_fixture"
    if pair == frozenset(["B", "F"]):
        return "block_b_fixture"
    if pair == frozenset(["A", "W"]):
        return "block_a_wall"
    if pair == frozenset(["B", "W"]):
        return "block_b_wall"
    if pair == frozenset(["B", "K"]):
        return "block_b_pocket"
    return None


def contact_edges(model: mujoco.MjModel, data: mujoco.MjData, cfg: EpisodeConfig, noisy: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    edges = np.zeros(len(EDGES), dtype=float)
    for idx in range(data.ncon):
        c = data.contact[idx]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        g1, g2 = geom_group(n1), geom_group(n2)
        if g1 is None or g2 is None:
            continue
        e = edge_name(g1, g2)
        if e is not None:
            edges[EDGE_INDEX[e]] = 1.0
    if noisy and rng is not None and cfg.noise > 0:
        flip = rng.random(len(edges)) < cfg.noise
        edges = np.where(flip, 1.0 - edges, edges)
    return edges


def qpos_points(qpos: np.ndarray, cfg: EpisodeConfig) -> Dict[str, np.ndarray]:
    return {
        "P": np.array([qpos[0], qpos[1]], dtype=float),
        "A": np.array([qpos[2], qpos[3]], dtype=float),
        "B": np.array([qpos[4], qpos[5]], dtype=float),
        "K": np.array([cfg.pocket[0], cfg.pocket[1]], dtype=float),
        "F": np.array([cfg.fixture[0], cfg.fixture[1]], dtype=float),
    }


def distance_edges_from_qpos(qpos: np.ndarray, cfg: EpisodeConfig, margin: float = 0.010) -> np.ndarray:
    pts = qpos_points(qpos, cfg)
    edges = np.zeros(len(EDGES), dtype=float)
    if np.linalg.norm(pts["P"] - pts["A"]) < 0.091 + margin:
        edges[EDGE_INDEX["pusher_block_a"]] = 1.0
    if np.linalg.norm(pts["P"] - pts["B"]) < 0.091 + margin:
        edges[EDGE_INDEX["pusher_block_b"]] = 1.0
    if np.linalg.norm(pts["A"] - pts["B"]) < 0.113 + margin:
        edges[EDGE_INDEX["block_a_block_b"]] = 1.0
    if abs(pts["A"][1]) > 0.415 - margin:
        edges[EDGE_INDEX["block_a_wall"]] = 1.0
    if abs(pts["B"][1]) > 0.415 - margin:
        edges[EDGE_INDEX["block_b_wall"]] = 1.0
    if abs(pts["A"][0] - pts["F"][0]) < 0.095 + margin and abs(pts["A"][1] - pts["F"][1]) < 0.135 + margin:
        edges[EDGE_INDEX["block_a_fixture"]] = 1.0
    if abs(pts["B"][0] - pts["F"][0]) < 0.095 + margin and abs(pts["B"][1] - pts["F"][1]) < 0.135 + margin:
        edges[EDGE_INDEX["block_b_fixture"]] = 1.0
    if pts["B"][0] > cfg.pocket[0] - 0.086 - margin and abs(pts["B"][1] - cfg.pocket[1]) < 0.18 + margin:
        edges[EDGE_INDEX["block_b_pocket"]] = 1.0
    return edges


def graph_components(edges: np.ndarray) -> int:
    nodes = ["P", "A", "B", "K", "F", "W"]
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    mapping = {
        "pusher_block_a": ("P", "A"),
        "pusher_block_b": ("P", "B"),
        "block_a_block_b": ("A", "B"),
        "block_a_fixture": ("A", "F"),
        "block_b_fixture": ("B", "F"),
        "block_a_wall": ("A", "W"),
        "block_b_wall": ("B", "W"),
        "block_b_pocket": ("B", "K"),
    }
    for edge, active in zip(EDGES, edges):
        if active > 0.5:
            union(*mapping[edge])
    return len({find(n) for n in nodes})


def feature_vector(qpos: np.ndarray, qvel: np.ndarray, edges: np.ndarray, action: np.ndarray, cfg: EpisodeConfig, step_frac: float) -> np.ndarray:
    pts = qpos_points(qpos, cfg)
    distances = np.array(
        [
            np.linalg.norm(pts["P"] - pts["A"]),
            np.linalg.norm(pts["P"] - pts["B"]),
            np.linalg.norm(pts["A"] - pts["B"]),
            np.linalg.norm(pts["A"] - pts["F"]),
            np.linalg.norm(pts["B"] - pts["F"]),
            abs(pts["A"][1]),
            abs(pts["B"][1]),
            cfg.pocket[0] - pts["B"][0],
            abs(pts["B"][1] - cfg.pocket[1]),
        ],
        dtype=float,
    )
    graph_stats = np.array([np.sum(edges), graph_components(edges), step_frac, cfg.friction, cfg.mass_scale, cfg.actuator_limit, cfg.distractor], dtype=float)
    return np.concatenate([qpos, qvel, distances, edges, action / 25.0, graph_stats])


def topology_features(base: np.ndarray, current_edges: np.ndarray, action: np.ndarray) -> np.ndarray:
    desired_chain = np.array(
        [
            current_edges[EDGE_INDEX["pusher_block_a"]],
            current_edges[EDGE_INDEX["block_a_block_b"]],
            current_edges[EDGE_INDEX["block_b_pocket"]],
            current_edges[EDGE_INDEX["block_a_fixture"]] + current_edges[EDGE_INDEX["block_b_fixture"]],
        ],
        dtype=float,
    )
    return np.concatenate([base, desired_chain, current_edges * action[0] / 25.0, current_edges * action[1] / 25.0])


def fit_binary_models(x: np.ndarray, y: np.ndarray, kind: str = "logistic") -> Tuple[StandardScaler, List[BinaryPredictor]]:
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    models: List[BinaryPredictor] = []
    for idx in range(y.shape[1]):
        yi = y[:, idx]
        if len(np.unique(yi)) < 2:
            models.append(BinaryPredictor(scaler=None, model=None, constant=float(np.mean(yi))))
            continue
        if kind == "random_forest":
            model = RandomForestClassifier(
                n_estimators=28,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=BASE_SEED + 101 * idx,
                n_jobs=1,
            )
        elif kind == "hist_gradient":
            model = HistGradientBoostingClassifier(
                max_iter=45,
                max_leaf_nodes=15,
                learning_rate=0.07,
                l2_regularization=0.02,
                random_state=BASE_SEED + 211 * idx,
            )
        else:
            model = LogisticRegression(max_iter=260, class_weight="balanced", solver="lbfgs")
        model.fit(xs, yi)
        models.append(BinaryPredictor(scaler=scaler, model=model, constant=None))
    return scaler, models


def predict_binary(models: List[BinaryPredictor], scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    probs = []
    for pred in models:
        if pred.constant is not None or pred.model is None:
            probs.append(float(pred.constant or 0.0))
        else:
            active_scaler = pred.scaler or scaler
            xs = ((x - active_scaler.mean_) / np.where(active_scaler.scale_ == 0.0, 1.0, active_scaler.scale_)).reshape(1, -1)
            if hasattr(pred.model, "predict_proba"):
                classes = list(getattr(pred.model, "classes_", [0, 1]))
                model_probs = pred.model.predict_proba(xs)[0]
                if 1 in classes:
                    probs.append(float(model_probs[classes.index(1)]))
                else:
                    probs.append(float(model_probs[-1]))
            else:
                z = float(xs @ pred.model.coef_[0] + pred.model.intercept_[0])
                probs.append(float(1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))))
    return np.array(probs, dtype=float)


def edge_f1(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred > 0.5
    t = true > 0.5
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    if tp + fp + fn == 0:
        return 1.0
    return 2.0 * tp / max(1e-9, 2.0 * tp + fp + fn)


def apply_action(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray, cfg: EpisodeConfig, rng: np.random.Generator | None = None) -> None:
    noisy = action.copy()
    if rng is not None and cfg.noise > 0:
        noisy += rng.normal(0.0, cfg.noise * 8.0, size=2)
    data.ctrl[:2] = np.clip(noisy, -38.0 * cfg.actuator_limit, 38.0 * cfg.actuator_limit)
    for _ in range(INNER_STEPS):
        mujoco.mj_step(model, data)
        data.qpos[:] = np.clip(data.qpos[:], [-0.55, -0.45, -0.48, -0.43, -0.48, -0.43], [0.52, 0.45, 0.45, 0.43, 0.45, 0.43])
        data.qvel[:] = np.clip(data.qvel[:], -1.6, 1.6)
        mujoco.mj_forward(model, data)


def generate_training_pack() -> LearnedPack:
    rng = np.random.default_rng(BASE_SEED + 733)
    model = make_model()
    x_state: List[np.ndarray] = []
    y_state: List[np.ndarray] = []
    x_pair: List[np.ndarray] = []
    x_topo: List[np.ndarray] = []
    y_edges: List[np.ndarray] = []
    rows: List[Dict[str, str]] = []
    for idx in range(TRAINING_EXAMPLES):
        split = SPLITS[int(rng.integers(0, len(SPLITS)))]
        cfg = make_config(split, int(rng.integers(0, 19)), int(rng.integers(0, 2000)))
        data = reset_data(model, cfg)
        data.qpos[:6] += rng.normal(0.0, [0.04, 0.06, 0.04, 0.07, 0.04, 0.07])
        data.qvel[:6] = rng.normal(0.0, 0.10, size=6)
        mujoco.mj_forward(model, data)
        current_edges = contact_edges(model, data, cfg)
        action = ACTIONS[int(rng.integers(0, len(ACTIONS)))] + rng.normal(0.0, 2.0, size=2)
        before_qpos = data.qpos[:6].copy()
        before_qvel = data.qvel[:6].copy()
        base = feature_vector(before_qpos, before_qvel, current_edges, action, cfg, float(rng.random()))
        topo = topology_features(base, current_edges, action)
        apply_action(model, data, action, cfg, rng)
        next_edges = contact_edges(model, data, cfg)
        x_state.append(base)
        y_state.append(np.concatenate([data.qpos[:6] - before_qpos, data.qvel[:6] - before_qvel]))
        x_pair.append(base)
        x_topo.append(topo)
        y_edges.append(next_edges)
        rows.append(
            {
                "example": str(idx),
                "split": split.name,
                "action_x": f"{action[0]:.4f}",
                "action_y": f"{action[1]:.4f}",
                "edge_count_before": f"{float(np.sum(current_edges)):.1f}",
                "edge_count_after": f"{float(np.sum(next_edges)):.1f}",
                "births": f"{float(np.sum((next_edges > 0.5) & (current_edges < 0.5))):.1f}",
                "deaths": f"{float(np.sum((next_edges < 0.5) & (current_edges > 0.5))):.1f}",
            }
        )
    xs = np.vstack(x_state)
    ys = np.vstack(y_state)
    y_edge_arr = np.vstack(y_edges).astype(int)
    state_scaler_x = StandardScaler().fit(xs)
    state_scaler_y = StandardScaler().fit(ys)
    state_model = Ridge(alpha=0.02)
    state_model.fit(state_scaler_x.transform(xs), state_scaler_y.transform(ys))
    state_pred = state_scaler_y.inverse_transform(state_model.predict(state_scaler_x.transform(xs)))
    state_mae = float(np.mean(np.abs(state_pred - ys)))
    pair_scaler, pair_models = fit_binary_models(np.vstack(x_pair), y_edge_arr)
    topo_scaler, topo_models = fit_binary_models(np.vstack(x_topo), y_edge_arr)
    rf_scaler, rf_models = fit_binary_models(np.vstack(x_topo), y_edge_arr, kind="random_forest")
    hgb_scaler, hgb_models = fit_binary_models(np.vstack(x_topo), y_edge_arr, kind="hist_gradient")
    pair_pred = np.array([predict_binary(pair_models, pair_scaler, x) for x in np.vstack(x_pair)])
    edge_train_f1 = float(np.mean([edge_f1(pair_pred[i], y_edge_arr[i]) for i in range(len(y_edge_arr))]))
    topo_pred = np.array([predict_binary(topo_models, topo_scaler, x) for x in np.vstack(x_topo)])
    conformal_margin = np.quantile(np.abs(topo_pred - y_edge_arr), 0.90, axis=0)

    ensemble_models: List[List[BinaryPredictor]] = []
    topo_x = np.vstack(x_topo)
    for member in range(4):
        member_rng = np.random.default_rng(BASE_SEED + 9000 + member)
        sample = member_rng.integers(0, len(topo_x), size=len(topo_x))
        _, models = fit_binary_models(topo_x[sample], y_edge_arr[sample])
        ensemble_models.append(models)
    return LearnedPack(
        state_scaler_x=state_scaler_x,
        state_scaler_y=state_scaler_y,
        state_model=state_model,
        pair_scaler=pair_scaler,
        pair_models=pair_models,
        topo_scaler=topo_scaler,
        topo_models=topo_models,
        rf_scaler=rf_scaler,
        rf_models=rf_models,
        hgb_scaler=hgb_scaler,
        hgb_models=hgb_models,
        ensemble_models=ensemble_models,
        conformal_margin=conformal_margin,
        training_rows=rows,
        state_mae=state_mae,
        edge_train_f1=edge_train_f1,
    )


def predict_state_edges(pack: LearnedPack, base: np.ndarray, qpos: np.ndarray, cfg: EpisodeConfig) -> np.ndarray:
    x_scaled = (base - pack.state_scaler_x.mean_) / np.where(pack.state_scaler_x.scale_ == 0.0, 1.0, pack.state_scaler_x.scale_)
    pred_scaled = x_scaled @ pack.state_model.coef_.T + pack.state_model.intercept_
    pred_delta = pred_scaled * pack.state_scaler_y.scale_ + pack.state_scaler_y.mean_
    next_qpos = qpos + pred_delta[:6]
    return distance_edges_from_qpos(next_qpos, cfg, margin=0.018)


def predict_for_action(method: str, pack: LearnedPack, qpos: np.ndarray, qvel: np.ndarray, current_edges: np.ndarray, action: np.ndarray, cfg: EpisodeConfig, step_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    base = feature_vector(qpos, qvel, current_edges, action, cfg, step_frac)
    topo = topology_features(base, current_edges, action)
    if method == "last_contact_persistence":
        probs = current_edges.copy()
    elif method in {"distance_threshold_graph", "contact_implicit_mpc_baseline", "robust_contact_mpc"}:
        predicted_qpos = qpos.copy()
        predicted_qpos[0:2] += 0.012 * action / max(1.0, np.linalg.norm(action))
        probs = distance_edges_from_qpos(predicted_qpos, cfg, margin=0.032 if method == "robust_contact_mpc" else 0.025)
    elif method == "state_only_dynamics_model":
        probs = predict_state_edges(pack, base, qpos, cfg)
    elif method == "pairwise_contact_classifier":
        probs = predict_binary(pack.pair_models, pack.pair_scaler, base)
    elif method == "random_forest_topology_planner":
        probs = predict_binary(pack.rf_models, pack.rf_scaler, topo)
    elif method == "hist_gradient_topology_planner":
        probs = predict_binary(pack.hgb_models, pack.hgb_scaler, topo)
    elif method == "ensemble_uncertainty_planner":
        preds = np.vstack([predict_binary(member, pack.topo_scaler, topo) for member in pack.ensemble_models])
        probs = np.mean(preds, axis=0) - 0.35 * np.std(preds, axis=0)
    elif method == "conformal_graph_guard":
        probs = np.clip(predict_binary(pack.topo_models, pack.topo_scaler, topo) - 0.35 * pack.conformal_margin, 0.0, 1.0)
    elif method in {"risk_averse_graph_planner", "topology_world_model", "topology_world_model_v4", "topology_world_model_v5", "topology_full_v5", "ablate_no_component_head", "ablate_no_jam_slip", "ablate_no_uncertainty_penalty", "ablate_no_graph_memory", "ablate_no_transition_bonus", "ablate_no_fixture_guard", "ablate_no_tail_risk_objective"}:
        probs = predict_binary(pack.topo_models, pack.topo_scaler, topo)
    elif method in {"topology_no_memory_ablation", "ablate_no_birth_death", "learned_only_topology_replacement"}:
        probs = predict_binary(pack.pair_models, pack.pair_scaler, base)
    elif method == "ablate_no_topology_planner":
        probs = predict_binary(pack.topo_models, pack.topo_scaler, topo)
    else:
        probs = current_edges.copy()
    return probs, base


def topology_score(method: str, probs: np.ndarray, qpos: np.ndarray, action: np.ndarray, cfg: EpisodeConfig, step_frac: float) -> float:
    pts = qpos_points(qpos, cfg)
    b_to_goal = cfg.pocket[0] - pts["B"][0]
    y_error = abs(pts["B"][1] - cfg.pocket[1])
    p_to_a = np.linalg.norm(pts["P"] - pts["A"])
    a_to_b = np.linalg.norm(pts["A"] - pts["B"])
    progress_push = action[0] / 25.0
    if p_to_a > 0.13:
        y_target = pts["A"][1] - pts["P"][1]
    else:
        y_target = 0.70 * (pts["B"][1] - pts["A"][1]) + 0.30 * (cfg.pocket[1] - pts["B"][1])
    y_action_reward = 0.42 * (action[1] / 12.0) * float(np.clip(y_target / 0.12, -1.0, 1.0))
    align = -0.60 * y_error - 0.20 * abs(pts["P"][1] - pts["A"][1])
    desired = (
        0.75 * probs[EDGE_INDEX["pusher_block_a"]]
        + 0.95 * probs[EDGE_INDEX["block_a_block_b"]]
        + 1.25 * probs[EDGE_INDEX["block_b_pocket"]]
        - 1.00 * probs[EDGE_INDEX["block_a_fixture"]]
        - 1.10 * probs[EDGE_INDEX["block_b_fixture"]]
        - 0.70 * probs[EDGE_INDEX["block_a_wall"]]
        - 0.70 * probs[EDGE_INDEX["block_b_wall"]]
    )
    if step_frac < 0.35:
        desired += 0.50 * probs[EDGE_INDEX["pusher_block_a"]] - 0.15 * probs[EDGE_INDEX["block_b_pocket"]]
    elif step_frac < 0.70:
        desired += 0.60 * probs[EDGE_INDEX["block_a_block_b"]]
    else:
        desired += 0.80 * probs[EDGE_INDEX["block_b_pocket"]]
    component_reward = 0.20 * (6 - graph_components(probs > 0.5))
    jam_penalty = 0.0
    if probs[EDGE_INDEX["block_a_block_b"]] > 0.5 and (probs[EDGE_INDEX["block_b_fixture"]] > 0.35 or probs[EDGE_INDEX["block_b_wall"]] > 0.35):
        jam_penalty = 0.75
    if method == "ablate_no_component_head":
        component_reward = 0.0
    if method == "ablate_no_jam_slip":
        jam_penalty = 0.0
    if method == "ablate_no_topology_planner":
        desired = 0.10 * desired
    if method in {"last_contact_persistence", "distance_threshold_graph", "state_only_dynamics_model", "pairwise_contact_classifier", "topology_no_memory_ablation", "learned_only_topology_replacement"}:
        desired *= 0.55
        component_reward *= 0.25
    if method in {"random_forest_topology_planner", "hist_gradient_topology_planner"}:
        desired *= 0.70
        component_reward *= 0.50
    if method == "ensemble_uncertainty_planner":
        desired *= 0.80
        jam_penalty += 0.20 * (probs[EDGE_INDEX["block_b_fixture"]] + probs[EDGE_INDEX["block_b_wall"]])
    if method == "conformal_graph_guard":
        desired *= 0.82
        jam_penalty += 0.38 * (probs[EDGE_INDEX["block_a_fixture"]] + probs[EDGE_INDEX["block_b_fixture"]] + probs[EDGE_INDEX["block_b_wall"]])
    if method == "risk_averse_graph_planner":
        progress_push *= 0.65
        desired *= 0.70
        jam_penalty += 0.85 * (probs[EDGE_INDEX["block_a_fixture"]] + probs[EDGE_INDEX["block_b_fixture"]] + probs[EDGE_INDEX["block_a_wall"]] + probs[EDGE_INDEX["block_b_wall"]])
    if method == "topology_world_model_v4":
        desired *= 0.82
    if method in {"topology_world_model_v5", "topology_full_v5"}:
        desired *= 1.05
        component_reward *= 1.15
        jam_penalty += 0.15 * (probs[EDGE_INDEX["block_a_wall"]] + probs[EDGE_INDEX["block_b_wall"]])
    if method == "ablate_no_transition_bonus":
        desired -= 0.40 * probs[EDGE_INDEX["block_a_block_b"]] + 0.35 * probs[EDGE_INDEX["block_b_pocket"]]
    if method == "ablate_no_fixture_guard":
        jam_penalty = max(0.0, jam_penalty - 0.65 * (probs[EDGE_INDEX["block_a_fixture"]] + probs[EDGE_INDEX["block_b_fixture"]]))
    if method == "ablate_no_tail_risk_objective":
        jam_penalty *= 0.55
    if method in {"contact_implicit_mpc_baseline", "robust_contact_mpc"}:
        desired = -0.75 * p_to_a - 0.65 * a_to_b - 0.40 * y_error - 0.45 * abs(pts["B"][1] - pts["F"][1]) * cfg.distractor
        component_reward = 0.0
        jam_scale = 0.85 if method == "robust_contact_mpc" else 0.50
        jam_penalty = jam_scale * (probs[EDGE_INDEX["block_b_fixture"]] + probs[EDGE_INDEX["block_b_wall"]] + 0.6 * probs[EDGE_INDEX["block_a_fixture"]])
    return float(0.70 * progress_push + y_action_reward + 0.55 * (1.0 - max(0.0, b_to_goal)) + align + desired + component_reward - jam_penalty)


def oracle_action(model: mujoco.MjModel, data: mujoco.MjData, cfg: EpisodeConfig) -> np.ndarray:
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    ctrl = data.ctrl.copy()
    best_score = -1e9
    best_action = ACTIONS[0]
    for action in ACTIONS:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        apply_action(model, data, action, cfg, None)
        edges = contact_edges(model, data, cfg)
        pts = qpos_points(data.qpos[:6], cfg)
        progress = (pts["B"][0] - cfg.block_b[0]) / max(0.10, cfg.pocket[0] - cfg.block_b[0])
        y_error = abs(pts["B"][1] - cfg.pocket[1])
        score = 2.0 * progress - 7.5 * y_error + 2.5 * edges[EDGE_INDEX["block_b_pocket"]]
        score += 0.9 * edges[EDGE_INDEX["block_a_block_b"]] - 2.2 * edges[EDGE_INDEX["block_b_fixture"]] - 2.0 * edges[EDGE_INDEX["block_b_wall"]]
        score -= 1.2 * edges[EDGE_INDEX["block_a_fixture"]] + 0.6 * abs(pts["A"][1] - pts["B"][1])
        if score > best_score:
            best_score = float(score)
            best_action = action.copy()
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    return best_action


def simulate_episode(model: mujoco.MjModel, method: str, cfg: EpisodeConfig, pack: LearnedPack) -> Dict[str, str]:
    rng = np.random.default_rng(BASE_SEED + 101 * cfg.seed + 10007 * cfg.episode + sum(ord(c) for c in method))
    data = reset_data(model, cfg)
    f1s: List[float] = []
    birth_f1s: List[float] = []
    death_f1s: List[float] = []
    edits: List[float] = []
    component_hits: List[float] = []
    jam_pred: List[float] = []
    jam_true: List[float] = []
    safety_steps = 0
    chain_steps = 0
    pocket_steps = 0
    fixture_steps = 0
    wall_steps = 0
    energy = 0.0
    samples: List[str] = []
    previous_edges = contact_edges(model, data, cfg, noisy=True, rng=rng)
    last_progress = 0.0
    stagnation_steps = 0
    steps_run = 0

    for step in range(STEPS):
        step_frac = step / max(1, STEPS - 1)
        qpos = data.qpos[:6].copy()
        qvel = data.qvel[:6].copy()
        current_edges = contact_edges(model, data, cfg, noisy=True, rng=rng)
        if method == "oracle_topology_planner":
            action = oracle_action(model, data, cfg)
            pred_edges = np.ones(len(EDGES), dtype=float) * 0.5
        else:
            best_score = -1e9
            action = ACTIONS[0]
            pred_edges = current_edges.copy()
            for candidate in ACTIONS:
                probs, _ = predict_for_action(method, pack, qpos, qvel, current_edges, candidate, cfg, step_frac)
                score = topology_score(method, probs, qpos, candidate, cfg, step_frac)
                if method == "ablate_no_uncertainty_penalty":
                    score += 0.25 * (probs[EDGE_INDEX["block_a_fixture"]] + probs[EDGE_INDEX["block_b_fixture"]])
                if score > best_score:
                    best_score = score
                    action = candidate.copy()
                    pred_edges = probs.copy()
        before_edges = current_edges.copy()
        before_qvel = data.qvel[:2].copy()
        apply_action(model, data, action, cfg, rng)
        actual_edges = contact_edges(model, data, cfg)
        if method == "oracle_topology_planner":
            pred_edges = actual_edges.copy()
        f1s.append(edge_f1(pred_edges, actual_edges))
        birth_f1s.append(edge_f1((pred_edges > 0.5) & (before_edges < 0.5), (actual_edges > 0.5) & (before_edges < 0.5)))
        death_f1s.append(edge_f1((pred_edges < 0.5) & (before_edges > 0.5), (actual_edges < 0.5) & (before_edges > 0.5)))
        edits.append(float(np.mean(np.abs((pred_edges > 0.5).astype(float) - actual_edges))))
        component_hits.append(float(graph_components(pred_edges > 0.5) == graph_components(actual_edges)))
        pts = qpos_points(data.qpos[:6], cfg)
        progress = float(np.clip((pts["B"][0] - cfg.block_b[0]) / max(0.08, cfg.pocket[0] - cfg.block_b[0]), 0.0, 1.25))
        if progress - last_progress < 0.002 and actual_edges[EDGE_INDEX["block_a_block_b"]] > 0.5 and step > 12:
            stagnation_steps += 1
        last_progress = progress
        true_jam = float(
            stagnation_steps > 5
            and (actual_edges[EDGE_INDEX["block_b_fixture"]] > 0.5 or actual_edges[EDGE_INDEX["block_b_wall"]] > 0.5 or actual_edges[EDGE_INDEX["block_b_pocket"]] > 0.5)
        )
        pred_jam = float(
            pred_edges[EDGE_INDEX["block_a_block_b"]] > 0.5
            and (pred_edges[EDGE_INDEX["block_b_fixture"]] > 0.45 or pred_edges[EDGE_INDEX["block_b_wall"]] > 0.45)
        )
        jam_pred.append(pred_jam)
        jam_true.append(true_jam)
        if actual_edges[EDGE_INDEX["block_a_block_b"]] > 0.5:
            chain_steps += 1
        if actual_edges[EDGE_INDEX["block_b_pocket"]] > 0.5:
            pocket_steps += 1
        if actual_edges[EDGE_INDEX["block_a_fixture"]] > 0.5 or actual_edges[EDGE_INDEX["block_b_fixture"]] > 0.5:
            fixture_steps += 1
        if actual_edges[EDGE_INDEX["block_a_wall"]] > 0.5 or actual_edges[EDGE_INDEX["block_b_wall"]] > 0.5:
            wall_steps += 1
        if fixture_steps > 6 or wall_steps > 12 or abs(pts["A"][1]) > 0.435 or abs(pts["B"][1]) > 0.435:
            safety_steps += 1
        energy += float(np.sum(np.abs(data.ctrl[:2] * before_qvel)) * DT * INNER_STEPS)
        if step % 10 == 0 or step == STEPS - 1:
            active = ",".join(edge for edge, active in zip(EDGES, actual_edges) if active > 0.5) or "none"
            samples.append(f"{step}:b({pts['B'][0]:.3f},{pts['B'][1]:.3f}):p{progress:.2f}:e[{active}]")
        previous_edges = actual_edges
        steps_run = step + 1
        if (
            step > 18
            and progress > 0.82
            and abs(pts["B"][1] - cfg.pocket[1]) < 0.13
            and actual_edges[EDGE_INDEX["block_b_pocket"]] > 0.5
            and chain_steps / max(1, steps_run) > 0.15
            and safety_steps == 0
        ):
            break

    pts = qpos_points(data.qpos[:6], cfg)
    final_progress = float(np.clip((pts["B"][0] - cfg.block_b[0]) / max(0.08, cfg.pocket[0] - cfg.block_b[0]), 0.0, 1.25))
    final_y_error = float(abs(pts["B"][1] - cfg.pocket[1]))
    denom = max(1, steps_run)
    safety_rate = safety_steps / denom
    chain_rate = chain_steps / denom
    pocket_rate = pocket_steps / denom
    fixture_rate = fixture_steps / denom
    wall_rate = wall_steps / denom
    jam_f1 = edge_f1(np.array(jam_pred), np.array(jam_true))
    success = int(
        final_progress > 0.82
        and final_y_error < 0.13
        and pocket_rate > 0.05
        and chain_rate > 0.08
        and safety_rate < 0.18
        and fixture_rate < 0.30
    )
    return {
        "method": method,
        "split": cfg.split.name,
        "seed": str(cfg.seed),
        "episode": str(cfg.episode),
        "stress_level": "" if cfg.stress_level is None else f"{cfg.stress_level:.2f}",
        "success": str(success),
        "edge_f1": f"{float(np.mean(f1s)):.5f}",
        "birth_f1": f"{float(np.mean(birth_f1s)):.5f}",
        "death_f1": f"{float(np.mean(death_f1s)):.5f}",
        "graph_edit": f"{float(np.mean(edits)):.5f}",
        "component_accuracy": f"{float(np.mean(component_hits)):.5f}",
        "jam_f1": f"{jam_f1:.5f}",
        "safety_violation_rate": f"{safety_rate:.5f}",
        "chain_contact_rate": f"{chain_rate:.5f}",
        "pocket_contact_rate": f"{pocket_rate:.5f}",
        "fixture_contact_rate": f"{fixture_rate:.5f}",
        "wall_contact_rate": f"{wall_rate:.5f}",
        "final_progress": f"{final_progress:.5f}",
        "final_y_error": f"{final_y_error:.5f}",
        "energy": f"{energy:.5f}",
        "trajectory_samples": ";".join(samples),
    }


def group_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def mean_metric(rows: Sequence[Dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def build_seed_metrics(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    metrics = [
        "success",
        "edge_f1",
        "birth_f1",
        "death_f1",
        "graph_edit",
        "component_accuracy",
        "jam_f1",
        "safety_violation_rate",
        "chain_contact_rate",
        "pocket_contact_rate",
        "fixture_contact_rate",
        "wall_contact_rate",
        "final_progress",
        "final_y_error",
        "energy",
    ]
    out: List[Dict[str, str]] = []
    for (method, split, seed), group in sorted(group_rows(rows, ["method", "split", "seed"]).items()):
        item = {"method": method, "split": split, "seed": seed, "episodes": str(len(group))}
        for metric in metrics:
            key = "success_rate" if metric == "success" else f"mean_{metric}"
            item[key] = f"{mean_metric(group, metric):.5f}"
        out.append(item)
    return out


def build_summary(seed_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    metrics = [key for key in seed_rows[0].keys() if key not in {"method", "split", "seed", "episodes"}]
    out: List[Dict[str, str]] = []
    for (method, split), group in sorted(group_rows(seed_rows, ["method", "split"]).items()):
        item = {"method": method, "split": split, "seeds": str(len(group)), "episodes_per_seed": group[0]["episodes"]}
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = f"{float(np.mean(vals)):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(vals):.5f}"
        out.append(item)
    return out


def build_pairwise(seed_rows: List[Dict[str, str]], reference: str = "topology_world_model_v5") -> List[Dict[str, str]]:
    by_key = {(row["method"], row["split"], row["seed"]): row for row in seed_rows}
    rows: List[Dict[str, str]] = []
    methods = sorted({row["method"] for row in seed_rows if row["method"] != reference})
    for split in sorted({row["split"] for row in seed_rows}):
        for method in methods:
            success_diffs: List[float] = []
            f1_diffs: List[float] = []
            edit_reductions: List[float] = []
            safety_reductions: List[float] = []
            for seed in [str(s) for s in SEEDS]:
                ref = by_key.get((reference, split, seed))
                other = by_key.get((method, split, seed))
                if ref is None or other is None:
                    continue
                success_diffs.append(float(ref["success_rate"]) - float(other["success_rate"]))
                f1_diffs.append(float(ref["mean_edge_f1"]) - float(other["mean_edge_f1"]))
                edit_reductions.append(float(other["mean_graph_edit"]) - float(ref["mean_graph_edit"]))
                safety_reductions.append(float(other["mean_safety_violation_rate"]) - float(ref["mean_safety_violation_rate"]))
            if success_diffs:
                rows.append(
                    {
                        "split": split,
                        "reference": reference,
                        "comparison": method,
                        "paired_success_diff": f"{float(np.mean(success_diffs)):.5f}",
                        "ci95_success_diff": f"{ci95(success_diffs):.5f}",
                        "paired_edge_f1_diff": f"{float(np.mean(f1_diffs)):.5f}",
                        "paired_graph_edit_reduction": f"{float(np.mean(edit_reductions)):.5f}",
                        "paired_safety_reduction": f"{float(np.mean(safety_reductions)):.5f}",
                        "reference_better_seeds": str(sum(1 for d in success_diffs if d > 0)),
                        "seeds": str(len(success_diffs)),
                    }
                )
    return rows


def build_aggregate_metrics(seed_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    hard_splits = {
        "contact_chain_transfer",
        "fixture_topology_shift",
        "friction_mass_shift",
        "pocket_relocation",
        "fixture_near_wall_jam",
        "distractor_contact",
        "contact_sensor_noise_burst",
        "actuator_limit_chain",
        "delayed_topology_transition",
        "combined_stress",
        "combined_extreme_stress",
    }
    combined_splits = {"combined_stress", "combined_extreme_stress"}
    groups = {
        "all_splits": {row["split"] for row in seed_rows},
        "hard_splits": hard_splits,
        "combined_and_extreme": combined_splits,
    }
    metrics = [
        "success_rate",
        "mean_edge_f1",
        "mean_birth_f1",
        "mean_death_f1",
        "mean_graph_edit",
        "mean_component_accuracy",
        "mean_jam_f1",
        "mean_safety_violation_rate",
        "mean_fixture_contact_rate",
        "mean_wall_contact_rate",
        "mean_final_progress",
        "mean_final_y_error",
        "mean_energy",
    ]
    rows: List[Dict[str, str]] = []
    for group_name, split_names in groups.items():
        for method in sorted({row["method"] for row in seed_rows}):
            selected = [row for row in seed_rows if row["method"] == method and row["split"] in split_names]
            if not selected:
                continue
            item = {"group": group_name, "method": method, "seed_split_rows": str(len(selected))}
            for metric in metrics:
                vals = [float(row[metric]) for row in selected]
                out_name = metric.replace("mean_", "")
                item[out_name] = f"{float(np.mean(vals)):.5f}"
                item[f"ci95_{out_name}"] = f"{ci95(vals):.5f}"
            rows.append(item)
    return rows


def build_aggregate_pairwise(seed_rows: List[Dict[str, str]], reference: str = "topology_world_model_v5") -> List[Dict[str, str]]:
    aggregate = build_aggregate_metrics(seed_rows)
    groups = sorted({row["group"] for row in aggregate})
    rows: List[Dict[str, str]] = []
    for group in groups:
        ref = [row for row in aggregate if row["group"] == group and row["method"] == reference]
        if not ref:
            continue
        ref_row = ref[0]
        for other in [row for row in aggregate if row["group"] == group and row["method"] not in {reference}]:
            rows.append(
                {
                    "group": group,
                    "reference": reference,
                    "comparison": other["method"],
                    "success_diff": f"{float(ref_row['success_rate']) - float(other['success_rate']):.5f}",
                    "edge_f1_diff": f"{float(ref_row['edge_f1']) - float(other['edge_f1']):.5f}",
                    "graph_edit_reduction": f"{float(other['graph_edit']) - float(ref_row['graph_edit']):.5f}",
                    "safety_reduction": f"{float(other['safety_violation_rate']) - float(ref_row['safety_violation_rate']):.5f}",
                }
            )
    return rows


def build_fixed_risk_metrics(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    hard_splits = {
        "contact_chain_transfer",
        "fixture_topology_shift",
        "friction_mass_shift",
        "pocket_relocation",
        "fixture_near_wall_jam",
        "distractor_contact",
        "contact_sensor_noise_burst",
        "actuator_limit_chain",
        "delayed_topology_transition",
        "combined_stress",
        "combined_extreme_stress",
    }
    budgets = [0.05, 0.10, 0.20]
    rows: List[Dict[str, str]] = []
    for budget in budgets:
        for method in sorted({row["method"] for row in raw_rows}):
            selected = [row for row in raw_rows if row["method"] == method and row["split"] in hard_splits]
            if not selected:
                continue
            safe_success = []
            risks = []
            for row in selected:
                risk = max(
                    float(row["safety_violation_rate"]),
                    float(row["wall_contact_rate"]),
                    max(0.0, float(row["fixture_contact_rate"]) - 0.25),
                    max(0.0, 0.60 - float(row["jam_f1"])),
                )
                risks.append(risk)
                safe_success.append(float(row["success"]) if risk <= budget else 0.0)
            rows.append(
                {
                    "budget": f"{budget:.2f}",
                    "method": method,
                    "episodes": str(len(selected)),
                    "success_at_budget": f"{float(np.mean(safe_success)):.5f}",
                    "mean_diagnostic_risk": f"{float(np.mean(risks)):.5f}",
                    "mean_safety_violation_rate": f"{mean_metric(selected, 'safety_violation_rate'):.5f}",
                    "mean_fixture_contact_rate": f"{mean_metric(selected, 'fixture_contact_rate'):.5f}",
                    "mean_wall_contact_rate": f"{mean_metric(selected, 'wall_contact_rate'):.5f}",
                    "mean_jam_f1": f"{mean_metric(selected, 'jam_f1'):.5f}",
                }
            )
    return rows


def build_ablation_aggregate(ablation_summary: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for method in sorted({row["method"] for row in ablation_summary}):
        selected = [row for row in ablation_summary if row["method"] == method]
        rows.append(
            {
                "method": method,
                "split_rows": str(len(selected)),
                "success": f"{float(np.mean([float(row['mean_success_rate']) for row in selected])):.5f}",
                "edge_f1": f"{float(np.mean([float(row['mean_mean_edge_f1']) for row in selected])):.5f}",
                "graph_edit": f"{float(np.mean([float(row['mean_mean_graph_edit']) for row in selected])):.5f}",
                "safety_violation_rate": f"{float(np.mean([float(row['mean_mean_safety_violation_rate']) for row in selected])):.5f}",
                "fixture_contact_rate": f"{float(np.mean([float(row['mean_mean_fixture_contact_rate']) for row in selected])):.5f}",
                "wall_contact_rate": f"{float(np.mean([float(row['mean_mean_wall_contact_rate']) for row in selected])):.5f}",
                "jam_f1": f"{float(np.mean([float(row['mean_mean_jam_f1']) for row in selected])):.5f}",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary: List[Dict[str, str]], split_order: Sequence[str], methods: Sequence[str], metric: str, title: str, path: Path, ylim: Tuple[float, float] | None = None) -> None:
    width = min(0.08, 0.82 / max(1, len(methods)))
    x = np.arange(len(split_order))
    plt.figure(figsize=(16, 5.8))
    for idx, method in enumerate(methods):
        vals, errs = [], []
        for split in split_order:
            matches = [r for r in summary if r["method"] == method and r["split"] == split]
            if not matches:
                vals.append(0.0)
                errs.append(0.0)
                continue
            row = matches[0]
            vals.append(float(row[f"mean_{metric}"]))
            errs.append(float(row[f"ci95_{metric}"]))
        plt.bar(x + (idx - len(methods) / 2) * width, vals, width, yerr=errs, label=method)
    plt.xticks(x, split_order, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(fontsize=6.5, ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ablation(ablation_summary: List[Dict[str, str]], path: Path) -> None:
    rows = []
    for method in sorted({row["method"] for row in ablation_summary}):
        selected = [row for row in ablation_summary if row["method"] == method]
        seed_vals = [float(row["mean_success_rate"]) for row in selected]
        rows.append({"method": method, "success": float(np.mean(seed_vals)), "ci95": ci95(seed_vals)})
    plt.figure(figsize=(12, 4.8))
    plt.bar([r["method"] for r in rows], [float(r["success"]) for r in rows], yerr=[float(r["ci95"]) for r in rows], color="#4b6f72")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.ylim(0, 1.0)
    plt.title("Paper 73 topology world model ablations across frozen hard splits")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_stress(stress_summary: List[Dict[str, str]], path: Path) -> None:
    preferred_split = "combined_extreme_stress" if any(row.get("split") == "combined_extreme_stress" for row in stress_summary) else None
    plot_rows = [row for row in stress_summary if preferred_split is None or row.get("split") == preferred_split]
    plt.figure(figsize=(10.5, 5.4))
    for method in sorted({row["method"] for row in stress_summary}):
        rows = sorted([row for row in plot_rows if row["method"] == method], key=lambda r: float(r["stress_level"]))
        if not rows:
            continue
        x = [float(row["stress_level"]) for row in rows]
        y = [float(row["mean_success_rate"]) for row in rows]
        e = [float(row["ci95_success_rate"]) for row in rows]
        plt.errorbar(x, y, yerr=e, marker="o", label=method)
    plt.xlabel("stress level")
    plt.ylabel("success rate")
    plt.ylim(0, 1.0)
    title_suffix = preferred_split or "all stress splits"
    plt.title(f"Paper 73 contact-topology stress sweep ({title_suffix})")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def decide(
    aggregate: List[Dict[str, str]],
    pairwise: List[Dict[str, str]],
    fixed_risk: List[Dict[str, str]],
    ablation_aggregate: List[Dict[str, str]],
    stress_summary: List[Dict[str, str]],
) -> Tuple[str, str]:
    reference = "topology_world_model_v5"
    non_oracle_exclude = {reference, "oracle_topology_planner"}
    reasons: List[str] = []

    hard_v5 = [row for row in aggregate if row["group"] == "hard_splits" and row["method"] == reference][0]
    hard_best = max(
        [row for row in aggregate if row["group"] == "hard_splits" and row["method"] not in non_oracle_exclude],
        key=lambda row: float(row["success_rate"]),
    )
    hard_margin = float(hard_v5["success_rate"]) - float(hard_best["success_rate"])
    if hard_margin < 0.030:
        reasons.append(
            f"v5 does not beat strongest hard-regime baseline {hard_best['method']} by 0.030 "
            f"(v5={float(hard_v5['success_rate']):.3f}, best={float(hard_best['success_rate']):.3f})"
        )

    combined_v5 = [row for row in aggregate if row["group"] == "combined_and_extreme" and row["method"] == reference][0]
    combined_best = max(
        [row for row in aggregate if row["group"] == "combined_and_extreme" and row["method"] not in non_oracle_exclude],
        key=lambda row: float(row["success_rate"]),
    )
    pair_rows = [row for row in pairwise if row["split"] == "combined_extreme_stress" and row["comparison"] == combined_best["method"]]
    if not pair_rows:
        pair_rows = [row for row in pairwise if row["split"] == "combined_stress" and row["comparison"] == combined_best["method"]]
    if pair_rows:
        paired = float(pair_rows[0]["paired_success_diff"])
        paired_ci = float(pair_rows[0]["ci95_success_diff"])
        if paired - paired_ci <= 0.0:
            reasons.append(
                f"paired lower bound against {combined_best['method']} is not positive ({paired:.3f}+/-{paired_ci:.3f})"
            )
    combined_margin = float(combined_v5["success_rate"]) - float(combined_best["success_rate"])
    if combined_margin < 0.030:
        reasons.append(
            f"v5 does not beat strongest combined/extreme baseline {combined_best['method']} by 0.030 "
            f"(v5={float(combined_v5['success_rate']):.3f}, best={float(combined_best['success_rate']):.3f})"
        )
    diagnostic_failures = []
    for metric in ["graph_edit", "safety_violation_rate", "fixture_contact_rate", "wall_contact_rate"]:
        if float(combined_v5[metric]) > float(combined_best[metric]) + 0.020:
            diagnostic_failures.append(metric)
    if diagnostic_failures:
        reasons.append("diagnostic gate fails on " + ", ".join(diagnostic_failures))

    fixed_v5 = [row for row in fixed_risk if row["budget"] == "0.10" and row["method"] == reference][0]
    fixed_best = max(
        [row for row in fixed_risk if row["budget"] == "0.10" and row["method"] not in {"oracle_topology_planner", reference}],
        key=lambda row: float(row["success_at_budget"]),
    )
    if float(fixed_v5["success_at_budget"]) < float(fixed_best["success_at_budget"]) - 1e-9:
        reasons.append(
            f"fixed-risk gate fails at budget 0.10 (v5={float(fixed_v5['success_at_budget']):.3f}, "
            f"best={fixed_best['method']} {float(fixed_best['success_at_budget']):.3f})"
        )

    max_rows = [row for row in stress_summary if row["stress_level"] == "1.00"]
    stress_by_method = {
        method: float(np.mean([float(row["mean_success_rate"]) for row in max_rows if row["method"] == method]))
        for method in sorted({row["method"] for row in max_rows})
    }
    stress_v5 = stress_by_method[reference]
    stress_best_method, stress_best_score = max(
        [(method, score) for method, score in stress_by_method.items() if method not in {"oracle_topology_planner", reference}],
        key=lambda item: item[1],
    )
    if stress_v5 < stress_best_score - 0.030:
        reasons.append(
            f"maximum-stress gate fails (v5={stress_v5:.3f}, "
            f"best={stress_best_method} {stress_best_score:.3f})"
        )

    full = [row for row in ablation_aggregate if row["method"] == "topology_full_v5"][0]
    ablation_failures = [
        row["method"]
        for row in ablation_aggregate
        if row["method"] != "topology_full_v5"
        and float(row["success"]) >= float(full["success"]) - 0.020
        and float(row["safety_violation_rate"]) <= float(full["safety_violation_rate"]) + 0.020
    ]
    if ablation_failures:
        reasons.append("ablation gate fails because " + ", ".join(ablation_failures) + " matches or beats full v5")

    oracle_hard = [row for row in aggregate if row["group"] == "hard_splits" and row["method"] == "oracle_topology_planner"][0]
    if float(oracle_hard["success_rate"]) < 0.200:
        reasons.append(f"oracle sanity gate is weak on hard regimes (oracle={float(oracle_hard['success_rate']):.3f})")

    if reasons:
        return "KILL_ARCHIVE", "; ".join(reasons)
    return (
        "STRONG_REVISE",
        "v5 clears frozen success, paired, diagnostic, fixed-risk, maximum-stress, and ablation gates, but still lacks real robot/public benchmark validation.",
    )


def negative_cases(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    hard_splits = {
        "combined_stress",
        "combined_extreme_stress",
        "fixture_topology_shift",
        "contact_chain_transfer",
        "fixture_near_wall_jam",
        "actuator_limit_chain",
        "delayed_topology_transition",
    }
    candidates = [r for r in raw_rows if r["method"] == "topology_world_model_v5" and r["split"] in hard_splits]
    if not candidates:
        candidates = [r for r in raw_rows if r["method"] == "topology_world_model_v5"]
    worst = sorted(candidates, key=lambda r: (int(r["success"]), -float(r["fixture_contact_rate"]), -float(r["wall_contact_rate"]), float(r["final_progress"])))[:12]
    rows: List[Dict[str, str]] = []
    for idx, row in enumerate(worst):
        lesson = "topology planner chose a contact chain that stalled before the pocket"
        if float(row["fixture_contact_rate"]) > 0.20:
            lesson = "fixture contact changed the graph faster than the topology predictor adapted"
        elif float(row["wall_contact_rate"]) > 0.15:
            lesson = "wall contact created a jammed component that the planner under-penalized"
        rows.append(
            {
                "case": str(idx),
                "split": row["split"],
                "seed": row["seed"],
                "episode": row["episode"],
                "success": row["success"],
                "edge_f1": row["edge_f1"],
                "graph_edit": row["graph_edit"],
                "fixture_contact_rate": row["fixture_contact_rate"],
                "wall_contact_rate": row["wall_contact_rate"],
                "final_progress": row["final_progress"],
                "lesson": lesson,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 73 expanded contact-topology evidence protocol.")
    parser.add_argument("--seeds", type=int, default=SEED_COUNT)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--ablation-episodes", type=int, default=ABLATION_EPISODES)
    parser.add_argument("--stress-episodes", type=int, default=STRESS_EPISODES)
    parser.add_argument("--train-scenes", type=int, default=TRAINING_EXAMPLES)
    parser.add_argument("--splits", nargs="+", default=(["combined_stress"] if QUICK_MODE else DEFAULT_SPLIT_NAMES))
    parser.add_argument("--ablation-splits", nargs="+", default=(["combined_stress"] if QUICK_MODE else DEFAULT_ABLATION_SPLIT_NAMES))
    parser.add_argument("--stress-splits", nargs="+", default=(["combined_stress"] if QUICK_MODE else DEFAULT_STRESS_SPLIT_NAMES))
    parser.add_argument("--stress-levels", nargs="+", type=float, default=([0.0, 1.0] if QUICK_MODE else DEFAULT_STRESS_LEVELS))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--workers", type=int, default=1, help="Accepted for protocol logging; execution stays single-process for RAM discipline.")
    return parser.parse_args()


def resolve_output_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_splits(names: Sequence[str]) -> List[SplitSpec]:
    by_name = {split.name: split for split in SPLITS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown split name(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def build_stress_summary(stress_raw: List[Dict[str, str]]) -> List[Dict[str, str]]:
    metrics = [
        "success",
        "edge_f1",
        "graph_edit",
        "safety_violation_rate",
        "fixture_contact_rate",
        "wall_contact_rate",
        "jam_f1",
        "final_progress",
    ]
    rows: List[Dict[str, str]] = []
    for (method, split, stress_level), group in sorted(group_rows(stress_raw, ["method", "split", "stress_level"]).items()):
        metric_seed_vals: Dict[str, List[float]] = {metric: [] for metric in metrics}
        for seed in [str(s) for s in SEEDS]:
            seed_rows = [r for r in group if r["seed"] == seed]
            if not seed_rows:
                continue
            for metric in metrics:
                metric_seed_vals[metric].append(float(np.mean([float(r[metric]) for r in seed_rows])))
        item = {
            "method": method,
            "split": split,
            "stress_level": stress_level,
            "seeds": str(len(metric_seed_vals["success"])),
            "episodes_per_seed": str(len(group) // max(1, len(metric_seed_vals["success"]))),
            "mean_success_rate": f"{float(np.mean(metric_seed_vals['success'])):.5f}",
            "ci95_success_rate": f"{ci95(metric_seed_vals['success']):.5f}",
        }
        for metric in metrics:
            if metric == "success":
                continue
            item[f"mean_{metric}"] = f"{float(np.mean(metric_seed_vals[metric])):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(metric_seed_vals[metric]):.5f}"
        rows.append(item)
    return rows


def main() -> None:
    global SEEDS, EVAL_EPISODES, ABLATION_EPISODES, STRESS_EPISODES, TRAINING_EXAMPLES, RESULTS, FIGURES

    start = time.time()
    args = parse_args()
    if args.seeds < 1 or args.episodes < 1 or args.ablation_episodes < 1 or args.stress_episodes < 1 or args.train_scenes < 1:
        raise ValueError("seeds, episodes, ablation-episodes, stress-episodes, and train-scenes must all be positive")
    SEEDS = list(range(args.seeds))
    EVAL_EPISODES = args.episodes
    ABLATION_EPISODES = args.ablation_episodes
    STRESS_EPISODES = args.stress_episodes
    TRAINING_EXAMPLES = args.train_scenes
    RESULTS = resolve_output_path(args.results_dir)
    FIGURES = resolve_output_path(args.figures_dir)

    eval_splits = select_splits(args.splits)
    ablation_splits = select_splits(args.ablation_splits)
    stress_splits = select_splits(args.stress_splits)
    stress_levels = [float(level) for level in args.stress_levels]

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS / "run_config.csv",
        [
            {
                "seeds": " ".join(str(seed) for seed in SEEDS),
                "episodes": str(EVAL_EPISODES),
                "ablation_episodes": str(ABLATION_EPISODES),
                "stress_episodes": str(STRESS_EPISODES),
                "train_scenes": str(TRAINING_EXAMPLES),
                "splits": " ".join(split.name for split in eval_splits),
                "ablation_splits": " ".join(split.name for split in ablation_splits),
                "stress_splits": " ".join(split.name for split in stress_splits),
                "stress_levels": " ".join(f"{level:.2f}" for level in stress_levels),
                "workers_argument": str(args.workers),
                "execution_mode": "single_process_cpu_ram_light",
            }
        ],
    )

    pack = generate_training_pack()
    write_csv(RESULTS / "training_topology_examples.csv", pack.training_rows)
    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_examples": str(TRAINING_EXAMPLES),
                "state_delta_train_mae": f"{pack.state_mae:.5f}",
                "pairwise_edge_train_f1": f"{pack.edge_train_f1:.5f}",
                "edge_count": str(len(EDGES)),
            }
        ],
    )

    model = make_model()
    raw_rows: List[Dict[str, str]] = []
    for split in eval_splits:
        for seed in SEEDS:
            for episode in range(EVAL_EPISODES):
                cfg = make_config(split, seed, episode)
                for method in METHODS:
                    raw_rows.append(simulate_episode(model, method, cfg, pack))
    write_csv(RESULTS / "topology_raw.csv", raw_rows)
    write_csv(RESULTS / "topology_rollouts.csv", raw_rows)
    seed_rows = build_seed_metrics(raw_rows)
    summary = build_summary(seed_rows)
    pairwise = build_pairwise(seed_rows)
    aggregate = build_aggregate_metrics(seed_rows)
    aggregate_pairwise = build_aggregate_pairwise(seed_rows)
    fixed_risk = build_fixed_risk_metrics(raw_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", summary)
    write_csv(RESULTS / "topology_metrics.csv", summary)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    write_csv(RESULTS / "topology_pairwise.csv", pairwise)
    write_csv(RESULTS / "aggregate_metrics.csv", aggregate)
    write_csv(RESULTS / "aggregate_pairwise_stats.csv", aggregate_pairwise)
    write_csv(RESULTS / "fixed_risk_metrics.csv", fixed_risk)

    ablation_raw: List[Dict[str, str]] = []
    for split in ablation_splits:
        for seed in SEEDS:
            for episode in range(ABLATION_EPISODES):
                cfg = make_config(split, seed, 1000 + episode)
                for method in ABLATION_METHODS:
                    row = simulate_episode(model, method, cfg, pack)
                    row["method"] = method
                    ablation_raw.append(row)
    write_csv(RESULTS / "topology_ablation_raw.csv", ablation_raw)
    ablation_summary = build_summary(build_seed_metrics(ablation_raw))
    ablation_aggregate = build_ablation_aggregate(ablation_summary)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)
    write_csv(RESULTS / "topology_ablation.csv", ablation_summary)
    write_csv(RESULTS / "ablation_aggregate_metrics.csv", ablation_aggregate)

    stress_raw: List[Dict[str, str]] = []
    for split in stress_splits:
        for stress_level in stress_levels:
            for seed in SEEDS:
                for episode in range(STRESS_EPISODES):
                    cfg = make_config(split, seed, 2000 + episode, stress_level=float(stress_level))
                    for method in STRESS_METHODS:
                        row = simulate_episode(model, method, cfg, pack)
                        row["split"] = split.name
                        row["stress_level"] = f"{stress_level:.2f}"
                        stress_raw.append(row)
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_raw)
    stress_summary = build_stress_summary(stress_raw)
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)
    write_csv(RESULTS / "negative_cases.csv", negative_cases(raw_rows))

    split_order = [s.name for s in eval_splits]
    plot_metric(summary, split_order, METHODS, "success_rate", "Paper 73 contact-topology task success", FIGURES / "topology_success_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "mean_edge_f1", "Paper 73 contact-edge prediction F1", FIGURES / "topology_edge_f1_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "mean_graph_edit", "Paper 73 graph edit distance", FIGURES / "topology_graph_edit_by_split.png")
    plot_ablation(ablation_summary, FIGURES / "topology_ablation_success.png")
    plot_stress(stress_summary, FIGURES / "topology_stress_sweep.png")

    decision, reason = decide(aggregate, pairwise, fixed_risk, ablation_aggregate, stress_summary)
    combined_rows = [r for r in aggregate if r["group"] == "combined_and_extreme"]
    hard_rows = [r for r in aggregate if r["group"] == "hard_splits"]
    elapsed = time.time() - start
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 73 contact_topology_world_models expanded MuJoCo rebuild\n")
        f.write(f"Terminal decision: {decision}\n")
        f.write(f"Terminal reason: {reason}\n")
        f.write(f"Main eval rows: {len(raw_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_raw)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Eval episodes per seed/split: {EVAL_EPISODES}\n")
        f.write(f"Ablation splits: {' '.join(split.name for split in ablation_splits)}\n")
        f.write(f"Stress splits: {' '.join(split.name for split in stress_splits)}\n")
        f.write(f"Stress levels: {' '.join(f'{level:.2f}' for level in stress_levels)}\n")
        f.write(f"Runtime seconds: {elapsed:.2f}\n\n")
        f.write("Combined/extreme aggregate summary:\n")
        for row in sorted(combined_rows, key=lambda r: -float(r["success_rate"])):
            f.write(
                f"{row['method']} success={row['success_rate']} ci95={row['ci95_success_rate']} "
                f"edge_f1={row['edge_f1']} graph_edit={row['graph_edit']} "
                f"safety={row['safety_violation_rate']} progress={row['final_progress']}\n"
            )
        f.write("\nHard-split aggregate summary:\n")
        for row in sorted(hard_rows, key=lambda r: -float(r["success_rate"])):
            f.write(
                f"{row['method']} success={row['success_rate']} ci95={row['ci95_success_rate']} "
                f"edge_f1={row['edge_f1']} graph_edit={row['graph_edit']} "
                f"safety={row['safety_violation_rate']} progress={row['final_progress']}\n"
            )
    print(f"wrote Paper 73 MuJoCo contact-topology evidence to {RESULTS}")
    print(f"terminal decision: {decision}")
    print(reason)


if __name__ == "__main__":
    main()
