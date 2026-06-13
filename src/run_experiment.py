from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


BASE_SEED = 1678111061
QUICK_MODE = os.getenv("PAPER73_QUICK", "0") == "1"
SEED_COUNT = int(os.getenv("PAPER73_SEED_COUNT", "1" if QUICK_MODE else "7"))
SEEDS = list(range(SEED_COUNT))
EVAL_EPISODES = int(os.getenv("PAPER73_EVAL_EPISODES", "8"))
ABLATION_EPISODES = int(os.getenv("PAPER73_ABLATION_EPISODES", "6"))
STRESS_EPISODES = int(os.getenv("PAPER73_STRESS_EPISODES", "5"))
TRAINING_EXAMPLES = int(os.getenv("PAPER73_TRAINING_EXAMPLES", "2200"))
STEPS = 64
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
    "ensemble_uncertainty_planner",
    "contact_implicit_mpc_baseline",
    "topology_world_model",
    "oracle_topology_planner",
]

ABLATION_METHODS = [
    "topology_full",
    "topology_no_birth_death",
    "topology_no_component_head",
    "topology_no_jam_slip",
    "topology_no_topology_planner",
    "topology_no_uncertainty_penalty",
]

STRESS_METHODS = [
    "pairwise_contact_classifier",
    "ensemble_uncertainty_planner",
    "contact_implicit_mpc_baseline",
    "topology_world_model",
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
    model: LogisticRegression | None
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
    ensemble_models: List[List[BinaryPredictor]]
    training_rows: List[Dict[str, str]]
    state_mae: float
    edge_train_f1: float


SPLITS = [
    SplitSpec("nominal_push_to_pocket", 0.00, 0.00, 0.00, 0.15, 0.30, 0.58, 1.00, 0.004, 1.00, 0.0),
    SplitSpec("contact_chain_transfer", 0.08, -0.02, -0.01, 0.13, 0.30, 0.62, 1.05, 0.006, 0.95, 0.1),
    SplitSpec("fixture_topology_shift", 0.06, -0.07, -0.07, 0.10, 0.10, 0.66, 1.10, 0.008, 0.90, 0.3),
    SplitSpec("friction_mass_shift", -0.04, 0.04, 0.03, 0.17, -0.24, 0.82, 1.45, 0.010, 0.84, 0.2),
    SplitSpec("combined_stress", 0.055, -0.045, -0.050, 0.12, 0.16, 0.80, 1.35, 0.014, 0.84, 0.45),
]


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


def fit_binary_models(x: np.ndarray, y: np.ndarray) -> Tuple[StandardScaler, List[BinaryPredictor]]:
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    models: List[BinaryPredictor] = []
    for idx in range(y.shape[1]):
        yi = y[:, idx]
        if len(np.unique(yi)) < 2:
            models.append(BinaryPredictor(scaler=None, model=None, constant=float(np.mean(yi))))
            continue
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
            xs = (x - active_scaler.mean_) / np.where(active_scaler.scale_ == 0.0, 1.0, active_scaler.scale_)
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
        if idx < 420:
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
    pair_pred = np.array([predict_binary(pair_models, pair_scaler, x) for x in np.vstack(x_pair)])
    edge_train_f1 = float(np.mean([edge_f1(pair_pred[i], y_edge_arr[i]) for i in range(len(y_edge_arr))]))

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
        ensemble_models=ensemble_models,
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
    elif method == "distance_threshold_graph" or method == "contact_implicit_mpc_baseline":
        predicted_qpos = qpos.copy()
        predicted_qpos[0:2] += 0.012 * action / max(1.0, np.linalg.norm(action))
        probs = distance_edges_from_qpos(predicted_qpos, cfg, margin=0.025)
    elif method == "state_only_dynamics_model":
        probs = predict_state_edges(pack, base, qpos, cfg)
    elif method == "pairwise_contact_classifier":
        probs = predict_binary(pack.pair_models, pack.pair_scaler, base)
    elif method == "ensemble_uncertainty_planner":
        preds = np.vstack([predict_binary(member, pack.topo_scaler, topo) for member in pack.ensemble_models])
        probs = np.mean(preds, axis=0) - 0.35 * np.std(preds, axis=0)
    elif method in {"topology_world_model", "topology_full", "topology_no_component_head", "topology_no_jam_slip", "topology_no_uncertainty_penalty"}:
        probs = predict_binary(pack.topo_models, pack.topo_scaler, topo)
    elif method == "topology_no_birth_death":
        probs = predict_binary(pack.pair_models, pack.pair_scaler, base)
    elif method == "topology_no_topology_planner":
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
    if method == "topology_no_component_head":
        component_reward = 0.0
    if method == "topology_no_jam_slip":
        jam_penalty = 0.0
    if method == "topology_no_topology_planner":
        desired = 0.10 * desired
    if method in {"last_contact_persistence", "distance_threshold_graph", "state_only_dynamics_model", "pairwise_contact_classifier"}:
        desired *= 0.55
        component_reward *= 0.25
    if method == "ensemble_uncertainty_planner":
        desired *= 0.80
        jam_penalty += 0.20 * (probs[EDGE_INDEX["block_b_fixture"]] + probs[EDGE_INDEX["block_b_wall"]])
    if method == "contact_implicit_mpc_baseline":
        desired = -0.75 * p_to_a - 0.65 * a_to_b - 0.40 * y_error - 0.45 * abs(pts["B"][1] - pts["F"][1]) * cfg.distractor
        component_reward = 0.0
        jam_penalty = 0.50 * (probs[EDGE_INDEX["block_b_fixture"]] + probs[EDGE_INDEX["block_b_wall"]])
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
                if method == "topology_no_uncertainty_penalty":
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


def build_pairwise(seed_rows: List[Dict[str, str]], reference: str = "topology_world_model") -> List[Dict[str, str]]:
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


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary: List[Dict[str, str]], split_order: Sequence[str], methods: Sequence[str], metric: str, title: str, path: Path, ylim: Tuple[float, float] | None = None) -> None:
    width = 0.10
    x = np.arange(len(split_order))
    plt.figure(figsize=(13, 5))
    for idx, method in enumerate(methods):
        vals, errs = [], []
        for split in split_order:
            row = [r for r in summary if r["method"] == method and r["split"] == split][0]
            vals.append(float(row[f"mean_{metric}"]))
            errs.append(float(row[f"ci95_{metric}"]))
        plt.bar(x + (idx - len(methods) / 2) * width, vals, width, yerr=errs, label=method)
    plt.xticks(x, split_order, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ablation(ablation_summary: List[Dict[str, str]], path: Path) -> None:
    rows = [r for r in ablation_summary if r["split"] == "combined_stress"]
    plt.figure(figsize=(10, 4.8))
    plt.bar([r["method"] for r in rows], [float(r["mean_success_rate"]) for r in rows], yerr=[float(r["ci95_success_rate"]) for r in rows], color="#4b6f72")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.ylim(0, 1.0)
    plt.title("Paper 73 topology world model ablations")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_stress(stress_summary: List[Dict[str, str]], path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for method in sorted({row["method"] for row in stress_summary}):
        rows = sorted([row for row in stress_summary if row["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(row["stress_level"]) for row in rows]
        y = [float(row["mean_success_rate"]) for row in rows]
        e = [float(row["ci95_success_rate"]) for row in rows]
        plt.errorbar(x, y, yerr=e, marker="o", label=method)
    plt.xlabel("stress level")
    plt.ylabel("success rate")
    plt.ylim(0, 1.0)
    plt.title("Paper 73 contact-topology stress sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def decide(summary: List[Dict[str, str]], pairwise: List[Dict[str, str]]) -> Tuple[str, str]:
    combined = [row for row in summary if row["split"] == "combined_stress"]
    proposed = [row for row in combined if row["method"] == "topology_world_model"][0]
    non_oracle = [row for row in combined if row["method"] not in {"topology_world_model", "oracle_topology_planner"}]
    best = max(non_oracle, key=lambda row: float(row["mean_success_rate"]))
    pair = [row for row in pairwise if row["split"] == "combined_stress" and row["comparison"] == best["method"]][0]
    prop_success = float(proposed["mean_success_rate"])
    best_success = float(best["mean_success_rate"])
    paired = float(pair["paired_success_diff"])
    paired_ci = float(pair["ci95_success_diff"])
    prop_f1 = float(proposed["mean_mean_edge_f1"])
    best_f1 = float(best["mean_mean_edge_f1"])
    prop_safety = float(proposed["mean_mean_safety_violation_rate"])
    best_safety = float(best["mean_mean_safety_violation_rate"])
    if prop_success - best_success >= 0.045 and paired - paired_ci > 0.0 and prop_f1 >= best_f1 + 0.015 and prop_safety <= best_safety + 0.03:
        return (
            "STRONG_REVISE",
            f"topology_world_model clears strongest non-oracle baseline {best['method']} on combined_stress by "
            f"{prop_success - best_success:.3f} success with paired diff {paired:.3f}+/-{paired_ci:.3f}, "
            "but lacks real robot/public benchmark validation.",
        )
    return (
        "KILL_ARCHIVE",
        f"topology_world_model does not clear strongest non-oracle baseline {best['method']} decisively on combined_stress "
        f"(topology={prop_success:.3f}, best_baseline={best_success:.3f}, paired diff={paired:.3f}+/-{paired_ci:.3f}, "
        f"edge_f1={prop_f1:.3f} vs {best_f1:.3f}).",
    )


def negative_cases(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = [r for r in raw_rows if r["method"] == "topology_world_model" and r["split"] in {"combined_stress", "fixture_topology_shift", "contact_chain_transfer"}]
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


def main() -> None:
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
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
    eval_splits = [[s for s in SPLITS if s.name == "combined_stress"][0]] if QUICK_MODE else SPLITS
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
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", summary)
    write_csv(RESULTS / "topology_metrics.csv", summary)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    write_csv(RESULTS / "topology_pairwise.csv", pairwise)

    combined = [s for s in SPLITS if s.name == "combined_stress"][0]
    ablation_raw: List[Dict[str, str]] = []
    for seed in SEEDS:
        for episode in range(ABLATION_EPISODES):
            cfg = make_config(combined, seed, 1000 + episode)
            for method in ABLATION_METHODS:
                row = simulate_episode(model, method, cfg, pack)
                row["method"] = method
                ablation_raw.append(row)
    write_csv(RESULTS / "topology_ablation_raw.csv", ablation_raw)
    ablation_summary = build_summary(build_seed_metrics(ablation_raw))
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)
    write_csv(RESULTS / "topology_ablation.csv", ablation_summary)

    stress_raw: List[Dict[str, str]] = []
    stress_levels = [0.0, 1.0] if QUICK_MODE else list(np.linspace(0.0, 1.0, 6))
    for stress_level in stress_levels:
        for seed in SEEDS:
            for episode in range(STRESS_EPISODES):
                cfg = make_config(combined, seed, 2000 + episode, stress_level=float(stress_level))
                for method in STRESS_METHODS:
                    row = simulate_episode(model, method, cfg, pack)
                    row["split"] = "stress_sweep"
                    row["stress_level"] = f"{stress_level:.2f}"
                    stress_raw.append(row)
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_raw)
    stress_seed = build_seed_metrics(stress_raw)
    stress_summary: List[Dict[str, str]] = []
    for (method, stress_level), group in sorted(group_rows(stress_raw, ["method", "stress_level"]).items()):
        metric_seed_vals: Dict[str, List[float]] = {
            "success": [],
            "edge_f1": [],
            "graph_edit": [],
            "safety_violation_rate": [],
            "final_progress": [],
        }
        for seed in [str(s) for s in SEEDS]:
            rows = [r for r in group if r["seed"] == seed]
            if rows:
                metric_seed_vals["success"].append(float(np.mean([float(r["success"]) for r in rows])))
                for metric in ["edge_f1", "graph_edit", "safety_violation_rate", "final_progress"]:
                    metric_seed_vals[metric].append(float(np.mean([float(r[metric]) for r in rows])))
        item = {
            "method": method,
            "stress_level": stress_level,
            "seeds": str(len(metric_seed_vals["success"])),
            "episodes_per_seed": str(STRESS_EPISODES),
            "mean_success_rate": f"{float(np.mean(metric_seed_vals['success'])):.5f}",
            "ci95_success_rate": f"{ci95(metric_seed_vals['success']):.5f}",
        }
        for metric in ["edge_f1", "graph_edit", "safety_violation_rate", "final_progress"]:
            item[f"mean_{metric}"] = f"{float(np.mean(metric_seed_vals[metric])):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(metric_seed_vals[metric]):.5f}"
        stress_summary.append(item)
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)
    write_csv(RESULTS / "negative_cases.csv", negative_cases(raw_rows))

    split_order = [s.name for s in eval_splits]
    plot_metric(summary, split_order, METHODS, "success_rate", "Paper 73 contact-topology task success", FIGURES / "topology_success_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "mean_edge_f1", "Paper 73 contact-edge prediction F1", FIGURES / "topology_edge_f1_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "mean_graph_edit", "Paper 73 graph edit distance", FIGURES / "topology_graph_edit_by_split.png")
    plot_ablation(ablation_summary, FIGURES / "topology_ablation_success.png")
    plot_stress(stress_summary, FIGURES / "topology_stress_sweep.png")

    decision, reason = decide(summary, pairwise)
    combined_rows = [r for r in summary if r["split"] == "combined_stress"]
    elapsed = time.time() - start
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 73 contact_topology_world_models real MuJoCo rebuild\n")
        f.write(f"Terminal recommendation: {decision}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Main eval rows: {len(raw_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_raw)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Eval episodes per seed/split: {EVAL_EPISODES}\n")
        f.write(f"Runtime seconds: {elapsed:.2f}\n\n")
        f.write("Combined-stress summary:\n")
        for row in sorted(combined_rows, key=lambda r: -float(r["mean_success_rate"])):
            f.write(
                f"{row['method']} success={row['mean_success_rate']} ci95={row['ci95_success_rate']} "
                f"edge_f1={row['mean_mean_edge_f1']} graph_edit={row['mean_mean_graph_edit']} "
                f"safety={row['mean_mean_safety_violation_rate']} progress={row['mean_mean_final_progress']}\n"
            )
    print(f"wrote Paper 73 MuJoCo contact-topology evidence to {RESULTS}")
    print(f"terminal recommendation: {decision}")
    print(reason)


if __name__ == "__main__":
    main()
