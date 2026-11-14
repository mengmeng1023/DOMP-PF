"""S18 candidate-only closed loop; all coordinates are continuous MATLAB1.

No image, ground-truth, file I/O, or experiment contract enters this module.
The input tracker is never mutated. Snapshot dictionaries use the predecision
roles H/T; carried snapshots use the roles in the returned tracker.
"""
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np

from . import core


def _finite_scalar(value, name, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f'{name} must be a finite real' + (' > 0' if positive else ''))


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class Candidate:
    candidate_id: int | str
    center_xy: tuple[float, float]
    image_score: float

    def __post_init__(self):
        if isinstance(self.candidate_id, bool) or not isinstance(self.candidate_id, (Integral, str)):
            raise ValueError('candidate_id must be an integer or string')
        center = np.asarray(self.center_xy, dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError('center_xy must be a finite x,y pair')
        _finite_scalar(self.image_score, 'image_score')
        object.__setattr__(self, 'center_xy', tuple(float(v) for v in center))


@dataclass
class Branch:
    path_id: int
    state: core.State


@dataclass
class Tracker:
    H: Branch
    T: Branch | None = None
    C_T: int = 0
    next_path_id: int = 1


@dataclass(frozen=True)
class Config:
    seed: int = 7
    K: int = 3
    dual_enabled: bool = True
    total_particles: int = 500
    position_sigma: float = 2.
    velocity_sigma: float = .001
    amplitude_sigma: float = .01
    dt: float = 1.
    sigma_c: float = 3.
    sigma_obs: float = 3.
    r: float = 5.
    Q_min: float = .02
    M_min: float = .01
    ess_fraction: float = .5

    def __post_init__(self):
        _integer(self.seed, 'seed')
        _integer(self.K, 'K', 1)
        if self.K not in (2, 3, 8):
            raise ValueError('K must be 2, 3 or 8')
        if not isinstance(self.dual_enabled, bool):
            raise ValueError('dual_enabled must be bool')
        _integer(self.total_particles, 'total_particles', 1)
        if self.total_particles != 500:
            raise ValueError('S18 total_particles must be 500')
        for name in ('dt', 'sigma_c', 'sigma_obs', 'r'):
            _finite_scalar(getattr(self, name), name, positive=True)
        for name in ('position_sigma', 'velocity_sigma', 'amplitude_sigma', 'Q_min', 'M_min', 'ess_fraction'):
            value = getattr(self, name)
            _finite_scalar(value, name)
            if value < 0 or (name in ('Q_min', 'M_min', 'ess_fraction') and value > 1):
                raise ValueError(f'{name} outside valid range')


@dataclass(frozen=True)
class Support:
    role: str
    path_id: int
    candidate_id: int | str
    Q: float
    M: float
    logM: float
    count: int
    nearest: float
    eligible: bool


@dataclass(frozen=True)
class Match:
    status: str
    candidate: Candidate | None
    support_rows: tuple[Support, ...]


@dataclass
class StepResult:
    output_xy: np.ndarray
    output_source: str
    action: str
    matches: dict[str, Match]
    support_rows: list[Support]
    accepted_candidate: dict[str, Candidate] = field(default_factory=dict)
    parent_indices: dict[str, np.ndarray] = field(default_factory=dict)
    parent_path_ids: dict[str, int] = field(default_factory=dict)
    resampling_u: dict[str, float] = field(default_factory=dict)
    unique_parent_count: dict[str, int] = field(default_factory=dict)
    ess: dict[str, float] = field(default_factory=dict)
    update_counts: dict[str, int] = field(default_factory=dict)
    predicted: dict[str, Branch] = field(default_factory=dict)
    posterior: dict[str, Branch] = field(default_factory=dict)
    carried: dict[str, Branch] = field(default_factory=dict)


def _particles(value):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError('particles must be a nonempty finite N by 5 array')
    return p


def _weights(value):
    w = np.asarray(value, dtype=np.float64)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('weights must be a finite nonnegative vector')
    try:
        total = math.fsum(w)
    except OverflowError as error:
        raise ValueError('weights must have finite positive total mass') from error
    if not np.isfinite(total) or total <= 0:
        raise ValueError('weights must have finite positive total mass')
    return w / total


def _log_prior(value, count):
    lp = np.asarray(value, dtype=np.float64)
    if (lp.shape != (count,) or np.isnan(lp).any() or np.isposinf(lp).any()
            or not np.isfinite(lp).any()):
        raise ValueError('log prior requires a matching shape and some finite mass')
    return lp


def _logsumexp(value):
    peak = float(np.max(value))
    if not np.isfinite(peak):
        raise ValueError('no finite log mass remains')
    return peak + math.log(float(np.exp(value - peak).sum()))


def position_posterior(particles, log_prior, z_xy, sigma=3.):
    """Return normalized position-only weights without moving particle states."""
    p = _particles(particles)
    lp = _log_prior(log_prior, len(p))
    z = np.asarray(z_xy, dtype=np.float64)
    if z.shape != (2,) or not np.isfinite(z).all():
        raise ValueError('observation center must be a finite pair')
    _finite_scalar(sigma, 'sigma', positive=True)
    post = lp - np.sum((p[:, :2] - z)**2, axis=1)/(2*sigma**2)
    peak = float(np.max(post))
    if not np.isfinite(peak):
        raise ValueError('no finite posterior mass remains')
    shifted = post - peak
    mass = np.exp(shifted)
    total = float(mass.sum())
    return mass/total, shifted - math.log(total)


def systematic_indices(weights, count, u):
    """Sample arbitrary count, with CDF equality selecting the right interval."""
    w = _weights(weights)
    _integer(count, 'count', 1)
    _finite_scalar(u, 'u')
    if not 0 <= u < 1:
        raise ValueError('u must be in [0, 1)')
    cdf = np.cumsum(w)
    cdf[-1] = 1.
    return np.minimum(np.searchsorted(cdf, (u + np.arange(count))/count, side='right'), len(w)-1)


def _validate_branch(branch):
    _integer(branch.path_id, 'path_id')
    s = branch.state
    p = _particles(s.particles)
    w = _weights(s.weights)
    lp = _log_prior(s.log_weights, len(p))
    if len(w) != len(p):
        raise ValueError('particle and weight shapes differ')
    normalized_lp = lp - _logsumexp(lp)
    if not np.allclose(np.exp(normalized_lp), w, rtol=1e-10, atol=1e-14):
        raise ValueError('weights and log_weights describe different priors')
    if len(s.frame_size) != 2:
        raise ValueError('frame_size requires height and width')
    for size in s.frame_size:
        _integer(size, 'frame dimension', 1)
    _integer(s.step_index, 'step_index')


def score_support(branch, candidate, config=Config(), role=''):
    """Branch-normalized Q and stable logM for one candidate, before updating."""
    _validate_branch(branch)
    p = branch.state.particles
    w = _weights(branch.state.weights)
    d2 = np.sum((p[:, :2] - candidate.center_xy)**2, axis=1)
    inside = d2 <= config.r**2
    Q = math.fsum(w[inside])
    positive = w > 0
    logM = _logsumexp(np.log(w[positive]) - d2[positive]/(2*config.sigma_c**2))
    M = math.exp(logM)
    return Support(role, branch.path_id, candidate.candidate_id, Q, M, logM,
                   int(inside.sum()), float(np.sqrt(d2.min())),
                   bool(Q >= config.Q_min and M >= config.M_min))


def match_candidates(branch, candidates, config=Config(), role=''):
    rows = tuple(score_support(branch, c, config, role) for c in candidates)
    eligible = [i for i, row in enumerate(rows) if row.eligible]
    if not eligible:
        return Match('none', None, rows)
    best = max(eligible, key=lambda i: rows[i].logM)
    tied = [i for i in eligible if abs(rows[i].logM - rows[best].logM) <= 1e-12]
    if len(tied) > 1:
        return Match('ambiguous', None, rows)
    return Match('unique', candidates[best], rows)


def _clone(branch):
    s = branch.state
    return Branch(branch.path_id, core.State(np.array(s.particles, dtype=float, copy=True),
                  np.array(s.weights, dtype=float, copy=True), np.array(s.log_weights, dtype=float, copy=True),
                  tuple(s.frame_size), s.step_index))


def _rng(config, frame_id, path_id, operation):
    return np.random.default_rng(np.random.SeedSequence([config.seed, frame_id, path_id, operation]))


def _ess(state):
    # Scale invariance also keeps exact equal-support boundaries (e.g. 125) exact.
    scaled = state.weights / np.max(state.weights)
    total = math.fsum(scaled)
    return total*total/math.fsum(scaled*scaled)


def _xy(branch):
    return _weights(branch.state.weights) @ branch.state.particles[:, :2]


def _sample(branch, count, operation, destination, frame_id, config, result):
    u = float(_rng(config, frame_id, branch.path_id, operation).random())
    parents = systematic_indices(branch.state.weights, count, u)
    result.parent_indices[destination] = parents.copy()
    result.parent_path_ids[destination] = branch.path_id
    result.resampling_u[destination] = u
    result.unique_parent_count[destination] = int(len(np.unique(parents)))
    state = branch.state
    return Branch(branch.path_id, core.State(state.particles[parents].copy(),
                  np.full(count, 1/count), np.full(count, -np.log(count)), state.frame_size, state.step_index))


def _observe(branch, role, match, config, result):
    branch.state.weights, branch.state.log_weights = position_posterior(
        branch.state.particles, branch.state.log_weights, match.candidate.center_xy, config.sigma_obs)
    result.update_counts[role] += 1
    result.accepted_candidate[role] = match.candidate
    result.posterior[role] = _clone(branch)
    result.ess[role] = _ess(branch.state)


def _regular_resample(branch, role, frame_id, config, result):
    if result.ess[role] < config.ess_fraction*len(branch.state.weights):
        return _sample(branch, len(branch.state.weights), 2, role, frame_id, config, result)
    return branch


def step(tracker, candidates, frame_id, config=Config(), *, branch_radius=None):
    """Predict once, decide from support, update at most once per existing path."""
    _integer(frame_id, 'frame_id')
    if branch_radius is not None:
        _finite_scalar(branch_radius, 'branch_radius', positive=True)
    _integer(tracker.C_T, 'C_T')
    _integer(tracker.next_path_id, 'next_path_id', 1)
    if tracker.T is None and tracker.C_T != 0:
        raise ValueError('single tracker must have C_T=0')
    if tracker.T is not None and (not config.dual_enabled or tracker.C_T >= config.K):
        raise ValueError('invalid dual tracker configuration or counter')
    branches = {'H': tracker.H}
    if tracker.T is not None:
        branches['T'] = tracker.T
    expected = config.total_particles // len(branches)
    for b in branches.values():
        _validate_branch(b)
        if len(b.state.weights) != expected:
            raise ValueError('single/dual branches require 500/250 particles')
    path_ids = [b.path_id for b in branches.values()]
    if len(set(path_ids)) != len(path_ids) or tracker.next_path_id <= max(path_ids):
        raise ValueError('path identities must be unique and next_path_id unused')
    candidates = list(candidates)
    if any(not isinstance(c, Candidate) for c in candidates):
        raise ValueError('candidates must contain Candidate instances only')
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids) or len({type(i) for i in ids}) > 1:
        raise ValueError('candidate IDs must be unique and have one sortable type')
    motion = core.make_config(dict(motionModel='legacy_motion', dt=config.dt,
        legacyPositionSigma=config.position_sigma, legacyVelocitySigma=config.velocity_sigma,
        intensitySigma=config.amplitude_sigma))
    predicted = {}
    for role, b in branches.items():
        b = _clone(b)
        core.predict(b.state, motion, _rng(config, frame_id, b.path_id, 1))
        predicted[role] = b
    matches = {role: match_candidates(b, candidates, config, role) for role, b in predicted.items()}
    h = _clone(predicted['H'])
    t = _clone(predicted['T']) if 'T' in predicted else None
    result = StepResult(_xy(h), 'H', 'predict_only' if t is None else 'hold', matches,
        [row for match in matches.values() for row in match.support_rows],
        ess={role: _ess(b.state) for role, b in predicted.items()},
        update_counts={role: 0 for role in predicted}, predicted=predicted,
        posterior={role: _clone(b) for role, b in predicted.items()})
    count = 0
    next_id = tracker.next_path_id
    hm = matches['H']
    if hm.status == 'unique':
        _observe(h, 'H', hm, config, result)
        result.output_xy = _xy(h)
        if t is None:
            result.action = 'single_update'
            h = _regular_resample(h, 'H', frame_id, config, result)
        else:
            result.action = 'confirm_H'
            h = _sample(h, config.total_particles, 5, 'H', frame_id, config, result)
            t = None
    elif hm.status == 'none' and candidates:
        # Admission affects births only, using the unsampled current H prediction.
        births = candidates
        if branch_radius is not None:
            state = predicted['H'].state
            weights = _weights(state.weights)
            # Accurate sums avoid a directional bias at the inclusive radius boundary.
            center = [math.fsum(weights * state.particles[:, axis]) for axis in (0, 1)]
            births = [c for c in candidates if math.hypot(c.center_xy[0] - center[0],
                c.center_xy[1] - center[1]) <= branch_radius]
        top = min(births, key=lambda c: (-c.image_score, c.center_xy[1], c.center_xy[0], c.candidate_id)) if births else None
        if t is None and config.dual_enabled and top is not None:
            result.action = 'fork'
            h = _sample(h, config.total_particles//2, 3, 'H', frame_id, config, result)
            t = _clone(h)
            t.path_id = next_id
            next_id += 1
            t.state.particles[:, :2] = top.center_xy
            # One shared parent draw is copied to each branch, never mixed.
            result.parent_indices['T'] = result.parent_indices['H'].copy()
            result.parent_path_ids['T'] = result.parent_path_ids['H']
            result.resampling_u['T'] = result.resampling_u['H']
            result.unique_parent_count['T'] = result.unique_parent_count['H']
            result.output_xy = _xy(h)
        elif t is not None:
            tm = matches['T']
            if tm.status == 'unique':
                count = tracker.C_T + 1
                _observe(t, 'T', tm, config, result)
                if count >= config.K:
                    result.action = 'promote_T'
                    result.output_xy, result.output_source = _xy(t), 'T'
                    h = _sample(t, config.total_particles, 5, 'H', frame_id, config, result)
                    t, count = None, 0
                else:
                    result.action = 'update_T'
                    t = _regular_resample(t, 'T', frame_id, config, result)
            elif tm.status == 'none' and top is not None:
                result.action = 'reseed_T'
                t = _sample(t, config.total_particles//2, 4, 'T', frame_id, config, result)
                t.path_id = next_id
                next_id += 1
                t.state.particles[:, :2] = top.center_xy
    out = Tracker(h, t, count, next_id)
    result.carried = {'H': _clone(h)}
    if t is not None:
        result.carried['T'] = _clone(t)
    return out, result
