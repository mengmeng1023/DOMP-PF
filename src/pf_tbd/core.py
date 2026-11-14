"""Conditional single-target PF with MATLAB1 continuous coordinates.

Configuration keys retain the frozen MATLAB names. Functions consuming State
(predict, update, resample and step) mutate and return that State; copy arrays
before calling when a historical snapshot is needed. Random sampling uses the
caller's NumPy Generator; matching MATLAB random streams is not a contract.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass
class State:
    particles: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray
    frame_size: tuple[int, int]
    step_index: int


def make_config(overrides=None):
    """Return validated defaults plus explicit overrides; Sigma means stddev."""
    cfg = dict(numParticles=500, dt=1, motionModel='acceleration_motion',
               legacyPositionSigma=10, legacyVelocitySigma=.001,
               accelerationSigma=2, accelerationPositionSigma=1,
               intensitySigma=.01, initialVelocityX=[-10, 10], initialVelocityY=[-5, 5],
               sigmaBlur=20, sigmaNoise=100, patchRadius=1, essThresholdFraction=.5)
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides)-set(cfg):
            raise ValueError('Configuration must contain only known configuration fields')
        cfg.update(overrides)
    if cfg['motionModel'] not in ('legacy_motion', 'acceleration_motion'):
        raise ValueError('Unknown motionModel')
    for key in ('numParticles', 'dt', 'legacyPositionSigma', 'legacyVelocitySigma',
                'accelerationSigma', 'accelerationPositionSigma', 'intensitySigma',
                'sigmaBlur', 'sigmaNoise', 'patchRadius', 'essThresholdFraction'):
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f'{key} must be a finite real scalar')
        if value < 0 or (key in ('numParticles', 'dt', 'sigmaBlur', 'sigmaNoise') and value == 0):
            raise ValueError(f'{key} is outside its valid range')
    for key in ('numParticles', 'patchRadius'):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f'{key} must be an integer')
        cfg[key] = int(cfg[key])
    if cfg['essThresholdFraction'] > 1:
        raise ValueError('essThresholdFraction must be in [0, 1]')
    for key in ('initialVelocityX', 'initialVelocityY'):
        interval = np.asarray(cfg[key])
        if interval.shape != (2,) or not np.issubdtype(interval.dtype, np.number) or not np.isrealobj(interval):
            raise ValueError(f'{key} must be a finite ordered pair')
        if not np.all(np.isfinite(interval)) or interval[1] < interval[0]:
            raise ValueError(f'{key} must be a finite ordered pair')
        cfg[key] = interval.astype(np.float64).tolist()
    return cfg


def _observation(value):
    frame = np.asarray(value)
    if (frame.ndim != 2 or frame.size == 0 or not np.issubdtype(frame.dtype, np.number)
            or not np.isrealobj(frame) or not np.all(np.isfinite(frame))):
        raise ValueError('Observation must be a nonempty finite real numeric 2D array')
    return frame


def _readout_xy(particles):
    # All valid particle positions are positive: MATLAB half-away-from-zero.
    return np.floor(particles[:, :2]+.5).astype(np.intp)


def initialize(observation, initial_xy, cfg, rng):
    """Initialize once at supplied MATLAB1 x,y; do not update this frame again."""
    frame = _observation(observation)
    center = np.asarray(initial_xy, dtype=np.float64)
    if (center.shape != (2,) or not np.all(np.isfinite(center))
            or np.any(center < 1) or center[0] > frame.shape[1] or center[1] > frame.shape[0]):
        raise ValueError('Initial x,y must be finite and inside the observation')
    n = cfg['numParticles']
    particles = np.zeros((n, 5), dtype=np.float64)
    particles[:, :2] = center
    for column, key in ((2, 'initialVelocityX'), (3, 'initialVelocityY')):
        low, high = cfg[key]
        particles[:, column] = low+(high-low)*rng.random(n)
    x, y = _readout_xy(particles[:1])[0]
    particles[:, 4] = float(frame[y-1, x-1])
    return State(particles, np.full(n, 1/n), np.full(n, -np.log(n)), tuple(frame.shape), 1)


def predict(state, cfg, rng):
    """Advance continuous particles in place, clipping only their positions."""
    particles = state.particles
    n = len(particles)
    dt = cfg['dt']
    if cfg['motionModel'] == 'acceleration_motion':
        acceleration = cfg['accelerationSigma']*rng.standard_normal((n, 2))
        particles[:, :2] += (dt*particles[:, 2:4] + .5*dt*dt*acceleration
                             + cfg['accelerationPositionSigma']*rng.standard_normal((n, 2)))
        particles[:, 2:4] += dt*acceleration
    elif cfg['motionModel'] == 'legacy_motion':
        particles[:, :2] += dt*particles[:, 2:4]+cfg['legacyPositionSigma']*rng.standard_normal((n, 2))
        particles[:, 2:4] += cfg['legacyVelocitySigma']*rng.standard_normal((n, 2))
    else:
        raise ValueError('Unknown motionModel')
    particles[:, 0] = np.clip(particles[:, 0], 1, state.frame_size[1])
    particles[:, 1] = np.clip(particles[:, 1], 1, state.frame_size[0])
    particles[:, 4] += cfg['intensitySigma']*rng.standard_normal(n)
    if not np.all(np.isfinite(particles)):
        raise ValueError('Prediction produced a nonfinite particle')
    state.step_index += 1
    return state


def update(state, observation, cfg):
    """Accumulate patch log likelihood ratios onto the authoritative log prior."""
    frame = _observation(observation)
    if frame.shape != state.frame_size:
        raise ValueError('Observation dimensions differ from the initialization frame')
    if not np.all(np.isfinite(state.particles)):
        raise ValueError('Particles must be finite before observation update')
    centers = _readout_xy(state.particles)
    height, width = state.frame_size
    log_likelihood = np.zeros(len(state.particles), dtype=np.float64)
    radius = cfg['patchRadius']
    for dy in range(-radius, radius+1):
        y = centers[:, 1]+dy
        for dx in range(-radius, radius+1):
            x = centers[:, 0]+dx
            valid = (x >= 1) & (x <= width) & (y >= 1) & (y <= height)
            signal = state.particles[valid, 4]*np.exp(-(dx*dx+dy*dy)/(2*cfg['sigmaBlur']**2))
            observed = frame[y[valid]-1, x[valid]-1].astype(np.float64)
            log_likelihood[valid] += signal*(2*observed-signal)/(2*cfg['sigmaNoise']**2)
    posterior = state.log_weights+log_likelihood
    shift = np.max(posterior)
    if not np.isfinite(shift) or np.any(np.isnan(posterior)) or np.any(posterior == np.inf):
        raise ValueError('At least one finite posterior weight is required')
    shifted = posterior-shift
    with np.errstate(under='ignore'):
        mass = np.exp(shifted)
    normalizer = np.sum(mass)
    state.log_weights = shifted-np.log(normalizer)
    state.weights = mass/normalizer
    return state, log_likelihood


def _normalized_weights(state):
    weights = state.weights
    total = np.sum(weights)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError('Weights must be finite, nonnegative and have positive total mass')
    return weights/total


def estimate(state):
    """Return weighted mean and population covariance before resampling."""
    weights = _normalized_weights(state)
    average = weights@state.particles
    centered = state.particles-average
    covariance = centered.T@(centered*weights[:, None])
    return average, covariance


def resample(state, rng):
    """Systematic resampling in place; CDF equality selects the next interval."""
    n = len(state.particles)
    cdf = np.cumsum(_normalized_weights(state))
    cdf[-1] = 1.
    thresholds = (rng.random()+np.arange(n))/n
    indices = np.minimum(np.searchsorted(cdf, thresholds, side='right'), n-1)
    state.particles = state.particles[indices].copy()
    state.weights = np.full(n, 1/n)
    state.log_weights = np.full(n, -np.log(n))
    return state


def step(state, observation, cfg, rng):
    """Predict, update, estimate, then conditionally resample; mutate State."""
    predict(state, cfg, rng)
    update(state, observation, cfg)
    average, covariance = estimate(state)
    ess = float(1/np.sum(state.weights**2))
    resampled = bool(ess < cfg['essThresholdFraction']*len(state.particles))
    diagnostics = dict(ess=ess, resampled=resampled, covariance=covariance)
    if resampled:
        resample(state, rng)
    return state, average, diagnostics