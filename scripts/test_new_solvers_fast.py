"""
Fast syntax + logic check — bypasses JAX JIT to verify code structure.
Uses numpy instead of JAX for fast validation.
"""
import sys
sys.path.insert(0, 'C:\\Users\\user\\gammo-agx')

# Patch jax before importing solvers
import numpy as np

# Monkey-patch JAX imports so we can test without JIT warmup
import types

jax_mock = types.ModuleType('jax')
jax_numpy_mock = types.ModuleType('jax.numpy')

# Copy ONLY required numpy functions to avoid triggering numpy.testing wmi crashes
_needed = ['maximum', 'minimum', 'linspace', 'sqrt', 'tanh', 'cosh', 'any', 'sum', 'clip', 'exp']
for name in _needed:
    if hasattr(np, name):
        setattr(jax_numpy_mock, name, getattr(np, name))
jax_numpy_mock.pi = np.pi
jax_numpy_mock.sum = np.sum
jax_numpy_mock.clip = np.clip
jax_numpy_mock.exp = np.exp

def noop_jit(f): return f
def noop_vmap(f): return np.vectorize(f)
jax_mock.jit = noop_jit
jax_mock.vmap = noop_vmap
jax_mock.numpy = jax_numpy_mock

sys.modules['jax'] = jax_mock
sys.modules['jax.numpy'] = jax_numpy_mock

print("=== Syntax/Logic Test: Krasnikov Solver ===")
from core.simulator.krasnikov import solve, KrashnikovParams
r = solve(KrashnikovParams(tube_radius=0.8, length=3.0, shell_thickness=0.3, boost_factor=0.5))
m = r['metrics']
print(f"  stability_score   = {m['stability_score']:.3f}")
print(f"  ford_roman_status = {m['ford_roman_status']}")
print(f"  geometry_class    = {m['geometry_class']}")
print(f"  energy_req        = {m['energy_requirement']:.4e}")
print(f"  casimir_gap_oom   = {m['casimir_gap_oom']:.1f}")
print(f"  null_energy_viol  = {m['null_energy_violated']}")
print(f"  bssn_stable       = {m['bssn_stable']}")
print("  PASS")

print()
print("=== Syntax/Logic Test: Schwarzschild Solver ===")
from core.simulator.schwarzschild import solve as schw_solve, SchwarzschildParams
r2 = schw_solve(SchwarzschildParams(mass=1.2, spin=0.0, charge=0.0, observer_distance=4.0))
m2 = r2['metrics']
print(f"  schwarzschild_rs  = {m2['schwarzschild_radius']:.3f}")
print(f"  isco_radius       = {m2['isco_radius']:.3f}")
print(f"  hawking_temp      = {m2['hawking_temperature']:.4e}")
print(f"  stability_score   = {m2['stability_score']:.3f}")
print(f"  geometry_class    = {m2['geometry_class']}")
print("  PASS")

print()
print("=== SymPy Validator: valid Krasnikov ===")
from core.symbolic.metric_validator import filter_configuration
ok, reason = filter_configuration({
    'geometry_type': 'krasnikov',
    'parameters': {
        'tube_radius': 0.8, 'length': 3.0,
        'shell_thickness': 0.3, 'boost_factor': 0.5
    }
})
print(f"  should_simulate = {ok}")
print(f"  reason          = {reason}")
assert ok, f"Expected pass: {reason}"
print("  PASS")

print()
print("=== SymPy Validator: boost > 1 (should reject) ===")
ok2, reason2 = filter_configuration({
    'geometry_type': 'krasnikov',
    'parameters': {
        'tube_radius': 0.8, 'length': 3.0,
        'shell_thickness': 0.3, 'boost_factor': 1.5
    }
})
print(f"  should_simulate = {ok2}")
print(f"  reason          = {reason2}")
assert not ok2, "Expected fail for boost > 1"
print("  PASS")

print()
print("=== Discovery Loop Import Check ===")
from loop.discovery_loop import DiscoveryLoop
dloop = DiscoveryLoop()
print(f"  DiscoveryLoop created OK")
print()
print("=== ALL TESTS PASSED ===")
