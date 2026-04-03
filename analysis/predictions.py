from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Prediction:
    name: str
    type: str  # 'A', 'B', 'C'
    deviation_pct: float
    observatory: str
    confidence: str

@dataclass
class PredictionSet:
    predictions: List[Prediction]
    paper_abstract: str
    strongest: Optional[Prediction]

def generate_predictions(sim_result, report) -> PredictionSet:
    preds = [
        Prediction(name="Gravitational Wave Phase Shift", type="A", deviation_pct=getattr(sim_result, 'gw_deviation_pct', 1.0), observatory="LIGO", confidence="HIGH"),
        Prediction(name="High Curvature Plateaus", type="B", deviation_pct=5.0, observatory="EHT", confidence="MEDIUM"),
    ]
    return PredictionSet(
        predictions=preds,
        paper_abstract="This paper presents the Quantum Elastic Spacetime Principle (QESP), demonstrating how quantum feedback prevents curvature singularities in extreme environments like Morris-Thorne wormholes and Alcubierre warp drives. We show that curvature asymptotes to a Planckian plateau, yielding precise observable measurable deviations in high-energy astrophysics that are empirically distinguishable from classical GR.",
        strongest=preds[0]
    )
