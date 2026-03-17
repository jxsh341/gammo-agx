# Gammo AGX

**Autonomous Hybrid Research Engine for Exotic Spacetime Physics**

> The first closed-loop AI system that continuously explores exotic spacetime geometries, builds the world's first structured exotic spacetime dataset, and answers researcher queries in real time.

---

## What It Is

Gammo AGX combines:
- A JAX-accelerated numerical relativity engine (BSSN + WENO-Z)
- A closed-loop AI scientific discovery system
- A multi-component AI architecture (Gemma 3 4B + DeepSeek R1 7B)
- A Supabase-powered knowledge store with real-time subscriptions
- A Tauri-wrapped Windows desktop application

It operates in two simultaneous modes:
- **Autonomous Mode** — runs 24/7, generating and simulating spacetime configurations continuously
- **Query Mode** — researchers query the live system in natural language at any time

---

## Geometries Supported

| Geometry | Reference | Status |
|---|---|---|
| Morris-Thorne Wormhole | Morris & Thorne (1988) | ✅ Phase 1 |
| Alcubierre Warp Drive | Alcubierre (1994) | 🔄 Phase 3 |
| Krasnikov Tube | Krasnikov (1995) | 🔄 Phase 3 |
| Schwarzschild Black Hole | Schwarzschild (1916) | 🔄 Phase 3 |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/gammo-agx.git
cd gammo-agx

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your Supabase URL and key to .env

# 5. Run the API backend
python scripts/run_api.py

# 6. Run the discovery loop
python scripts/run_loop.py
```

---

## Architecture

```
gammo-agx/
├── core/          # Physics engine (JAX, BSSN, WENO-Z, Casimir)
├── ai/            # Multi-system AI (Gemma 3 + DeepSeek R1, LoRA, retrieval)
├── loop/          # Autonomous discovery loop
├── store/         # Supabase knowledge store
├── graph/         # GNN spacetime representation
├── api/           # FastAPI backend
├── app/           # Tauri Windows desktop application
├── visualizer/    # Three.js + Plotly visualization
├── datasets/      # Dataset pipeline (arXiv, HuggingFace, synthetic)
├── config/        # Configuration
├── tests/         # Test suite
└── scripts/       # Utility scripts
```

---

## Industries

Theoretical Physics · Computational Science · Advanced Propulsion · Deep Tech · AI/ML

---

## License

Apache 2.0 — open source, no restrictions.

---

*Built for the final frontier.*
