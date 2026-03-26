# Gammo AGX — AI Coder Handoff Document
## Complete Project State & Remaining Build Tasks

---

## What Gammo AGX Is

**Gammo AGX** is the world's first autonomous hybrid research engine for exotic spacetime physics. It combines a JAX-accelerated numerical relativity simulator, a closed-loop AI scientific discovery system, a multi-component AI architecture, and a human query interface.

**Two operating modes:**
- **Autonomous Mode** — runs 24/7 without human input, generating spacetime configurations, validating them symbolically, simulating them numerically, scoring and storing every result, generating AI hypotheses
- **Query Mode** — researcher queries the live system in natural language at any time, loop never pauses

**Target user:** Physics researchers, NASA/ESA scientists, the general public via a Windows desktop app

**Demo date:** July 25, 2025 — YC Startup School

---

## Repository

- **GitHub:** https://github.com/jxsh341/gammo-agx
- **Local path:** `C:\Users\user\gammo-agx`
- **Branch:** main

---

## Tech Stack

| Layer | Technology |
|---|---|
| Physics Engine | JAX + JIT (CPU currently, CUDA attempted) |
| NR Evolution | BSSN formulation (scaffolded) |
| Diff Schemes | WENO-Z (in morris_thorne.py) |
| Symbolic Math | SymPy |
| Loop AI Model | Gemma 3 4B INT4 via llama-cpp-python |
| LLM Runtime | llama-cpp-python 0.3.16 |
| Reasoning Pipeline | Custom multi-pass (8 steps) |
| Knowledge Store | Supabase (PostgreSQL + pgvector) |
| Vector Search | Supabase pgvector (match_simulations RPC) |
| API Backend | FastAPI + Uvicorn |
| Desktop App | Tauri (NOT YET BUILT) |
| Visualizer | Three.js + Plotly/Dash (standalone HTML built, not connected to API) |
| Dataset Pipeline | Not yet built |
| Fine-tuning | Not yet done |

---

## Environment Setup

```bash
# Navigate to project
cd C:\Users\user\gammo-agx

# Set Python path (required every new terminal)
$env:PYTHONPATH = "C:\Users\user\gammo-agx"

# Activate venv
venv\Scripts\activate

# Start API
python scripts/run_api.py
```

**.env file** (already configured):
```
SUPABASE_URL=<already set>
SUPABASE_ANON_KEY=<already set>
SUPABASE_SERVICE_KEY=<already set>
GEMMA_MODEL_PATH=models/google_gemma-3-4b-it-Q4_K_M.gguf
DEEPSEEK_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf
API_HOST=0.0.0.0
API_PORT=8000
LOOP_INTERVAL_SECONDS=30
LOOP_AUTO_START=True
JAX_ENABLE_X64=True
JAX_PLATFORM_NAME=gpu
```

**Model files in `models/`:**
- `google_gemma-3-4b-it-Q4_K_M.gguf` — 2.49GB — WORKING
- `DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf` — downloaded but FAILS to load (RAM issue, see Known Issues)

---

## What Is Currently Working

### Physics Engine
- `core/simulator/morris_thorne.py` — JAX-accelerated Morris-Thorne wormhole solver. Solves Einstein field equations. Returns geometry arrays and physics metrics. FULLY WORKING.
- `core/symbolic/metric_validator.py` — SymPy symbolic validation. Checks parameter bounds, flaring-out condition, asymptotic flatness. Ford-Roman check is a SOFT filter (only rejects factor > 50). FULLY WORKING.
- `core/quantum/casimir.py` — Casimir energy engine. Parallel plate, spherical, cylindrical, toroidal geometries. Gap analysis. FULLY WORKING.
- `core/quantum/ford_roman.py` — Ford-Roman quantum inequality checker. Multi-timescale checks. FULLY WORKING.
- `core/descriptors/extractor.py` — 64-dimensional descriptor vector extractor. Covers curvature (dims 0-15), energy (16-31), stability (32-47), quantum (48-63). FULLY WORKING.

### AI System
- `ai/models/gemma_runner.py` — Gemma 3 4B runner. Loaded via llama-cpp-python. Has retry logic for Windows WinError. WORKING but occasionally throws `[WinError -529697949]` — fallback handles it.
- `ai/models/deepseek_runner.py` — DeepSeek R1 runner. BUILT but model FAILS TO LOAD due to RAM buffer issue (see Known Issues).
- `ai/reasoning_pipeline.py` — 8-step multi-pass reasoning pipeline. Retrieval → Pass1 → SymPy tool → Pass2 critique → Self-consistency (3 samples) → JAX tool → Pass3 final → Evidential uncertainty. FULLY WORKING. Takes 2-3 minutes per run. Triggered for novel/high-confidence discoveries.

### Discovery Loop
- `loop/discovery_loop.py` — 7-step autonomous discovery loop. All steps wired with real physics. FULLY WORKING.
  - Step 1: Random config generation (TODO: replace with AI-guided generation)
  - Step 2: SymPy symbolic validation
  - Step 3: JAX Morris-Thorne simulation
  - Step 4: Constraint evaluation (real Casimir gap computed)
  - Step 5: Gemma 3 4B hypothesis generation
  - Step 6: Supabase write (all physics fields + descriptor vector)
  - Step 7: Multi-pass pipeline for novel/high-confidence discoveries

### Knowledge Store
- Supabase project live with tables: `simulations`, `literature_embeddings`, `discovered_metrics`, `hypotheses`, `loop_state`
- pgvector enabled, `match_simulations` RPC function deployed
- 300+ real physics simulation records accumulated
- Descriptor vectors stored for semantic search

### API
- `api/main.py` — FastAPI app with lifespan, CORS, auto-starts discovery loop
- `api/state.py` — shared discovery_loop instance (fixes circular import)
- `api/routes/loop.py` — loop status, start, stop, feed
- `api/routes/query.py` — natural language query, structured query, stats, similar, stable, ford-roman
- All routes working. API docs at http://localhost:8000/docs

### Store
- `store/supabase_client.py` — singleton Supabase client
- `store/writer.py` — writes simulation records to Supabase
- `store/query.py` — structured queries over simulation records
- `store/search.py` — semantic search via pgvector (search_by_vector, search_by_params, search_by_natural_language, find_most_stable, find_ford_roman_satisfied, find_similar_to_record)

---

## Known Issues

### 1. DeepSeek R1 RAM Issue (CRITICAL)
**Error:** `ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 3121348608`
**Cause:** llama.cpp tries to allocate 3.1GB contiguous RAM for CPU_REPACK buffer. 16GB RAM machine can't provide contiguous block with everything else running.
**Current state:** DeepSeek runner is built but model fails to load. The multi-pass pipeline only uses Gemma.
**Solution options:**
- Option A: Use Anthropic API for deep reasoning (replace deepseek_runner.py calls with anthropic SDK calls to claude-sonnet-4-20250514)
- Option B: Try IQ2_XS quantization (smallest available, ~2.3GB)
- Option C: Accept Gemma-only architecture and focus on fine-tuning quality

### 2. Gemma WinError
**Error:** `[WinError -529697949] Windows Error 0xe06d7363`
**Cause:** llama.cpp C++ exception on Windows for certain long prompts
**Current state:** Retry logic added in gemma_runner.py — retries with shorter prompt (500 chars) and max_tokens=150
**Status:** Partially fixed. Still occasionally fails on very long prompts.

### 3. Alcubierre/Krasnikov Solvers Not Built
**Status:** Scaffolded as empty files. Only Morris-Thorne is implemented.
**Files needed:**
- `core/simulator/alcubierre.py`
- `core/simulator/krasnikov.py`
- `core/simulator/schwarzschild.py`

### 4. Config Generation Still Random
**Status:** Step 1 of discovery loop uses random.uniform for parameters. Should be replaced with AI-guided generation (Bayesian optimization via Optuna, or Gemma-suggested parameters based on knowledge store patterns).

### 5. CUDA Not Active
**Status:** JAX running on CPU. `jax.devices()` returns `[CpuDevice(id=0)]`. RTX 3050 has 4GB VRAM. CUDA toolkit may need reinstall. Not blocking — CPU is adequate for development.

---

## What Needs To Be Built

### Priority 1 — Tauri Windows Desktop App (CRITICAL for demo)

**Goal:** Wrap everything into a single .exe installer that runs on any Windows machine.

**Architecture:**
- Python FastAPI backend runs as a sidecar process
- Tauri wraps the web frontend (Plotly/Dash visualizer) in a native window
- Single installer built with `tauri build`

**What exists:**
- `app/src-tauri/src/main.rs` — basic Tauri entry point
- `app/src-tauri/Cargo.toml` — Rust dependencies
- `app/src-tauri/tauri.conf.json` — Tauri config (devPath points to localhost:8000)
- `app/package.json` — npm config

**What needs to be built:**
1. Install prerequisites: Node.js (https://nodejs.org LTS) and Rust (https://rustup.rs)
2. Update `tauri.conf.json` to point devPath at localhost:8000
3. Build Tauri sidecar that starts the Python API automatically on app launch
4. Create Windows installer with `npm run tauri build`
5. Test the .exe on a clean Windows machine

**Tauri sidecar approach for Python:**
In `tauri.conf.json`, add sidecar configuration:
```json
"bundle": {
  "externalBin": ["binaries/gammo-agx-api"]
}
```
Use PyInstaller to bundle the Python API into a single executable:
```bash
pip install pyinstaller
pyinstaller --onefile scripts/run_api.py --name gammo-agx-api
```
Then Tauri launches it automatically.

**Alternative simpler approach:** Skip PyInstaller, just check if Python is installed and launch `python scripts/run_api.py` from Rust sidecar code. Less portable but faster to build.

### Priority 2 — Connect Visualizer to Live API

**What exists:** `gammo_agx_v2_visualizer.html` — fully working standalone Three.js visualizer with 4 geometries, physics panels, geodesic tracer, heatmap. Built and tested.

**What needs to be done:**
1. Add a live feed section to the visualizer that polls `GET /loop/status` every 5 seconds
2. Add a knowledge store section that polls `GET /query/stats` every 30 seconds
3. Add a query panel that calls `POST /query/natural`
4. Show the record count ticking up in real time
5. Show the latest hypothesis from the running loop

**API endpoints to connect:**
- `GET http://localhost:8000/loop/status` — loop iteration, total simulations, novel discoveries, last hypothesis
- `GET http://localhost:8000/query/stats` — record counts by geometry
- `POST http://localhost:8000/query/natural` — natural language search
- `GET http://localhost:8000/query/stable` — most stable configs
- `GET http://localhost:8000/query/ford-roman` — Ford-Roman satisfied configs

### Priority 3 — Alcubierre Warp Drive Solver

**File:** `core/simulator/alcubierre.py`

**Physics:** Alcubierre (1994) metric:
```
ds² = -dt² + (dx - v_s f(r_s) dt)² + dy² + dz²
```
Where `f(r_s)` is the shape function, `v_s` is warp speed, `r_s` is distance from bubble center.

**Shape function:** `f(r_s) = (tanh(sigma*(r_s + R)) - tanh(sigma*(r_s - R))) / (2*tanh(sigma*R))`

**Key metrics to compute:**
- Energy requirement (negative, proportional to v_s²)
- Bubble stability
- Ford-Roman violation factor
- Casimir gap

**Wire into loop:** Add `alcubierre` as a second geometry type in `loop/discovery_loop.py` step 1 and step 3.

### Priority 4 — Krasnikov Tube Solver

**File:** `core/simulator/krasnikov.py`

**Physics:** Krasnikov (1995) — a tube of modified spacetime that enables effective FTL travel.

**Key parameters:** tube radius, length, shell thickness, causal boost factor

**Similar structure to Morris-Thorne solver.**

### Priority 5 — Dataset Pipeline and Fine-tuning

**Goal:** Fine-tune Gemma 3 4B on physics literature + Gammo AGX simulation outputs to dramatically improve hypothesis quality.

**Dataset sources (all free):**
1. `https://huggingface.co/datasets/camel-ai/physics` — physics Q&A pairs
2. `https://huggingface.co/datasets/allenai/sciq` — science exam questions
3. `https://huggingface.co/datasets/TIGER-Lab/MathInstruct` — math reasoning chains
4. `https://huggingface.co/datasets/open-r1/OpenR1-Math-220k` — R1 reasoning traces
5. arXiv gr-qc section — use arXiv API: `http://export.arxiv.org/api/query?search_query=cat:gr-qc&max_results=1000`
6. Gammo AGX Supabase records — export via `GET /dataset/export`

**Fine-tuning approach:**
- Method: LoRA r=16, alpha=32 (runs on 4GB VRAM)
- Library: HuggingFace PEFT + transformers
- Base model: `google/gemma-3-4b-it` (HuggingFace format, not GGUF)
- Training data format: instruction-response pairs
- After fine-tuning: convert to GGUF with llama.cpp quantize tool

**Script to build:** `scripts/download_datasets.py` and `ai/finetuning/lora_trainer.py`

### Priority 6 — AI-Guided Configuration Generation

**Current state:** Step 1 of discovery loop uses `random.uniform` — completely random.

**Goal:** Replace with intelligent generation that learns from accumulated simulation data.

**Approach 1 — Bayesian optimization (easiest):**
```python
import optuna

def objective(trial):
    b0 = trial.suggest_float("throat_radius", 0.3, 3.0)
    rho = trial.suggest_float("exotic_density", 0.01, 1.0)
    tide = trial.suggest_float("tidal_force", 0.01, 0.99)
    phi0 = trial.suggest_float("redshift_factor", 0.01, 1.0)
    # Run simulation, return stability score
    return stability_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

**Approach 2 — Gemma-guided (more interesting):**
Ask Gemma to suggest parameter ranges based on the best-performing configurations in the knowledge store. Feed it the top 10 most stable configs and ask it to propose a new configuration that might do even better.

### Priority 7 — Novel Metric Discovery

**Goal:** Have Gemma propose entirely new spacetime metric ansatzes beyond Morris-Thorne.

**What exists:** `ai/models/deepseek_runner.py` has `propose_novel_metric()` function — but DeepSeek doesn't load. Needs to be ported to use Gemma or Anthropic API.

**Implementation:** Add a `propose_novel_metric()` function to `ai/models/gemma_runner.py` that:
1. Pulls top 10 most stable configs from Supabase
2. Asks Gemma to identify patterns
3. Asks Gemma to propose a novel modification to the MT shape function
4. Passes proposal through SymPy for symbolic validation
5. If valid, runs JAX simulation
6. Stores result in `discovered_metrics` table

### Priority 8 — Update YC Application

**Application form fields already written in:** `YC_Startup_School_Application.docx`

**Update the "things you've built" section to include Gammo AGX with:**
- What it is (1 sentence)
- The live demo link or GitHub URL
- Key metrics: 300+ simulations, real JAX physics, Gemma AI hypotheses, semantic search
- The multi-pass reasoning pipeline (novel architecture)

---

## Project Structure Reference

```
gammo-agx/
├── core/
│   ├── simulator/
│   │   ├── morris_thorne.py    ✅ COMPLETE
│   │   ├── alcubierre.py       ❌ EMPTY - needs implementation
│   │   ├── krasnikov.py        ❌ EMPTY - needs implementation
│   │   └── schwarzschild.py    ❌ EMPTY - needs implementation
│   ├── symbolic/
│   │   └── metric_validator.py ✅ COMPLETE
│   ├── quantum/
│   │   ├── casimir.py          ✅ COMPLETE
│   │   └── ford_roman.py       ✅ COMPLETE
│   └── descriptors/
│       └── extractor.py        ✅ COMPLETE (64-dim vectors)
├── ai/
│   ├── models/
│   │   ├── gemma_runner.py     ✅ WORKING (with retry logic)
│   │   └── deepseek_runner.py  ⚠️ BUILT BUT MODEL FAILS TO LOAD
│   ├── reasoning_pipeline.py   ✅ COMPLETE (8-step multi-pass)
│   └── finetuning/             ❌ NOT BUILT
├── loop/
│   └── discovery_loop.py       ✅ COMPLETE (all 7 steps)
├── store/
│   ├── supabase_client.py      ✅ COMPLETE
│   ├── writer.py               ✅ COMPLETE
│   ├── query.py                ✅ COMPLETE
│   └── search.py               ✅ COMPLETE (pgvector semantic search)
├── api/
│   ├── main.py                 ✅ COMPLETE
│   ├── state.py                ✅ COMPLETE
│   └── routes/
│       ├── loop.py             ✅ COMPLETE
│       └── query.py            ✅ COMPLETE
├── app/                        ⚠️ SCAFFOLDED - Tauri NOT built
├── models/
│   ├── google_gemma-3-4b-it-Q4_K_M.gguf    ✅ WORKING
│   └── DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf  ⚠️ FAILS TO LOAD
├── scripts/
│   ├── run_api.py              ✅ COMPLETE
│   ├── run_loop.py             ✅ COMPLETE
│   └── setup_supabase.py       ✅ COMPLETE
├── config/
│   └── settings.py             ✅ COMPLETE
├── .env                        ✅ CONFIGURED
├── requirements.txt            ✅ COMPLETE
└── start.bat                   ✅ COMPLETE
```

---

## Supabase Schema

All tables in the `public` schema:

### simulations (main table)
```sql
id, created_at, geometry_type, parameters (jsonb),
descriptor_vector (vector(64)), stability_score,
energy_requirement, casimir_gap_oom, ford_roman_status,
null_energy_violated, constraint_error, traversal_time,
bssn_stable, hypothesis, hypothesis_confidence,
uncertainty_type, novelty_flag, novelty_score,
geometry_class, simulation_duration_ms, model_used, loop_iteration
```

### literature_embeddings
```sql
id, created_at, title, authors, year, arxiv_id,
abstract, content, embedding (vector(384)), category
```

### discovered_metrics
```sql
id, created_at, metric_name, metric_ansatz,
energy_req, stability, novelty_score, hypothesis,
confidence, validated
```

### hypotheses
```sql
id, created_at, geometry_type, hypothesis_text,
confidence, uncertainty_type, novelty_flag,
falsifiability, simulation_id
```

### RPC Functions
```sql
match_simulations(query_vector vector(64), match_count int, min_stability float)
-- Returns similar simulations ordered by cosine distance
```

---

## API Endpoints Reference

```
GET  /                    -- system status
GET  /health              -- health check
GET  /docs                -- interactive API docs

GET  /loop/status         -- loop iteration, simulations, novel_discoveries, last_hypothesis
POST /loop/start          -- start the discovery loop
POST /loop/stop           -- stop the discovery loop
GET  /loop/feed           -- recent 20 simulations + loop status

POST /query/natural       -- natural language search {"query": "...", "limit": 10}
POST /query/structured    -- structured filter query
GET  /query/stats         -- record counts by geometry
POST /query/similar       -- find similar by parameters
GET  /query/stable        -- most stable configurations
GET  /query/ford-roman    -- Ford-Roman satisfied configurations
```

---

## The Demo Script (July 25)

**5 minutes, live system, nothing pre-recorded:**

1. **0:00** — Open dashboard. Show knowledge store: 1000+ records. Show loop running live.
2. **1:00** — Show a novel discovery in Supabase with full multi-pass hypothesis.
3. **2:00** — Type natural language query: "find stable wormholes where Ford-Roman is satisfied"
4. **3:00** — Inject custom config. Show 30-second turnaround to full hypothesis.
5. **4:00** — Show the multi-pass reasoning trace (8 steps, SymPy + JAX tool calls).
6. **5:00** — Back to loop feed. Record count ticking up. System never stops.

---

## Remaining Build Priority Order

1. **Fix DeepSeek RAM issue OR switch to Anthropic API** — 1-2 hours
2. **Connect visualizer to live API** — 2-3 hours
3. **Tauri desktop app** — 1-2 days
4. **Alcubierre solver** — 3-4 hours
5. **Krasnikov solver** — 2-3 hours
6. **Dataset download pipeline** — 2 hours
7. **Gemma fine-tuning** — overnight run
8. **AI-guided config generation** — 3-4 hours
9. **Novel metric discovery** — 4-5 hours
10. **Update YC application** — 1 hour

---

## Physics Reference

### Morris-Thorne Metric
```
ds² = -e^(2Φ(r)) dt² + dl² + r²(l)(dθ² + sin²θ dφ²)
Shape function: b(r) = b₀²/r
Embedding: z(r) = sqrt(r² - b₀²)
```

### Ford-Roman Quantum Inequality
```
∫ ρ(τ) f(τ) dτ ≥ -C_FR / t₀⁴
where C_FR = 3 / (32π²) ≈ 0.0095
```

### Casimir Energy Density (parallel plates)
```
ρ = -π²ℏc / (720 d⁴)
At d=1nm: ρ ≈ -4.33×10⁸ J/m³
```

### Stability Score Formula
```python
stability = max(0, min(1,
    (1 / (tidal_force + 0.1)) * exotic_density * 0.5 +
    (1 - redshift_factor) * 0.3
))
```

---

## Notes for the Incoming AI

1. **Always set PYTHONPATH** before running anything: `$env:PYTHONPATH = "C:\Users\user\gammo-agx"`

2. **The Ford-Roman filter is intentionally soft** — factor > 50 only. Most random configs violate Ford-Roman but are still physically interesting to simulate.

3. **Gemma generates real hypotheses** but occasionally throws WinError on Windows. The fallback template kicks in automatically.

4. **The multi-pass pipeline takes 2-3 minutes** — this is intentional and only runs for novel/high-confidence discoveries, not every cycle.

5. **Supabase has real data** — 300+ physics simulation records with descriptor vectors. The semantic search actually works.

6. **The visualizer HTML file** (`gammo_agx_v2_visualizer.html`) is a complete standalone app but not connected to the live API yet. It uses hardcoded mock physics.

7. **DeepSeek R1 Q3_K_M.gguf is in models/** but fails to load due to RAM. Consider Anthropic API as replacement for deep reasoning.

8. **All physics is real** — JAX solves actual Einstein field equations, SymPy derives real tensor components, Casimir values are from the actual formula.

9. **The discovery loop runs every 30 seconds** — `LOOP_INTERVAL_SECONDS=30` in .env. Can be reduced to 10 for faster accumulation.

10. **GitHub is current** — always `git pull` before making changes.

---

*Document created: March 22, 2026*
*Project: Gammo AGX v0.2*
*Status: Core system operational, demo prep in progress*
