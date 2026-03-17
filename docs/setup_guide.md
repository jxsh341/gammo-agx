# Gammo AGX — Setup Guide

## Prerequisites

### Required
- Python 3.11+
- Git
- VS Code (recommended)
- NVIDIA GPU with CUDA support (RTX 3050 or better)

### Install Before Starting
1. **Node.js** (for Tauri) — https://nodejs.org (LTS version)
2. **Rust** (for Tauri) — https://rustup.rs
3. **CUDA Toolkit** — https://developer.nvidia.com/cuda-downloads

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/gammo-agx.git
cd gammo-agx
```

---

## Step 2 — Python Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Step 3 — Supabase Setup

1. Go to https://supabase.com and create a free account
2. Create a new project
3. Copy your project URL and anon key from Settings → API
4. Run the schema setup:
```bash
python scripts/setup_supabase.py
```
5. Paste the generated SQL into the Supabase SQL Editor and run it

---

## Step 4 — Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` and add:
- `SUPABASE_URL` — your project URL
- `SUPABASE_ANON_KEY` — your anon key

---

## Step 5 — Download AI Models

Create a `models/` directory and download:

**Gemma 3 4B INT4** (Loop model — ~2.5GB):
```
https://huggingface.co/bartowski/gemma-3-4b-it-GGUF
```
Download: `gemma-3-4b-it-Q4_K_M.gguf`

**DeepSeek R1 Distill Qwen 7B INT4** (Reasoning model — ~4GB):
```
https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF
```
Download: `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`

Update paths in `.env` accordingly.

---

## Step 6 — Run the System

**Terminal 1 — API Backend:**
```bash
python scripts/run_api.py
```

**Terminal 2 — Discovery Loop:**
```bash
python scripts/run_loop.py
```

**Terminal 3 — Tauri App (once Node.js + Rust installed):**
```bash
cd app
npm install
npm run tauri dev
```

---

## Verification

Open http://localhost:8000 — you should see:
```json
{
  "name": "Gammo AGX",
  "status": "operational",
  "loop_running": true
}
```

Open your Supabase dashboard — simulation records should appear in the `simulations` table within 30 seconds.

---

## Next Steps

- Read `docs/architecture.md` for the full system overview
- Read `docs/physics.md` for the physics foundations
- Open VS Code and start with `core/simulator/morris_thorne.py`
