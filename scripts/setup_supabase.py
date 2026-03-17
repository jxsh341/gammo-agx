"""
Gammo AGX — Supabase Setup Script
Run once to create all required tables and enable pgvector.
Run with: python scripts/setup_supabase.py
"""

from store.supabase_client import get_client
from loguru import logger


def setup():
    client = get_client()
    logger.info("Setting up Supabase schema for Gammo AGX...")

    # SQL to create all tables
    # Run this in the Supabase SQL editor if the Python client
    # doesn't have sufficient permissions

    sql = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Simulations table
    CREATE TABLE IF NOT EXISTS simulations (
        id                    UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at            TIMESTAMPTZ DEFAULT NOW(),
        geometry_type         TEXT NOT NULL,
        parameters            JSONB NOT NULL DEFAULT '{}',
        descriptor_vector     VECTOR(64),
        stability_score       FLOAT8,
        energy_requirement    FLOAT8,
        casimir_gap_oom       FLOAT8,
        ford_roman_status     TEXT,
        null_energy_violated  BOOLEAN,
        constraint_error      FLOAT8,
        traversal_time        FLOAT8,
        bssn_stable           BOOLEAN,
        hypothesis            TEXT,
        hypothesis_confidence FLOAT8,
        uncertainty_type      TEXT,
        novelty_flag          BOOLEAN DEFAULT FALSE,
        novelty_score         FLOAT8,
        geometry_class        TEXT,
        simulation_duration_ms INT4,
        model_used            TEXT,
        loop_iteration        INT8
    );

    -- Literature embeddings table
    CREATE TABLE IF NOT EXISTS literature_embeddings (
        id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        title       TEXT NOT NULL,
        authors     TEXT,
        year        INT4,
        arxiv_id    TEXT,
        abstract    TEXT,
        content     TEXT,
        embedding   VECTOR(384),
        category    TEXT
    );

    -- Discovered metrics table
    CREATE TABLE IF NOT EXISTS discovered_metrics (
        id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        metric_name     TEXT,
        metric_ansatz   TEXT,
        energy_req      FLOAT8,
        stability       FLOAT8,
        novelty_score   FLOAT8,
        hypothesis      TEXT,
        confidence      FLOAT8,
        validated       BOOLEAN DEFAULT FALSE
    );

    -- Hypotheses table
    CREATE TABLE IF NOT EXISTS hypotheses (
        id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        geometry_type   TEXT,
        hypothesis_text TEXT,
        confidence      FLOAT8,
        uncertainty_type TEXT,
        novelty_flag    BOOLEAN,
        falsifiability  TEXT,
        simulation_id   UUID REFERENCES simulations(id)
    );

    -- Loop state table
    CREATE TABLE IF NOT EXISTS loop_state (
        id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        updated_at      TIMESTAMPTZ DEFAULT NOW(),
        iteration       INT8 DEFAULT 0,
        total_sims      INT8 DEFAULT 0,
        novel_count     INT8 DEFAULT 0,
        current_geo     TEXT DEFAULT 'morris_thorne'
    );

    -- Indexes for fast query
    CREATE INDEX IF NOT EXISTS idx_simulations_geometry
        ON simulations(geometry_type);
    CREATE INDEX IF NOT EXISTS idx_simulations_stability
        ON simulations(stability_score DESC);
    CREATE INDEX IF NOT EXISTS idx_simulations_novelty
        ON simulations(novelty_flag) WHERE novelty_flag = TRUE;
    CREATE INDEX IF NOT EXISTS idx_simulations_ford_roman
        ON simulations(ford_roman_status);

    -- Enable real-time on simulations table
    ALTER PUBLICATION supabase_realtime ADD TABLE simulations;
    """

    logger.info("Schema SQL ready. Paste the above into your Supabase SQL editor.")
    logger.info("Go to: https://supabase.com/dashboard → your project → SQL Editor")
    print("\n" + "="*60)
    print("PASTE THIS SQL INTO YOUR SUPABASE SQL EDITOR:")
    print("="*60)
    print(sql)
    print("="*60 + "\n")


if __name__ == "__main__":
    setup()
