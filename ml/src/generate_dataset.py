"""
generate_dataset.py
===================
Generates labelled telemetry datasets for training the digital twin fault
classifier and anomaly detector models using the simulator physics model,
fault perturbation functions, and mission profiles.

Output CSV schema matches `engine/telemetry`:
  timestamp, rpm, egt, cht, oil_pressure, oil_temp, fuel_flow,
  vibration, altitude, ambient_temp, fault_label
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add project root to sys.path if needed
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulator.simulator.engine_model import EngineModel
from simulator.simulator.fault_injection import get_fault_delta, FAULT_MAP
from simulator.simulator.mission_profiles import profile_generator, PROFILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_SCENARIOS = ["normal"] + list(FAULT_MAP.keys())
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "labelled_telemetry.csv"


def generate_telemetry_dataset(
    samples_per_class: int = 1500,
    severities: tuple[float, ...] = (0.3, 0.6, 1.0),
    profiles: tuple[str, ...] = ("standard_isa", "ladakh", "haa"),
    output_path: Optional[Path | str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic labelled telemetry for all fault classes.

    Args:
        samples_per_class: Number of samples to generate per fault class.
        severities: Severity levels to cycle through for fault scenarios.
        profiles: Mission profile names to cycle through.
        output_path: Optional CSV output filepath.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame containing generated telemetry records.
    """
    rng = np.random.default_rng(seed)
    records = []
    base_time = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    current_time = base_time

    for scenario in ALL_SCENARIOS:
        logger.info("Generating %d samples for class '%s'...", samples_per_class, scenario)
        model = EngineModel(rng_seed=int(rng.integers(0, 1_000_000)))

        for i in range(samples_per_class):
            profile_name = profiles[i % len(profiles)]
            # Cycle through profile phases
            gen = profile_generator(profile_name)
            # Advance generator partially to simulate diverse mission phases
            skip_steps = (i * 13) % 200
            for _ in range(skip_steps):
                next(gen)
            altitude, ambient_temp = next(gen)

            model.set_mission_conditions(altitude, ambient_temp)
            state = model.tick()

            if scenario != "normal":
                base_sev = severities[i % len(severities)]
                # Add minor continuous jitter to severity
                sev = float(np.clip(base_sev + rng.normal(0, 0.05), 0.1, 1.2))
                delta, label = get_fault_delta(scenario, severity=sev)
                if delta:
                    model.apply_perturbation(delta)
                model.set_fault_label(label)
            else:
                model.set_fault_label("normal")

            state = model.tick()
            current_time += timedelta(seconds=1)

            records.append({
                "timestamp": current_time.isoformat(),
                "rpm": round(state.rpm, 1),
                "egt": round(state.egt, 1),
                "cht": round(state.cht, 1),
                "oil_pressure": round(state.oil_pressure, 3),
                "oil_temp": round(state.oil_temp, 1),
                "fuel_flow": round(state.fuel_flow, 2),
                "vibration": round(state.vibration, 4),
                "altitude": round(state.altitude, 1),
                "ambient_temp": round(state.ambient_temp, 1),
                "fault_label": state.fault_label or "normal",
            })

    df = pd.DataFrame(records)
    # Shuffle records
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved %d records to %s", len(df), out)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labelled telemetry dataset")
    parser.add_argument(
        "--samples", type=int, default=1500, help="Number of samples per class"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output CSV file",
    )
    args = parser.parse_args()

    df_out = generate_telemetry_dataset(
        samples_per_class=args.samples, output_path=args.output
    )
    print(f"Generated dataset shape: {df_out.shape}")
    print(df_out["fault_label"].value_counts())
