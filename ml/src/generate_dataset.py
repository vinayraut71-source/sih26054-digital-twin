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
    samples_per_class: int = 1000,
    incipient_ratio: float = 0.40,
    noise_scale: float = 1.9,
    profiles: tuple[str, ...] = ("standard_isa", "ladakh", "haa"),
    output_path: Optional[Path | str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic labelled telemetry with realistic sensor noise and incipient fault overlap.

    Args:
        samples_per_class: Number of samples to generate per fault class.
        incipient_ratio: Fraction of fault samples that are low-severity incipient degradation.
        noise_scale: Standard deviation scaling for transducer/environmental noise.
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
            gen = profile_generator(profile_name)
            skip_steps = (i * 13) % 200
            for _ in range(skip_steps):
                next(gen)
            altitude, ambient_temp = next(gen)

            model.set_mission_conditions(altitude, ambient_temp)
            state = model.tick()

            if scenario != "normal":
                if rng.random() < incipient_ratio:
                    # Incipient stage: low severity, overlaps naturally with normal operating variations
                    sev = float(rng.uniform(0.04, 0.20))
                else:
                    # Developed fault stage
                    sev = float(rng.uniform(0.25, 0.85))
                delta, label = get_fault_delta(scenario, severity=sev)
                if delta:
                    model.apply_perturbation(delta)
                model.set_fault_label(label)
            else:
                model.set_fault_label("normal")

            state = model.tick()
            current_time += timedelta(seconds=1)

            # Inject realistic sensor measurement noise & environmental variance
            rpm_val = round(float(state.rpm + rng.normal(0, 15 * noise_scale)), 1)
            egt_val = round(float(state.egt + rng.normal(0, 8 * noise_scale)), 1)
            cht_val = round(float(state.cht + rng.normal(0, 4 * noise_scale)), 1)
            oil_p_val = round(float(np.clip(state.oil_pressure + rng.normal(0, 0.15 * noise_scale), 0.5, 8.0)), 3)
            oil_t_val = round(float(state.oil_temp + rng.normal(0, 3 * noise_scale)), 1)
            fuel_f_val = round(float(np.clip(state.fuel_flow + rng.normal(0, 0.5 * noise_scale), 2.0, 35.0)), 2)
            vib_val = round(float(np.clip(state.vibration + rng.normal(0, 0.03 * noise_scale), 0.01, 1.5)), 4)

            records.append({
                "timestamp": current_time.isoformat(),
                "rpm": rpm_val,
                "egt": egt_val,
                "cht": cht_val,
                "oil_pressure": oil_p_val,
                "oil_temp": oil_t_val,
                "fuel_flow": fuel_f_val,
                "vibration": vib_val,
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
