"""
Generate DAG Visualizations for All Players -- All Cycles
==========================================================

Generates a schematic (grouped-covariate) causal DAG for every match cycle
of every player. One PNG file per cycle.

Output:
    images/DAGs/player X/player_X_cycle_Y_schematic.png

Usage:
    python src/utils/generate_visualizations.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # src/utils -> src -> project_root
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import matplotlib
matplotlib.use('Agg')

from methods.dag_creator import DAGCreator


STATE_VARS = [
    'Fatigue (z)',
    'Readiness (z)',
    'Soreness (z)',
    'Days Until Match',
]
TREATMENT = 'Training Intensity Score'
OUTCOME = 'Match Performance'

IMAGES_DIR = project_root / 'images' / 'DAGs'
DATA_PATH = project_root / 'data' / 'processed' / 'RTT.xlsx'


def generate_all_player_dags(
    state_vars=STATE_VARS,
    treatment_var=TREATMENT,
    outcome_var=OUTCOME,
    images_dir=IMAGES_DIR,
    data_path=DATA_PATH,
    dpi=300,
):
    images_dir = Path(images_dir)
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{data_path}'. "
            "Run src/data/data_preprocessing.py first."
        )

    df = pd.read_excel(data_path, usecols=['Player ID'])
    player_ids = sorted(df['Player ID'].unique())
    print(f"Found {len(player_ids)} players: {list(player_ids)}")

    total_generated = 0

    for pid in player_ids:
        player_dir = images_dir / f"player {pid}"
        player_dir.mkdir(parents=True, exist_ok=True)

        try:
            creator = DAGCreator(
                player_id=pid,
                state_vars=state_vars,
                treatment_var=treatment_var,
                outcome_var=outcome_var,
                cross_var_carryover=False,
                data_path=str(data_path),
            )
        except Exception as e:
            print(f"  Player {pid}: SKIPPED -- {e}")
            continue

        n_cycles = creator.metadata['n_cycles']
        print(f"  Player {pid}: {n_cycles} cycles, lengths = {creator.cycle_lengths}")

        for cycle_idx in range(1, n_cycles + 1):
            try:
                creator.visualize(
                    cycles=cycle_idx,
                    completeness='schematic',
                    output_dir=str(player_dir),
                    dpi=dpi,
                    show=False,
                )
                total_generated += 1
            except Exception as e:
                print(f"    Cycle {cycle_idx}: FAILED -- {e}")

        import matplotlib.pyplot as plt
        plt.close('all')

    print(f"\n  Total DAGs generated: {total_generated}")
    print(f"  Saved to: {images_dir}")


if __name__ == "__main__":
    generate_all_player_dags()
