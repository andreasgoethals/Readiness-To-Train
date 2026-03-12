"""
Generate DAG Visualizations for All Players
============================================

Generates causal DAG visualizations for every player in the dataset using the
DAGCreator. Creates four DAGs per player:

1. First cycle, schematic (collapsed Player State)
2. First cycle, detailed (all individual covariates)
3. All cycles, schematic (full longitudinal view)
4. All cycles, detailed (full longitudinal view with all covariates)

Output structure:
    images/DAGs/
    ├── player 1/
    │   ├── player_1_cycle_1_schematic.png
    │   ├── player_1_cycle_1_detailed.png
    │   ├── player_1_all_cycles_schematic.png
    │   └── player_1_all_cycles_detailed.png
    ├── player 2/
    │   └── ...
    └── player 28/
        └── ...

Usage:
    python scripts/generate_visualizations.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
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
    print(f"Found {len(player_ids)} players: {player_ids}")

    for pid in player_ids:
        player_dir = images_dir / f"player {pid}"
        player_dir.mkdir(parents=True, exist_ok=True)

        print(chr(10) + "=" * 60)
        print(f"  Player {pid}")
        print("=" * 60)

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
            print(f"  SKIPPED: {e}")
            continue

        n_cycles = creator.metadata['n_cycles']
        print(f"  Cycles: {n_cycles}, Lengths: {creator.cycle_lengths}")

        print("  [1/4] First cycle -- schematic")
        creator.visualize(cycles=1, completeness='schematic', output_dir=str(player_dir), dpi=dpi, show=False)

        print("  [2/4] First cycle -- detailed")
        creator.visualize(cycles=1, completeness='detailed', output_dir=str(player_dir), dpi=dpi, show=False)

        print("  [3/4] All cycles -- schematic")
        creator.visualize(cycles=None, completeness='schematic', output_dir=str(player_dir), dpi=dpi, show=False)

        print("  [4/4] All cycles -- detailed")
        creator.visualize(cycles=None, completeness='detailed', output_dir=str(player_dir), dpi=dpi, show=False)

        import matplotlib.pyplot as plt
        plt.close('all')

    print(chr(10) + "=" * 60)
    print(f"  All visualizations saved to: {images_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_player_dags()
