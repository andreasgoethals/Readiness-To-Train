"""
DAG Creator for Causal Analysis of Longitudinal Match Cycles
(Dynamic Treatment Regime Framework) -- Player-Specific Version

Builds a player-specific causal Directed Acyclic Graph (DAG) over all observed
match cycles for one player. The DAG encodes the assumed causal structure between
player state (Lt), daily treatment intensity (At), and match-day performance (Y),
with inter-cycle feedback where match performance affects the starting state of
the next cycle.

=== USAGE ===

    from src.methods.DAG_Creator import DAGCreator

    creator = DAGCreator(
        player_id=5,
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)', 'Days Until Match'],
        treatment_var='Training Intensity Score',
        outcome_var='Match Performance',
        cross_var_carryover=False,   # self-only carryover (default)
    )

    # --- Object role: query the causal structure ---
    print(creator.summary())
    adj   = creator.to_adjacency_matrix()
    tvc   = creator.get_time_varying_confounders()
    paths = creator.get_causal_paths('Fatigue (z)_c1_t1', 'Match Performance_c1')

    # --- Visualisation role: two dimensions of control ---
    #   cycles      (timing)      : None=all, int=one, list=subset
    #   completeness (detail)     : 'schematic' or 'detailed'

    creator.visualize(cycles=None,   completeness='schematic',
                      save_path='results/dag_all_schematic.png')
    creator.visualize(cycles=[1, 2], completeness='detailed',
                      save_path='results/dag_cycles_1_2_detailed.png')
    creator.visualize(cycles=1,      completeness='schematic',
                      save_path='results/dag_cycle1_schematic.png')

=== CAUSAL CONTEXT ===

This project estimates an individualised optimal Dynamic Treatment Regime (DTR):
a sequence of decision rules mapping each player's current and historical state
(Lt) to an optimal continuous training intensity score At in [0, 1]. The key
challenges are:

1. Time-Varying Confounding Affected by Prior Treatment:
   Lt is simultaneously a consequence of A(t-1) (training causes fatigue) and
   a cause of At (coaches prescribe based on observed state). Standard
   regression cannot handle this correctly -- this is the setting for G-methods.

2. Treatment-Confounder Feedback:
   Coaching decisions are observational. Hard sessions are prescribed when
   players appear fresh; recovery when fatigued. Methods must disentangle the
   physiological treatment effect from coach selection behaviour.

3. Sequential Decision-Making:
   The optimal policy is a sequence A1, A2, ..., AK over the entire match cycle,
   not a single-point intervention. This is a K-stage DTR with variable K.

=== MATCH CYCLE STRUCTURE ===

A cycle runs from the day after one player-played match to the day of the next
match where the player is selected. Non-selection on a team match day extends
the current cycle -- the player keeps training until they next play.

    CYCLE k (N training days + match at the end)

    L1 --> A1 --> L2 --> A2 --> ... --> LN --> AN
     |     |       |     |               |      |
     |     | (treatment_effect)          |      |
     |     +------►                      |      |
     |        (state_carryover)          |      |
     +──────────────────────────────────►      |
                                               v
                                               Y  (match-day performance)

    Between cycles: Yk --> L1_(k+1)           (outcome_to_first_state feedback)
                    L_final_k --> L1_(k+1)    (final_state_to_first_state carry-over)

=== NON-SELECTED PLAYERS ===

If a player is NOT selected for a match (Match Day=1 for the team but
Selected=0 for this player), that day is treated as a training day in their
current cycle. The cycle only ends when the player personally plays. This means
a cycle may span a team match day where the player did not participate, resulting
in a longer cycle (e.g., 12 training days instead of 6).

=== CROSS-VARIABLE CARRYOVER ===

cross_var_carryover=False (default -- self-only, parsimonious):
    Fatigue_t1 --> Fatigue_t2
    Readiness_t1 --> Readiness_t2
    (each variable only directly affects the same variable at t+1)

cross_var_carryover=True (ALL-to-ALL, more biologically complete):
    Fatigue_t1 --> {Fatigue_t2, Readiness_t2, Soreness_t2, ...}
    Readiness_t1 --> {Fatigue_t2, Readiness_t2, Soreness_t2, ...}
    (every state variable at t directly affects every state variable at t+1)

Applied consistently to day-->day+1 carryover edges.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple


class DAGCreator:
    """
    Player-specific causal DAG builder for longitudinal DTR match cycles.

    Loads processed data for the specified player, auto-detects match cycles
    from the player's own Activity Type Today (non-selected team match days
    are treated as training days, making that cycle longer), and builds a
    networkx DiGraph representing the full causal structure over all observed
    cycles.

    Parameters
    ----------
    player_id : int
        Player ID from the processed dataset (1-27).
    state_vars : list of str
        Covariate (player state) variable names Lt. Appear as daily covariates
        (t=1..N) in each cycle. The first state (t=1) represents the player's
        condition at cycle start.
    treatment_var : str
        Treatment variable name At. Continuous daily training intensity in [0,1].
    outcome_var : str
        Outcome variable name Y. Match-day physical performance metric.
    data_path : str or Path, optional
        Path to processed data file. Defaults to data/processed/RTT.xlsx
        relative to the project root.
    cross_var_carryover : bool, default=False
        If False: self-only carryover -- each state variable at t only directly
        affects the same variable at t+1 (N edges per step, parsimonious).
        If True: ALL-to-ALL -- every state variable at t affects every state
        variable at t+1 (N^2 edges per step, biologically richer).
        Applied consistently to all day->day+1 transitions within each cycle.

    Attributes
    ----------
    dag : nx.DiGraph
        The constructed causal DAG. Built automatically at init.
    metadata : dict
        Metadata about the DAG structure (cycle counts, node/edge counts, etc.).
    cycle_lengths : list of int
        Auto-detected lengths of each match cycle for this player.
    """

    def __init__(
        self,
        player_id: int,
        state_vars: List[str],
        treatment_var: str,
        outcome_var: str,
        data_path: Optional[Union[str, Path]] = None,
        cross_var_carryover: bool = False,
    ):
        # Validate inputs
        if not state_vars:
            raise ValueError("state_vars must not be empty")
        if not treatment_var:
            raise ValueError("treatment_var must not be empty")
        if not outcome_var:
            raise ValueError("outcome_var must not be empty")
        if treatment_var in state_vars:
            raise ValueError(
                f"treatment_var '{treatment_var}' must not appear in state_vars"
            )

        self.player_id = player_id
        self.state_vars = list(state_vars)
        self.treatment_var = treatment_var
        self.outcome_var = outcome_var
        self.cross_var_carryover = cross_var_carryover

        # Load data, detect cycles, build DAG (all automatic)
        self._player_df: pd.DataFrame = self._load_player_data(data_path)
        self.cycle_lengths: List[int] = self._detect_cycle_lengths()
        self.dag: Optional[nx.DiGraph] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._build_dag()

    # =========================================================================
    # DATA LOADING & CYCLE DETECTION
    # =========================================================================

    def _load_player_data(
        self,
        data_path: Optional[Union[str, Path]],
    ) -> pd.DataFrame:
        """Load and return the processed data rows for this player, sorted by Date."""
        if data_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            data_path = project_root / "data" / "processed" / "RTT.xlsx"

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at '{data_path}'. "
                "Run src/data/data_preprocessing.py first."
            )

        # Support both .xlsx and .csv
        if str(data_path).endswith('.xlsx'):
            df = pd.read_excel(data_path, parse_dates=['Date'], engine='openpyxl')
        else:
            df = pd.read_csv(data_path, parse_dates=['Date'])
        player_df = df[df['Player ID'] == self.player_id].copy()

        if len(player_df) == 0:
            valid_ids = sorted(df['Player ID'].unique().tolist())
            raise ValueError(
                f"Player ID {self.player_id} not found in the processed data. "
                f"Valid IDs: {valid_ids}"
            )

        return player_df.sort_values('Date').reset_index(drop=True)

    def _detect_cycle_lengths(self) -> List[int]:
        """
        Auto-detect match cycle lengths from this player's data.

        A cycle ends when the player personally plays a match
        (Activity Type Today == 'Game' for this specific player).

        Non-selected match days (Match Day=1 for the team, but this player's
        Activity Type Today != 'Game') are treated as regular training days,
        extending the current cycle. This correctly models the case where a
        player's cycle spans a team match they didn't participate in.

        Returns
        -------
        list of int
            Cycle lengths = number of training-day nodes (At nodes) per cycle.
            Empty list if no matches found for this player.
        """
        df = self._player_df

        if 'Activity Type Today' not in df.columns:
            raise ValueError(
                "'Activity Type Today' column not found. "
                "Run src/data/data_preprocessing.py to create the processed dataset."
            )

        # Only rows where THIS PLAYER personally played (not just team match days)
        played_mask = df['Activity Type Today'].str.strip().str.lower() == 'game'
        match_positions = df.index[played_mask].tolist()  # Positions in 0-indexed sorted df

        if not match_positions:
            return []

        cycle_lengths = []
        prev_end = -1  # Position of the previous match row

        for match_pos in match_positions:
            # Cycle starts at the row immediately after the previous match
            cycle_start = prev_end + 1
            # Cycle ends at the match row (inclusive)
            cycle_end = match_pos

            # First state (L1) = row at cycle_start + 1 (day after previous match)
            # Full cycle: L1, A1, L2, A2, ... through to match day
            n_treatment_days = cycle_end - cycle_start

            # Guarantee at least 1 treatment day (even if match follows immediately)
            cycle_lengths.append(max(1, n_treatment_days))
            prev_end = match_pos

        return cycle_lengths

    # =========================================================================
    # NODE NAMING (static helpers)
    # =========================================================================

    @staticmethod
    def _node_name(var: str, cycle: int, day: int) -> str:
        """State/treatment node name: '{var}_c{cycle}_t{day}'."""
        return f"{var}_c{cycle}_t{day}"

    @staticmethod
    def _outcome_node_name(var: str, cycle: int) -> str:
        """Outcome node name: '{var}_c{cycle}'."""
        return f"{var}_c{cycle}"

    @staticmethod
    def _wrap_label(text: str, max_chars: int = 14) -> str:
        """Wrap text onto multiple lines so each line fits within *max_chars*."""
        if len(text) <= max_chars:
            return text
        words = text.split()
        lines: List[str] = []
        current = ""
        for w in words:
            test = f"{current} {w}".strip()
            if len(test) <= max_chars:
                current = test
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return "\n".join(lines)

    # =========================================================================
    # DAG CONSTRUCTION (private)
    # =========================================================================

    def _build_single_cycle(
        self,
        G: nx.DiGraph,
        cycle: int,
        n_days: int,
        absolute_time_offset: int,
    ) -> Dict[str, Any]:
        """
        Build nodes and intra-cycle edges for one match cycle.

        Parameters
        ----------
        G : nx.DiGraph
            Graph to extend in place.
        cycle : int
            1-based cycle index.
        n_days : int
            Number of training days (= number of At nodes) in this cycle.
        absolute_time_offset : int
            Absolute time index for t=1 of this cycle.

        Returns
        -------
        dict with 'outcome_node', 'absolute_time_end'.
        """
        state_vars = self.state_vars
        treatment_var = self.treatment_var
        outcome_var = self.outcome_var
        cross = self.cross_var_carryover

        # --- Step 1: Daily covariate + treatment nodes (t=1..N) ---
        for day in range(1, n_days + 1):
            abs_time = absolute_time_offset + day - 1
            for var in state_vars:
                node = self._node_name(var, cycle, day)
                G.add_node(node, variable=var, cycle=cycle, day=day,
                           absolute_time=abs_time, role='covariate')
            treat_node = self._node_name(treatment_var, cycle, day)
            G.add_node(treat_node, variable=treatment_var,
                       cycle=cycle, day=day, absolute_time=abs_time,
                       role='treatment')

        # --- Step 2: Outcome node Y ---
        outcome_abs_time = absolute_time_offset + n_days
        outcome_node = self._outcome_node_name(outcome_var, cycle)
        G.add_node(outcome_node, variable=outcome_var, cycle=cycle,
                   day=n_days + 1, absolute_time=outcome_abs_time, role='outcome')

        # --- Step 3: Daily structure (days 1..N) ---
        for day in range(1, n_days + 1):
            treat_node = self._node_name(treatment_var, cycle, day)

            # 3a. Lt --> At  (confounding: coaches prescribe based on player state)
            for var in state_vars:
                cov_node = self._node_name(var, cycle, day)
                G.add_edge(cov_node, treat_node, relation='confounding')

            # 3b. Carry-over to the next day
            if day < n_days:
                next_day = day + 1

                # At --> L(t+1)  (treatment effect: training changes fatigue/fitness)
                for var in state_vars:
                    next_cov = self._node_name(var, cycle, next_day)
                    G.add_edge(treat_node, next_cov, relation='treatment_effect')

                # Lt --> L(t+1)  (state carryover / persistence)
                for src_var in state_vars:
                    curr_cov = self._node_name(src_var, cycle, day)
                    tgt_vars = state_vars if cross else [src_var]
                    for tgt_var in tgt_vars:
                        next_cov = self._node_name(tgt_var, cycle, next_day)
                        G.add_edge(curr_cov, next_cov, relation='state_carryover')

        # --- Step 4: Final day --> Outcome Y ---
        final_treat = self._node_name(treatment_var, cycle, n_days)
        G.add_edge(final_treat, outcome_node, relation='treatment_to_outcome')
        for var in state_vars:
            final_cov = self._node_name(var, cycle, n_days)
            G.add_edge(final_cov, outcome_node, relation='covariate_to_outcome')

        return {
            'outcome_node': outcome_node,
            'absolute_time_end': outcome_abs_time,
        }

    def _build_dag(self) -> None:
        """Build the full multi-cycle DAG from detected cycle lengths."""
        if not self.cycle_lengths:
            raise ValueError(
                f"No matches found for player {self.player_id}. "
                "Cannot build a DTR DAG without at least one observed outcome. "
                "Check that the processed data contains 'Activity Type Today' "
                "entries equal to 'Game' for this player."
            )

        G = nx.DiGraph()
        n_cycles = len(self.cycle_lengths)
        absolute_time_offset = 0
        cycle_info = []
        prev_outcome_node = None
        prev_n_days = None

        for k in range(1, n_cycles + 1):
            n_days = self.cycle_lengths[k - 1]

            result = self._build_single_cycle(
                G, cycle=k, n_days=n_days,
                absolute_time_offset=absolute_time_offset,
            )

            # Inter-cycle feedback: Y_{k-1} --> L1_k
            if prev_outcome_node is not None:
                for var in self.state_vars:
                    first_state = self._node_name(var, k, 1)
                    G.add_edge(prev_outcome_node, first_state,
                               relation='outcome_to_first_state')

                # Final state carry-over: L_final_{k-1} --> L1_k
                for src_var in self.state_vars:
                    prev_final = self._node_name(src_var, k - 1, prev_n_days)
                    tgt_vars = (self.state_vars
                                if self.cross_var_carryover else [src_var])
                    for tgt_var in tgt_vars:
                        first_state = self._node_name(tgt_var, k, 1)
                        G.add_edge(prev_final, first_state,
                                   relation='final_state_to_first_state')

            cycle_info.append({
                'cycle': k,
                'n_days': n_days,
                'n_covariate_nodes': len(self.state_vars) * n_days,
                'n_treatment_nodes': n_days,
                'n_outcome_nodes': 1,
            })

            prev_outcome_node = result['outcome_node']
            prev_n_days = n_days
            absolute_time_offset = result['absolute_time_end'] + 1

        if not nx.is_directed_acyclic_graph(G):
            raise RuntimeError(
                "Constructed graph contains cycles -- this should not happen "
                "with temporal unrolling. Please report this as a bug."
            )

        n_feedback = sum(
            1 for _, _, d in G.edges(data=True)
            if d.get('relation') in ('outcome_to_first_state', 'final_state_to_first_state')
        )

        self.dag = G
        self.metadata = {
            'player_id': self.player_id,
            'n_cycles': n_cycles,
            'cycle_lengths': list(self.cycle_lengths),
            'state_vars': list(self.state_vars),
            'daily_treatment_var': self.treatment_var,
            'outcome_var': self.outcome_var,
            'cross_var_carryover': self.cross_var_carryover,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_feedback_edges': n_feedback,
            'per_cycle': cycle_info,
        }

    # =========================================================================
    # QUERY METHODS -- DAG OBJECT ROLE
    # =========================================================================

    def _check_dag(self):
        """Raise RuntimeError if no DAG has been built."""
        if self.dag is None:
            raise RuntimeError("No DAG built. This should not happen -- check __init__.")

    def get_nodes_by_role(self, role: str) -> List[str]:
        """
        Get all node names with a given role, sorted by absolute time.

        Parameters
        ----------
        role : str
            One of 'covariate', 'treatment', 'outcome'.
        """
        self._check_dag()
        nodes = [
            n for n, attrs in self.dag.nodes(data=True)
            if attrs.get('role') == role
        ]
        return sorted(nodes, key=lambda n: (
            self.dag.nodes[n].get('absolute_time', 0),
            self.dag.nodes[n].get('variable', '')
        ))

    def get_nodes_at_time(self, time: int) -> List[str]:
        """Get all node names at a specific absolute time index."""
        self._check_dag()
        return sorted([
            n for n, attrs in self.dag.nodes(data=True)
            if attrs.get('absolute_time') == time
        ])

    def get_nodes_in_cycle(self, cycle: int) -> List[str]:
        """Get all nodes belonging to a specific cycle (1-based), sorted by day."""
        self._check_dag()
        nodes = [
            n for n, attrs in self.dag.nodes(data=True)
            if attrs.get('cycle') == cycle
        ]
        return sorted(nodes, key=lambda n: (
            self.dag.nodes[n].get('day', 0),
            self.dag.nodes[n].get('variable', '')
        ))

    def get_nodes_at_cycle_day(self, cycle: int, day: int) -> List[str]:
        """Get all nodes at a specific day within a specific cycle."""
        self._check_dag()
        return sorted([
            n for n, attrs in self.dag.nodes(data=True)
            if attrs.get('cycle') == cycle and attrs.get('day') == day
        ])

    def get_cycle_outcome(self, cycle: int) -> str:
        """Get the outcome node name for a given cycle (1-based)."""
        self._check_dag()
        m = self.metadata
        if cycle < 1 or cycle > m['n_cycles']:
            raise ValueError(f"cycle must be between 1 and {m['n_cycles']}, got {cycle}")
        return self._outcome_node_name(m['outcome_var'], cycle)

    def get_feedback_edges(self) -> List[tuple]:
        """Get all inter-cycle feedback edges (outcome → first state AND final state → first state)."""
        return (self.get_edges_by_relation('outcome_to_first_state')
                + self.get_edges_by_relation('final_state_to_first_state'))

    def get_parents(self, node: str) -> List[str]:
        """Get direct causes (parents) of a node."""
        self._check_dag()
        return sorted(list(self.dag.predecessors(node)))

    def get_children(self, node: str) -> List[str]:
        """Get direct effects (children) of a node."""
        self._check_dag()
        return sorted(list(self.dag.successors(node)))

    def get_edges_by_relation(self, relation: str) -> List[tuple]:
        """
        Get all edges with a specific relation type.

        Parameters
        ----------
        relation : str
            One of: 'confounding', 'treatment_effect', 'state_carryover',
            'treatment_to_outcome', 'covariate_to_outcome',
            'outcome_to_first_state', 'final_state_to_first_state'.
        """
        self._check_dag()
        return [
            (u, v) for u, v, attrs in self.dag.edges(data=True)
            if attrs.get('relation') == relation
        ]

    def get_time_varying_confounders(self) -> List[str]:
        """
        Identify time-varying confounders affected by prior treatment.

        A node Lt is a time-varying confounder affected by prior treatment if:
          1. It is a covariate (role='covariate')
          2. It has a treatment node as a parent  (affected by prior treatment)
          3. It has a treatment node as a child   (confounds current treatment)

        These are precisely the nodes requiring G-methods for unbiased estimation.
        """
        self._check_dag()
        G = self.dag
        result = []
        for node, attrs in G.nodes(data=True):
            if attrs.get('role') != 'covariate':
                continue
            parents_are_treatment = any(
                G.nodes[p].get('role') == 'treatment'
                for p in G.predecessors(node)
            )
            children_are_treatment = any(
                G.nodes[c].get('role') == 'treatment'
                for c in G.successors(node)
            )
            if parents_are_treatment and children_are_treatment:
                result.append(node)
        return sorted(result, key=lambda n: (
            G.nodes[n].get('absolute_time', 0),
            G.nodes[n].get('variable', '')
        ))

    def to_adjacency_matrix(self) -> pd.DataFrame:
        """
        Export the DAG as a square adjacency matrix (pandas DataFrame).

        Nodes are sorted by absolute_time then variable name. Entry [i, j] = 1
        if there is a directed edge from node i to node j.
        """
        self._check_dag()
        nodes = sorted(
            self.dag.nodes(),
            key=lambda n: (
                self.dag.nodes[n].get('absolute_time', 0),
                self.dag.nodes[n].get('variable', n)
            )
        )
        adj = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
        for u, v in self.dag.edges():
            adj.loc[u, v] = 1
        return adj

    def get_causal_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Find all directed causal paths from source to target node.

        Returns
        -------
        list of list of str
            Each inner list is a path (sequence of node names). Empty if none.
        """
        self._check_dag()
        try:
            return list(nx.all_simple_paths(self.dag, source, target))
        except nx.NetworkXError:
            return []

    # =========================================================================
    # VISUALIZATION -- PUBLICATION-READY FIGURES
    # =========================================================================

    _COLORS = {
        'covariate':  '#1D4E89',   # Deep blue     -- time-varying state Lt
        'treatment':  '#B5451B',   # Burnt sienna  -- treatment At
        'outcome':    '#7B1D1D',   # Deep crimson  -- match outcome Y
        'feedback':   '#4A1472',   # Deep purple   -- inter-cycle feedback
        'background': '#F8F9FA',   # Off-white background
        'cycle_bg':   '#EEF2F7',   # Subtle cycle background
        'text':       '#1C2B39',   # Dark navy text
        'edge':       '#5A6A7A',   # Blue-grey default edge
    }

    _EDGE_STYLES = {
        'confounding': {
            'color': '#B5451B', 'style': '--', 'alpha': 0.92, 'width': 2.2,
            'label': 'Confounding (Lt -> At)',
        },
        'treatment_effect': {
            'color': '#8B0000', 'style': '-',  'alpha': 0.92, 'width': 2.4,
            'label': 'Treatment effect (At -> L(t+1))',
        },
        'state_carryover': {
            'color': '#1D4E89', 'style': '-',  'alpha': 0.70, 'width': 2.0,
            'label': 'State carry-over (Lt -> L(t+1))',
        },
        'treatment_to_outcome': {
            'color': '#8B0000', 'style': '-',  'alpha': 0.94, 'width': 2.6,
            'label': 'Treatment -> outcome',
        },
        'covariate_to_outcome': {
            'color': '#1D4E89', 'style': '-',  'alpha': 0.82, 'width': 2.2,
            'label': 'State -> outcome',
        },
        'outcome_to_first_state': {
            'color': '#4A1472', 'style': '-',  'alpha': 0.95, 'width': 2.8,
            'label': 'Inter-cycle feedback (Yk -> L1_(k+1))',
        },
        'final_state_to_first_state': {
            'color': '#0B7A75', 'style': '-',  'alpha': 0.88, 'width': 2.4,
            'label': 'Final state carry-over (L_final -> L1_(k+1))',
        },
    }

    def _get_node_color(self, role: str) -> str:
        return self._COLORS.get(role, '#888888')

    def visualize(
        self,
        cycles: Optional[Union[int, List[int]]] = None,
        completeness: str = 'schematic',
        save_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualize the causal DAG.

        Parameters
        ----------
        cycles : None, int, or list of int
            TIMING -- which match cycles to include in the visualization.
              None    : show all cycles (default).
              int     : show only that cycle number (e.g., cycles=1).
              list    : show those specific cycles (e.g., cycles=[1, 3]).
        completeness : str, default='schematic'
            COMPLETENESS -- how much structural detail to show.
              'schematic' : all state variables at each time step are collapsed
                            into a single "Player State" node. Clean and
                            publication-ready.
              'detailed'  : each state variable is shown as its own node, but
                            grouped visually under a "Player State" bounding box
                            to make clear they represent the same concept.
        save_path : str or None
            Save figure to this explicit path (.png, .pdf, .svg). None = don't
            save. Takes precedence over output_dir if both are provided.
        output_dir : str or None
            Directory where the figure should be saved with an auto-generated
            filename. Ignored if save_path is provided. The auto-generated name
            encodes player ID, cycles shown, and completeness level.
        figsize : (float, float) or None
            Figure size in inches. Auto-computed if None.
        title : str or None
            Custom title. Auto-generated from DAG metadata if None.
        dpi : int, default=150
            Resolution for raster outputs.
        show : bool, default=True
            Whether to call plt.show().

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_dag()
        m = self.metadata

        # Resolve cycles (timing)
        all_cycles = list(range(1, m['n_cycles'] + 1))
        if cycles is None:
            cycles_to_show = all_cycles
        elif isinstance(cycles, int):
            if cycles < 1 or cycles > m['n_cycles']:
                raise ValueError(
                    f"cycles={cycles} out of range [1, {m['n_cycles']}]"
                )
            cycles_to_show = [cycles]
        else:
            for c in cycles:
                if c < 1 or c > m['n_cycles']:
                    raise ValueError(
                        f"cycles value {c} out of range [1, {m['n_cycles']}]"
                    )
            cycles_to_show = sorted(list(cycles))

        # Dispatch on completeness
        if completeness == 'schematic':
            fig = self._visualize_schematic(cycles_to_show, figsize=figsize, title=title)
        elif completeness == 'detailed':
            fig = self._visualize_detailed(cycles_to_show, figsize=figsize, title=title)
        else:
            raise ValueError(
                f"Unknown completeness='{completeness}'. "
                "Choose 'schematic' or 'detailed'."
            )

        # Resolve save location: explicit save_path takes precedence,
        # otherwise auto-generate filename in output_dir if provided.
        resolved_path = save_path
        if resolved_path is None and output_dir is not None:
            pid = self.metadata['player_id']
            if cycles is None:
                cycle_tag = "all_cycles"
            elif isinstance(cycles, int):
                cycle_tag = f"cycle_{cycles}"
            else:
                cycle_tag = f"cycles_{'_'.join(str(c) for c in cycles_to_show)}"
            fname = f"player_{pid}_{cycle_tag}_{completeness}.png"
            resolved_path = str(Path(output_dir) / fname)

        if resolved_path is not None:
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
            # Auto-cap DPI if the resulting image would exceed matplotlib's
            # 2^16 pixel limit in either dimension.
            _MAX_PX = 65000  # safe margin below 2^16 = 65536
            fig_w_in, fig_h_in = fig.get_size_inches()
            max_dim_in = max(fig_w_in, fig_h_in)
            effective_dpi = dpi
            if max_dim_in * dpi > _MAX_PX:
                effective_dpi = int(_MAX_PX / max_dim_in)
                effective_dpi = max(72, effective_dpi)  # never below 72
            fig.savefig(resolved_path, dpi=effective_dpi, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"     Saved: {resolved_path}  (dpi={effective_dpi})")

        if show:
            plt.show()

        return fig

    # -------------------------------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------------------------------

    def _draw_arrow(
        self,
        ax: plt.Axes,
        x1: float, y1: float,
        x2: float, y2: float,
        relation: str,
        connectionstyle: str = 'arc3,rad=0.0',
        shrink_a: float = 22,
        shrink_b: float = 22,
    ):
        """Draw a styled directed arrow for a given relation type."""
        style = self._EDGE_STYLES.get(relation, {
            'color': self._COLORS['edge'], 'style': '-', 'alpha': 0.5, 'width': 1.0
        })
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>, head_width=0.22, head_length=0.18",
                color=style['color'],
                lw=style['width'],
                alpha=style['alpha'],
                connectionstyle=connectionstyle,
                linestyle=style['style'],
                shrinkA=shrink_a,
                shrinkB=shrink_b,
            ),
            zorder=2,
        )

    def _draw_state_node(
        self, ax: plt.Axes, x: float, y: float,
        label: str, role: str,
        node_w: float = 1.8, node_h: float = 0.85,
        fontsize: float = 7.5,
    ):
        """Draw an elliptical state (covariate) node."""
        color = self._get_node_color(role)
        ellipse = mpatches.Ellipse(
            (x, y), node_w, node_h,
            facecolor=color, edgecolor='white',
            linewidth=1.8, alpha=0.95, zorder=3,
        )
        ax.add_patch(ellipse)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4,
                linespacing=1.1)
        return ellipse

    def _draw_treatment_node(
        self, ax: plt.Axes, x: float, y: float,
        label: str,
        node_w: float = 1.6, node_h: float = 0.75,
        fontsize: float = 7.5,
    ):
        """Draw a rounded-rectangle treatment node."""
        color = self._get_node_color('treatment')
        rect = FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2),
            node_w, node_h,
            boxstyle="round,pad=0.10",
            facecolor=color, edgecolor='white',
            linewidth=1.8, alpha=0.95, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4,
                linespacing=1.1)
        return rect

    def _draw_outcome_node(
        self, ax: plt.Axes, x: float, y: float,
        label: str, radius: float = 0.72,
        fontsize: float = 7.5,
    ):
        """Draw a diamond-shaped outcome node."""
        color = self._get_node_color('outcome')
        diamond = mpatches.RegularPolygon(
            (x, y), numVertices=4, radius=radius,
            orientation=0,
            facecolor=color, edgecolor='white',
            linewidth=2.0, alpha=0.95, zorder=3,
        )
        ax.add_patch(diamond)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4,
                linespacing=1.1)
        return diamond

    def _draw_fancy_arrow(
        self,
        ax: plt.Axes,
        pos_a: Tuple[float, float],
        pos_b: Tuple[float, float],
        relation: str,
        patch_a=None,
        patch_b=None,
        connectionstyle: str = 'arc3,rad=0.0',
    ):
        """Draw a styled directed arrow using FancyArrowPatch with optional patch clipping."""
        style = self._EDGE_STYLES.get(relation, {
            'color': self._COLORS['edge'], 'style': '-', 'alpha': 0.5, 'width': 1.0
        })
        arrow = FancyArrowPatch(
            posA=pos_a, posB=pos_b,
            patchA=patch_a, patchB=patch_b,
            arrowstyle="-|>,head_width=0.25,head_length=0.20",
            connectionstyle=connectionstyle,
            color=style['color'],
            lw=style['width'],
            alpha=style['alpha'],
            linestyle=style['style'],
            mutation_scale=18,
            zorder=2,
        )
        ax.add_patch(arrow)
        return arrow

    def _draw_cycle_box(
        self, ax: plt.Axes,
        x_left: float, x_right: float,
        y_bottom: float, y_top: float,
        label: str,
        pad: float = 0.75,
    ):
        """Draw a labelled background box for a match cycle."""
        rect = FancyBboxPatch(
            (x_left - pad, y_bottom - pad),
            (x_right - x_left) + 2 * pad,
            (y_top - y_bottom) + 2 * pad,
            boxstyle="round,pad=0.3",
            facecolor=self._COLORS['cycle_bg'],
            edgecolor='#C8D4E0',
            linewidth=0.9,
            alpha=0.55,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            (x_left + x_right) / 2,
            y_top + pad + 0.25,
            label,
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold',
            color=self._COLORS['text'], alpha=0.85,
            zorder=1,
        )

    def _draw_state_group_box(
        self, ax: plt.Axes,
        x: float,
        y_bottom: float,
        y_top: float,
        width: float,
        role: str,
    ):
        """
        Draw a grouping box behind state variable nodes at a single time step.

        This box makes it visually clear that all individual covariate nodes
        at a given time step together represent the player's state (Lt or L0).
        Uses a light fill + colored border matching the node role color.
        """
        color = self._get_node_color(role)
        # Light fill
        rect_fill = FancyBboxPatch(
            (x - width / 2, y_bottom),
            width, y_top - y_bottom,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor='none',
            alpha=0.08,
            zorder=1,
        )
        ax.add_patch(rect_fill)
        # Colored border
        rect_border = FancyBboxPatch(
            (x - width / 2, y_bottom),
            width, y_top - y_bottom,
            boxstyle="round,pad=0.08",
            facecolor='none',
            edgecolor=color,
            linewidth=1.4,
            alpha=0.45,
            zorder=2,
        )
        ax.add_patch(rect_border)

    # -------------------------------------------------------------------------
    # Schematic visualization (collapsed Player State)
    # -------------------------------------------------------------------------

    def _visualize_schematic(
        self,
        cycles_to_show: List[int],
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Collapsed view: all state variables summarized into one 'Player State' node."""
        m = self.metadata

        total_slots = sum(m['cycle_lengths'][k - 1] + 1 for k in cycles_to_show)
        n_gaps = max(0, len(cycles_to_show) - 1)
        _fig_w_target = max(11.0, total_slots * 2.0)
        _fig_h_target = 7.0
        _data_x_target = (_fig_w_target / _fig_h_target) * 4.8
        _denom = total_slots + n_gaps * 0.8
        X_STEP = max(1.8, min(2.8, _data_x_target / max(1, _denom)))
        Y_STATE, Y_TREAT = 1.8, -1.3
        # Multi-cycle: outcome at treatment level for alignment; single: midway
        Y_OUTCOME = Y_TREAT if len(cycles_to_show) > 1 else 0.25
        CYCLE_GAP = max(1.8, X_STEP * 0.85)
        _ns = X_STEP / 2.8
        NODE_W_STATE = max(0.9, 1.80 * _ns)
        NODE_H_STATE = 0.82
        NODE_W_TREAT = max(0.8, 1.55 * _ns)
        NODE_H_TREAT = 0.72
        OUTCOME_R = 0.72
        _state_lbl = 'Player\nState'
        _treat_lbl = 'Training\nIntensity'

        outcome_pos: Dict[int, Tuple[float, float]] = {}
        state_pos: Dict[Tuple[int, int], Tuple[float, float]] = {}
        treat_pos: Dict[Tuple[int, int], Tuple[float, float]] = {}
        cycle_extents: Dict[int, Tuple[float, float]] = {}

        x_cursor = 0.0
        for k in cycles_to_show:
            n_days = m['cycle_lengths'][k - 1]
            x_left = x_cursor
            for d in range(1, n_days + 1):
                state_pos[(k, d)] = (x_cursor, Y_STATE)
                treat_pos[(k, d)] = (x_cursor, Y_TREAT)
                x_cursor += X_STEP
            outcome_pos[k] = (x_cursor, Y_OUTCOME)
            cycle_extents[k] = (x_left, x_cursor)
            x_cursor += CYCLE_GAP

        if figsize is None:
            figsize = (_fig_w_target, _fig_h_target)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self._COLORS['background'])
        ax.set_facecolor(self._COLORS['background'])

        for k in cycles_to_show:
            xl, xr = cycle_extents[k]
            n_days = m['cycle_lengths'][k - 1]
            y_box_bot = min(Y_TREAT, Y_OUTCOME) - OUTCOME_R * 0.5
            self._draw_cycle_box(ax, xl, xr, y_box_bot, Y_STATE,
                                 f"Cycle {k}   \u00b7   {n_days} training day{'s' if n_days != 1 else ''}", pad=0.75)

        # Draw nodes and collect patches
        sp: Dict[Tuple[int, int], Any] = {}
        tp: Dict[Tuple[int, int], Any] = {}
        op: Dict[int, Any] = {}
        for k in cycles_to_show:
            n_days = m['cycle_lengths'][k - 1]
            for d in range(1, n_days + 1):
                sx, sy = state_pos[(k, d)]
                tx, ty = treat_pos[(k, d)]
                sp[(k, d)] = self._draw_state_node(ax, sx, sy, _state_lbl, 'covariate', NODE_W_STATE, NODE_H_STATE)
                ax.text(sx, sy + NODE_H_STATE / 2 + 0.18, 'L\u209c', ha='center', va='bottom',
                        fontsize=11, fontstyle='italic', color=self._COLORS['text'], alpha=0.80, zorder=5)
                tp[(k, d)] = self._draw_treatment_node(ax, tx, ty, _treat_lbl, NODE_W_TREAT, NODE_H_TREAT)
                ax.text(tx, ty - NODE_H_TREAT / 2 - 0.30, 'A\u209c', ha='center', va='top',
                        fontsize=11, fontstyle='italic', color=self._COLORS['text'], alpha=0.80, zorder=5)
            ox, oy = outcome_pos[k]
            op[k] = self._draw_outcome_node(ax, ox, oy, 'Match\nPerformance', OUTCOME_R)
            ax.text(ox, oy - OUTCOME_R - 0.22, f'Y\u2096', ha='center', va='top',
                    fontsize=11, fontstyle='italic', color=self._COLORS['text'], alpha=0.80, zorder=5)

        # Edges using FancyArrowPatch with patch clipping
        for k in cycles_to_show:
            n_days = m['cycle_lengths'][k - 1]
            for d in range(1, n_days + 1):
                sx, sy = state_pos[(k, d)]
                tx, ty = treat_pos[(k, d)]
                if d > 1:
                    self._draw_fancy_arrow(ax, state_pos[(k, d - 1)], (sx, sy), 'state_carryover',
                                           patch_a=sp[(k, d - 1)], patch_b=sp[(k, d)])
                self._draw_fancy_arrow(ax, (sx, sy), (tx, ty), 'confounding',
                                       patch_a=sp[(k, d)], patch_b=tp[(k, d)])
                if d < n_days:
                    self._draw_fancy_arrow(ax, (tx, ty), state_pos[(k, d + 1)], 'treatment_effect',
                                           patch_a=tp[(k, d)], patch_b=sp[(k, d + 1)], connectionstyle='arc3,rad=0.3')
            ox, oy = outcome_pos[k]
            self._draw_fancy_arrow(ax, state_pos[(k, n_days)], (ox, oy), 'covariate_to_outcome',
                                   patch_a=sp[(k, n_days)], patch_b=op[k], connectionstyle='arc3,rad=0.2')
            self._draw_fancy_arrow(ax, treat_pos[(k, n_days)], (ox, oy), 'treatment_to_outcome',
                                   patch_a=tp[(k, n_days)], patch_b=op[k], connectionstyle='arc3,rad=-0.2')

        # Inter-cycle feedback
        for idx, k in enumerate(cycles_to_show[:-1]):
            next_k = cycles_to_show[idx + 1]
            ox, oy = outcome_pos[k]
            nsx, nsy = state_pos[(next_k, 1)]
            self._draw_fancy_arrow(ax, (ox, oy), (nsx, nsy), 'outcome_to_first_state',
                                   patch_a=op[k], patch_b=sp[(next_k, 1)], connectionstyle='arc3,rad=-0.55')
            n_days_k = m['cycle_lengths'][k - 1]
            fsx, fsy = state_pos[(k, n_days_k)]
            self._draw_fancy_arrow(ax, (fsx, fsy), (nsx, nsy), 'final_state_to_first_state',
                                   patch_a=sp[(k, n_days_k)], patch_b=sp[(next_k, 1)], connectionstyle='arc3,rad=0.0')

        if title is None:
            state_str = ', '.join(m['state_vars'])
            if len(state_str) > 70:
                state_str = state_str[:67] + '\u2026'
            if len(cycles_to_show) == 1:
                subtitle = f"Cycle {cycles_to_show[0]}  \u00b7  {m['cycle_lengths'][cycles_to_show[0]-1]} training days"
            else:
                lengths = [m['cycle_lengths'][k - 1] for k in cycles_to_show]
                subtitle = f"{len(cycles_to_show)} cycle(s)  \u00b7  lengths {lengths}"
            title = (f"Causal DAG \u2014 Player {m['player_id']}  \u00b7  {subtitle}\n"
                     f"State: [{state_str}]   |   Treatment: {m['daily_treatment_var']}   |   Outcome: {m['outcome_var']}")
        ax.set_title(title, fontsize=10, fontweight='bold', color=self._COLORS['text'], pad=22, linespacing=1.5)

        from matplotlib.lines import Line2D
        legend_handles = [
            mpatches.Patch(facecolor=self._COLORS['covariate'], edgecolor='white', label='Player state (L\u209c)'),
            mpatches.Patch(facecolor=self._COLORS['treatment'], edgecolor='white', label='Treatment \u2014 Training Intensity (A\u209c \u2208 [0,1])'),
            mpatches.Patch(facecolor=self._COLORS['outcome'], edgecolor='white', label='Match outcome (Y)'),
            Line2D([0], [0], color=self._EDGE_STYLES['state_carryover']['color'], lw=1.6, label='State carry-over (L\u209c \u2192 L\u209c\u208a\u2081)'),
            Line2D([0], [0], color=self._EDGE_STYLES['confounding']['color'], lw=1.6, linestyle='--', label='Confounding (L\u209c \u2192 A\u209c)'),
            Line2D([0], [0], color=self._EDGE_STYLES['treatment_effect']['color'], lw=1.8, label='Treatment effect (A\u209c \u2192 L\u209c\u208a\u2081)'),
        ]
        if len(cycles_to_show) > 1:
            legend_handles.extend([
                Line2D([0], [0], color=self._EDGE_STYLES['outcome_to_first_state']['color'], lw=2.2, label='Inter-cycle feedback (Y\u2096 \u2192 L\u2081)'),
                Line2D([0], [0], color=self._EDGE_STYLES['final_state_to_first_state']['color'], lw=1.8, label='Final state carry-over (L\u209c \u2192 L\u2081)'),
            ])
        ax.legend(handles=legend_handles, loc='lower right', fontsize=7.5, framealpha=0.92,
                  edgecolor='#C0CCD8', fancybox=True, ncol=2, columnspacing=1.0)

        ax.axis('off')
        ax.margins(0.10)
        fig.tight_layout(pad=2.0)
        return fig

    # -------------------------------------------------------------------------
    # Detailed visualization (individual variable nodes + grouping boxes)
    # -------------------------------------------------------------------------

    def _visualize_detailed(
        self,
        cycles_to_show: List[int],
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Detailed view: each state variable is a separate node.

        State variable nodes at each time step are grouped inside a labelled
        bounding box to make visually clear that they collectively represent
        the player's state (Lt or L0). Treatment and outcome are single nodes.

        cross_var_carryover=True produces N^2 carryover arrows per step.
        cross_var_carryover=False produces N self-only arrows (cleaner).
        """
        m = self.metadata
        G = self.dag
        n_state = len(m['state_vars'])

        # Adaptive layout constants
        total_slots = sum(m['cycle_lengths'][k - 1] + 1 for k in cycles_to_show)
        n_gaps = max(0, len(cycles_to_show) - 1)
        _fig_w_target = max(14.0, total_slots * 2.5)
        _fig_h_target = max(8.0, n_state * 1.6 + 4.5)
        _data_y_range = n_state * 1.4 + 3.5
        _data_x_target = (_fig_w_target / _fig_h_target) * _data_y_range
        _denom = total_slots + n_gaps * 0.8
        X_STEP = max(1.8, min(3.5, _data_x_target / max(1, _denom)))

        Y_SPACING = 1.3       # Vertical spacing between state vars within a group
        Y_STATE_TOP = (n_state - 1) * Y_SPACING / 2.0   # y of the topmost state node
        Y_TREAT = -Y_STATE_TOP - 2.0    # treatment node is below the state group
        CYCLE_GAP = max(2.2, X_STEP * 0.9)

        NODE_W = max(1.0, 1.6 * X_STEP / 3.5)
        NODE_H = 0.70
        TREAT_W = max(0.9, 1.4 * X_STEP / 3.5)
        TREAT_H = 0.78    # Taller to accommodate wrapped text
        OUTCOME_R = 0.72

        GROUP_PAD_X = 0.30   # Padding around the state group box (x)
        GROUP_PAD_Y = 0.32   # Padding around the state group box (y)

        # Multi-cycle: outcome at treatment level for alignment; single: midway
        _multi = len(cycles_to_show) > 1
        Y_OUTCOME_DETAIL = Y_TREAT if _multi else (Y_STATE_TOP + Y_TREAT) / 2

        # Build unified position dict: node_name -> (x, y)
        pos: Dict[str, Tuple[float, float]] = {}
        cycle_extents: Dict[int, Tuple[float, float, float, float]] = {}  # (xl, xr, yb, yt)

        x_cursor = 0.0
        for k in cycles_to_show:
            n_days = m['cycle_lengths'][k - 1]
            x_left = x_cursor

            # Training days (t=1..N) — no separate baseline; day 1 IS the cycle start
            for d in range(1, n_days + 1):
                dx = x_cursor
                for i, var in enumerate(m['state_vars']):
                    node = self._node_name(var, k, d)
                    pos[node] = (dx, Y_STATE_TOP - i * Y_SPACING)
                treat_node = self._node_name(m['daily_treatment_var'], k, d)
                pos[treat_node] = (dx, Y_TREAT)
                x_cursor += X_STEP

            # Outcome
            ox = x_cursor
            outcome_node = self._outcome_node_name(m['outcome_var'], k)
            pos[outcome_node] = (ox, Y_OUTCOME_DETAIL)

            # Cycle extents for background box
            cycle_extents[k] = (
                x_left,
                ox,
                Y_TREAT - TREAT_H / 2 - GROUP_PAD_Y - 0.5,
                Y_STATE_TOP + NODE_H / 2 + GROUP_PAD_Y + 1.4
            )
            x_cursor = ox + CYCLE_GAP

        # Figure
        if figsize is None:
            figsize = (_fig_w_target, _fig_h_target)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self._COLORS['background'])
        ax.set_facecolor(self._COLORS['background'])

        # --- Cycle background boxes ---
        for k in cycles_to_show:
            x_left, x_right, y_bot, y_top = cycle_extents[k]
            n_days = m['cycle_lengths'][k - 1]
            self._draw_cycle_box(ax, x_left, x_right, y_bot, y_top,
                                 f"Cycle {k}  \u00b7  {n_days} training day{'s' if n_days != 1 else ''}",
                                 pad=0.5)

        # --- State group boxes (one per time step, all covariate role) ---
        _sub_digits = '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089'
        def _to_sub(n):
            return ''.join(_sub_digits[int(c)] for c in str(n))

        for k in cycles_to_show:
            n_days = m['cycle_lengths'][k - 1]
            for d in range(1, n_days + 1):
                state_nodes_at_d = [
                    self._node_name(var, k, d) for var in m['state_vars']
                ]
                x = pos[state_nodes_at_d[0]][0]
                ys = [pos[n][1] for n in state_nodes_at_d]
                y_top_box  = max(ys) + NODE_H / 2 + GROUP_PAD_Y
                y_bot_box  = min(ys) - NODE_H / 2 - GROUP_PAD_Y
                box_width  = NODE_W + 2 * GROUP_PAD_X
                self._draw_state_group_box(ax, x, y_bot_box, y_top_box, box_width, 'covariate')

                # "Player State" caption above the group box
                ax.text(x, y_top_box + 0.10, 'Player\nState',
                        ha='center', va='bottom', fontsize=6.5,
                        color=self._get_node_color('covariate'), alpha=0.60, zorder=5,
                        linespacing=1.2)

                # Subscript label ABOVE the caption
                lbl = f'L{_to_sub(d)}'
                ax.text(x, y_top_box + 0.65, lbl,
                        ha='center', va='bottom', fontsize=8, fontstyle='italic',
                        color=self._get_node_color('covariate'), alpha=0.85, zorder=5)

        # --- Draw nodes and collect patches for FancyArrowPatch clipping ---
        node_patches: Dict[str, Any] = {}
        _fs_state = 6.0   # font size for state variable nodes
        _fs_treat = 6.0   # font size for treatment nodes
        _fs_outcome = 6.5  # font size for outcome node
        for k in cycles_to_show:
            n_days = m['cycle_lengths'][k - 1]

            for d in range(1, n_days + 1):
                for i, var in enumerate(m['state_vars']):
                    node = self._node_name(var, k, d)
                    if node not in pos:
                        continue
                    x, y = pos[node]
                    short = var.replace(' (z)', '').replace(' Yesterday', '')
                    short = self._wrap_label(short, max_chars=12)
                    node_patches[node] = self._draw_state_node(
                        ax, x, y, short, 'covariate', NODE_W, NODE_H,
                        fontsize=_fs_state)

                # Treatment node
                treat_node = self._node_name(m['daily_treatment_var'], k, d)
                if treat_node in pos:
                    tx, ty = pos[treat_node]
                    treat_short = self._wrap_label(
                        m['daily_treatment_var'], max_chars=16)
                    node_patches[treat_node] = self._draw_treatment_node(
                        ax, tx, ty, treat_short, TREAT_W, TREAT_H,
                        fontsize=_fs_treat)
                    ax.text(tx, ty - TREAT_H / 2 - 0.30, f'A{_to_sub(d)}',
                            ha='center', va='top', fontsize=7, fontstyle='italic',
                            color=self._COLORS['text'], alpha=0.75, zorder=5)

            # Outcome node
            outcome_node = self._outcome_node_name(m['outcome_var'], k)
            if outcome_node in pos:
                ox, oy = pos[outcome_node]
                out_short = self._wrap_label(m['outcome_var'], max_chars=16)
                node_patches[outcome_node] = self._draw_outcome_node(
                    ax, ox, oy, out_short, OUTCOME_R, fontsize=_fs_outcome)
                ax.text(ox, oy - OUTCOME_R - 0.18, f'Y{_to_sub(k)}',
                        ha='center', va='top', fontsize=8, fontstyle='italic',
                        color=self._COLORS['text'], alpha=0.75, zorder=5)

        # --- Edges using FancyArrowPatch with patch clipping ---
        shown_nodes = set(pos.keys())
        for u, v, data in G.edges(data=True):
            if u not in shown_nodes or v not in shown_nodes:
                continue
            u_cycle = G.nodes[u].get('cycle')
            v_cycle = G.nodes[v].get('cycle')
            if u_cycle not in cycles_to_show and v_cycle not in cycles_to_show:
                continue

            relation = data.get('relation', 'unknown')

            # Choose connection style
            if relation == 'outcome_to_first_state':
                conn = 'arc3,rad=-0.5'
            elif relation == 'final_state_to_first_state':
                conn = 'arc3,rad=0.0'
            elif relation == 'confounding':
                conn = 'arc3,rad=0.0'
            elif relation == 'treatment_effect':
                conn = 'arc3,rad=-0.25'
            elif relation == 'state_carryover':
                u_var = G.nodes[u].get('variable', '')
                v_var = G.nodes[v].get('variable', '')
                u_idx = m['state_vars'].index(u_var) if u_var in m['state_vars'] else 0
                v_idx = m['state_vars'].index(v_var) if v_var in m['state_vars'] else 0
                diff = v_idx - u_idx
                rad = 0.0 + diff * 0.15
                conn = f'arc3,rad={rad:.2f}'
            else:
                conn = 'arc3,rad=0.10'

            self._draw_fancy_arrow(
                ax, pos[u], pos[v], relation,
                patch_a=node_patches.get(u),
                patch_b=node_patches.get(v),
                connectionstyle=conn,
            )

        # --- Title ---
        if title is None:
            cross_note = "(ALL-to-ALL carryover)" if m['cross_var_carryover'] else "(self-only carryover)"
            if len(cycles_to_show) == 1:
                k = cycles_to_show[0]
                subtitle = f"Cycle {k}  \u00b7  {m['cycle_lengths'][k-1]} training days"
            else:
                lengths = [m['cycle_lengths'][k - 1] for k in cycles_to_show]
                subtitle = f"{len(cycles_to_show)} cycle(s)  \u00b7  lengths {lengths}"
            title = (
                f"Causal DAG (Detailed) \u2014 Player {m['player_id']}  \u00b7  "
                f"{subtitle}  {cross_note}\n"
                f"State: {m['state_vars']}   |   Treatment: {m['daily_treatment_var']}"
            )
        ax.set_title(title, fontsize=9, fontweight='bold',
                     color=self._COLORS['text'], pad=20, linespacing=1.4)

        # --- Legend ---
        from matplotlib.lines import Line2D
        legend_handles = [
            mpatches.Patch(facecolor=self._COLORS['covariate'], edgecolor='white',
                           label='Player state (L\u209c)'),
            mpatches.Patch(facecolor=self._COLORS['treatment'], edgecolor='white',
                           label='Treatment (A\u209c)'),
            mpatches.Patch(facecolor=self._COLORS['outcome'], edgecolor='white',
                           label='Match outcome (Y)'),
            Line2D([0], [0], color=self._EDGE_STYLES['state_carryover']['color'],
                   lw=1.4, label='State carry-over (L\u209c \u2192 L\u209c\u208a\u2081)'),
            Line2D([0], [0], color=self._EDGE_STYLES['confounding']['color'],
                   lw=1.4, linestyle='--', label='Confounding (L\u209c \u2192 A\u209c)'),
            Line2D([0], [0], color=self._EDGE_STYLES['treatment_effect']['color'],
                   lw=1.8, label='Treatment effect (A\u209c \u2192 L\u209c\u208a\u2081)'),
        ]
        if len(cycles_to_show) > 1:
            legend_handles.extend([
                Line2D([0], [0], color=self._EDGE_STYLES['outcome_to_first_state']['color'],
                       lw=2.2, label='Inter-cycle feedback (Y\u2096 \u2192 L\u2081)'),
                Line2D([0], [0], color=self._EDGE_STYLES['final_state_to_first_state']['color'],
                       lw=1.8, label='Final state carry-over (L\u209c \u2192 L\u2081)'),
            ])
        ax.legend(handles=legend_handles, loc='lower right', fontsize=7.5,
                  framealpha=0.92, edgecolor='#C0CCD8', fancybox=True, ncol=2)

        ax.axis('off')
        ax.margins(0.10)
        fig.tight_layout(pad=2.0)
        return fig

    # =========================================================================
    # SUMMARY & DISPLAY
    # =========================================================================

    def summary(self) -> str:
        """Return a human-readable summary of the DAG structure."""
        if self.dag is None or self.metadata is None:
            return "No DAG built."

        m = self.metadata
        carryover = "ALL-to-ALL" if m['cross_var_carryover'] else "self-only"
        lines = [
            "=" * 70,
            "  DAG Summary: Longitudinal Causal Graph (DTR Framework)",
            "=" * 70,
            f"  Player ID:         {m['player_id']}",
            f"  Cycles:            {m['n_cycles']} (lengths: {m['cycle_lengths']})",
            f"  State vars (Lt):   {len(m['state_vars'])} -- {', '.join(m['state_vars'])}",
            f"  Treatment (At):    {m['daily_treatment_var']}",
            f"  Outcome (Y):       {m['outcome_var']}",
            f"  Carryover type:    {carryover}",
            "-" * 70,
        ]

        for info in m['per_cycle']:
            k = info['cycle']
            n = info['n_days']
            total = (info['n_covariate_nodes']
                     + info['n_treatment_nodes'] + info['n_outcome_nodes'])
            lines.append(
                f"  Cycle {k} ({n:>2d} days):  "
                f"{info['n_covariate_nodes']:>3d} covariate | "
                f"{info['n_treatment_nodes']:>2d} treatment | "
                f"1 outcome  =  {total} nodes"
            )

        lines.append("-" * 70)
        lines.append(f"  Total nodes:       {m['n_nodes']}")
        lines.append(f"  Total edges:       {m['n_edges']}")

        relation_counts: Dict[str, int] = {}
        for _, _, attrs in self.dag.edges(data=True):
            rel = attrs.get('relation', 'unknown')
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        lines.append("  Edge breakdown:")
        for rel, count in sorted(relation_counts.items()):
            lines.append(f"    {rel:<28s} {count:>4d}")

        if m['n_feedback_edges'] > 0:
            lines.append(
                f"  Inter-cycle feedback:  {m['n_cycles'] - 1} connection(s), "
                f"{m['n_feedback_edges']} edge(s)"
            )

        tvc = self.get_time_varying_confounders()
        if tvc:
            lines.append(f"  Time-varying confounders: {len(tvc)} nodes")
            lines.append("    (nodes that are both effects of A(t-1) and causes of A(t)")
            lines.append("     require G-methods for unbiased causal estimation)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.metadata is None:
            return "DAGCreator(no DAG built)"
        m = self.metadata
        return (
            f"DAGCreator("
            f"player={m['player_id']}, "
            f"cycles={m['n_cycles']}, "
            f"lengths={m['cycle_lengths']}, "
            f"state_vars={len(m['state_vars'])}, "
            f"nodes={m['n_nodes']}, "
            f"edges={m['n_edges']}, "
            f"cross_carryover={m['cross_var_carryover']}"
            f")"
        )


# =============================================================================
# STANDALONE EXECUTION -- DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    script_dir = _Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Project-aligned variable configuration
    # These should be set by higher-level orchestration in real experiments
    STATE_VARS = [
        'Fatigue (z)',
        'Readiness (z)',
        'Soreness (z)',
        'Days Until Match',
    ]
    TREATMENT = 'Training Intensity Score'
    OUTCOME   = 'Match Performance'
    PLAYER_ID = 1   # Change to any valid player ID (1-27)

    print("=" * 70)
    print(f"  Building player-specific DAG for Player {PLAYER_ID}")
    print("=" * 70)

    # Create player-specific DAGCreator (auto-detects cycle lengths from data)
    creator = DAGCreator(
        player_id=PLAYER_ID,
        state_vars=STATE_VARS,
        treatment_var=TREATMENT,
        outcome_var=OUTCOME,
        cross_var_carryover=False,   # Self-only carryover (default)
    )

    print(creator.summary())
    print(repr(creator))

    print(f"\nAuto-detected cycle lengths: {creator.cycle_lengths}")

    # Object role: query examples
    outcome_c1 = creator.get_cycle_outcome(1)
    print(f"\nCausal parents of {outcome_c1}:")
    for p in creator.get_parents(outcome_c1):
        print(f"  <- {p}")

    tvc = creator.get_time_varying_confounders()
    print(f"\nTime-varying confounders ({len(tvc)} nodes, require G-methods):")
    for n in tvc[:6]:
        print(f"  {n}")
    if len(tvc) > 6:
        print(f"  ... ({len(tvc)} total)")

    adj = creator.to_adjacency_matrix()
    print(f"\nAdjacency matrix: {adj.shape} ({adj.values.sum()} directed edges)")

    print("\n\n" + "=" * 70)
    print("  Generating visualizations")
    print("=" * 70)

    # (a) All cycles, schematic
    print("\n[a] All cycles -- schematic (collapsed Player State)...")
    creator.visualize(
        cycles=None,
        completeness='schematic',
        save_path=str(results_dir / "dag_all_cycles_schematic.png"),
        show=False,
    )

    # (b) First 2 cycles, schematic
    if len(creator.cycle_lengths) >= 2:
        print("\n[b] First 2 cycles -- schematic...")
        creator.visualize(
            cycles=[1, 2],
            completeness='schematic',
            save_path=str(results_dir / "dag_cycles_1_2_schematic.png"),
            show=False,
        )

    # (c) Cycle 1 only, detailed (individual variable nodes + grouping boxes)
    print("\n[c] Cycle 1 -- detailed (all individual state variable nodes)...")
    creator.visualize(
        cycles=1,
        completeness='detailed',
        save_path=str(results_dir / "dag_cycle1_detailed.png"),
        show=False,
    )

    # (d) Cycle 1, detailed, with ALL-to-ALL carryover
    print("\n[d] Cycle 1 -- detailed (ALL-to-ALL cross-variable carryover)...")
    creator_cross = DAGCreator(
        player_id=PLAYER_ID,
        state_vars=STATE_VARS,
        treatment_var=TREATMENT,
        outcome_var=OUTCOME,
        cross_var_carryover=True,
    )
    creator_cross.visualize(
        cycles=1,
        completeness='detailed',
        save_path=str(results_dir / "dag_cycle1_detailed_cross_carryover.png"),
        show=False,
    )

    print("\n" + "=" * 70)
    print(f"  All visualizations saved to: {results_dir}")
    print("=" * 70)
