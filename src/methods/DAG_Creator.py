"""
DAG Creator for Causal Analysis of Longitudinal Match Cycles

Dynamically builds causal Directed Acyclic Graphs (DAGs) for player readiness
modelling over multi-day training cycles. The DAG encodes the assumed causal
structure between player state (baseline/covariates), daily treatments, and
match-day performance (outcome), including inter-cycle feedback where match
performance affects the player's state entering the next cycle.

=== DESIGN PHILOSOPHY ===

The class is variable-name agnostic: it receives lists of variable names
from a higher-level orchestrator and builds the temporal DAG structure
programmatically. This means:
- No variable names are hardcoded
- The same class works for any set of state variables, treatments, or outcomes
- The DAG structure (which nodes cause which) is determined by the temporal
  unrolling logic, not by domain-specific knowledge baked into this class

=== UNIFIED STATE CONCEPT ===

Baseline variables, daily covariates, and post-match state all represent the
same underlying concept: the player's state, measured through summarizing
variables. The parameter `state_vars` captures all of these:
- At t0 of each cycle: state variables serve as the BASELINE (role='baseline')
- At t1..tN of each cycle: state variables serve as COVARIATES (role='covariate')
- After the match: the outcome causally affects the state entering the next cycle

=== MULTI-CYCLE CAUSAL STRUCTURE ===

The DAG supports multiple consecutive match cycles with variable lengths,
connected by outcome-to-baseline feedback edges:

    CYCLE 1 (e.g. 5 days)                       CYCLE 2 (e.g. 7 days)
    =====================                        =====================

    State_c1_t0 (baseline)                       State_c2_t0 (baseline)
        |                                            |
        v                                            v
    State_c1_t1 --> Treat_c1_t1                  State_c2_t1 --> Treat_c2_t1
        |               |                           |               |
        v               v                           v               v
    State_c1_t2 --> Treat_c1_t2                  State_c2_t2 --> Treat_c2_t2
        |               |                           |               |
        :               :                           :               :
        v               v                           v               v
    State_c1_t5 --> Treat_c1_t5                  State_c2_t7 --> Treat_c2_t7
        |               |                           |               |
        v               v                           v               v
        +-------+-------+                           +-------+-------+
                |                                            |
                v                                            v
          Outcome_c1  -------(feedback)-------->       Outcome_c2
       (match performance)  outcome_to_baseline     (match performance)

Within each cycle:
1. Baseline --> Day 1 state (initial conditions)
2. Day t state --> Day t treatment (confounding: coaches decide based on state)
3. Day t treatment --> Day t+1 state (causal effect of training)
4. Day t state --> Day t+1 state (state carry-over / persistence)
5. Final day treatment + state --> Outcome (match performance)

Between cycles:
6. Outcome_c{k} --> Baseline_c{k+1} (match performance affects next cycle's
   starting state — e.g. physical toll of a match causes post-match fatigue)

=== VARIABLE-LENGTH CYCLES ===

Each player may have different cycle lengths depending on their match schedule.
A player selected for every match might have cycles of [5, 5, 6, 5], while
a player who misses a match might have [10, 5, 7] (the 10-day cycle spans
two team match days but the player only played the second one). The
`cycle_lengths` parameter accepts a list of integers to model this.

Usage:
    from src.methods.DAG_Creator import DAGCreator

    creator = DAGCreator()

    # Single cycle
    dag = creator.build_dag(
        cycle_lengths=5,
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )

    # Multi-cycle with variable lengths
    dag = creator.build_dag(
        cycle_lengths=[5, 7, 5],
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )

    # Query feedback edges between cycles
    feedback = creator.get_feedback_edges()
    # [('Status Decrease_c1', 'Fatigue (z)_c2_t0'), ...]
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple


class DAGCreator:
    """
    Dynamically builds causal DAGs for longitudinal match cycles.

    The DAG is constructed by 'unrolling' a temporal template over one or
    more cycles of variable length, creating time-indexed nodes and directed
    edges that encode the assumed causal relationships between player state,
    daily treatments, and match-day performance.

    Multi-cycle DAGs include feedback edges where the outcome of each cycle
    causally affects the baseline state of the next cycle, capturing how
    match performance influences the player's recovery and starting state
    for the following training block.

    All variable names are provided externally — nothing is hardcoded.

    Attributes
    ----------
    dag : nx.DiGraph or None
        The most recently built DAG. None before build_dag() is called.
    metadata : dict or None
        Metadata about the most recently built DAG.
    """

    def __init__(self):
        self.dag: Optional[nx.DiGraph] = None
        self.metadata: Optional[Dict[str, Any]] = None

    # =====================================================================
    # NODE NAMING
    # =====================================================================

    @staticmethod
    def _node_name(var: str, cycle: int, day: int) -> str:
        """
        Create a node name for a state or treatment variable.

        Format: '{var}_c{cycle}_t{day}'

        Parameters
        ----------
        var : str
            The variable name (e.g., 'Fatigue (z)').
        cycle : int
            The cycle index (1-based).
        day : int
            The day index within the cycle (0 for baseline, 1..N for daily).

        Returns
        -------
        str
            Node name, e.g. 'Fatigue (z)_c1_t3'.
        """
        return f"{var}_c{cycle}_t{day}"

    @staticmethod
    def _outcome_node_name(var: str, cycle: int) -> str:
        """
        Create a node name for an outcome variable.

        Format: '{var}_c{cycle}'

        Parameters
        ----------
        var : str
            The outcome variable name (e.g., 'Status Decrease').
        cycle : int
            The cycle index (1-based).

        Returns
        -------
        str
            Node name, e.g. 'Status Decrease_c1'.
        """
        return f"{var}_c{cycle}"

    # =====================================================================
    # SINGLE-CYCLE CONSTRUCTION (PRIVATE)
    # =====================================================================

    def _build_single_cycle(
        self,
        G: nx.DiGraph,
        cycle: int,
        n_days: int,
        state_vars: List[str],
        daily_treatment_var: str,
        outcome_var: str,
        absolute_time_offset: int,
    ) -> Dict[str, Any]:
        """
        Build nodes and intra-cycle edges for a single match cycle.

        Parameters
        ----------
        G : nx.DiGraph
            The graph to add nodes and edges to (modified in place).
        cycle : int
            The cycle index (1-based).
        n_days : int
            Number of training days in this cycle.
        state_vars : list of str
            Player state variables (baseline at t0, covariates at t1..tN).
        daily_treatment_var : str
            Treatment variable name.
        outcome_var : str
            Outcome variable name (match-day performance).
        absolute_time_offset : int
            The absolute time index for t0 of this cycle.

        Returns
        -------
        dict
            Contains:
            - 'baseline_nodes': list of baseline node names
            - 'outcome_node': the outcome node name
            - 'absolute_time_end': absolute time of the outcome node
        """
        # ---------------------------------------------------------------
        # Step 1: Create baseline nodes (state_vars at t0)
        # ---------------------------------------------------------------
        baseline_nodes = []
        for var in state_vars:
            node = self._node_name(var, cycle, 0)
            G.add_node(node, variable=var, cycle=cycle, day=0,
                       absolute_time=absolute_time_offset, role='baseline')
            baseline_nodes.append(node)

        # ---------------------------------------------------------------
        # Step 2: Create daily covariate + treatment nodes (t1..tN)
        # ---------------------------------------------------------------
        for day in range(1, n_days + 1):
            abs_time = absolute_time_offset + day

            # Covariate nodes (state at this day)
            for var in state_vars:
                node = self._node_name(var, cycle, day)
                G.add_node(node, variable=var, cycle=cycle, day=day,
                           absolute_time=abs_time, role='covariate')

            # Treatment node
            treat_node = self._node_name(daily_treatment_var, cycle, day)
            G.add_node(treat_node, variable=daily_treatment_var,
                       cycle=cycle, day=day, absolute_time=abs_time,
                       role='treatment')

        # ---------------------------------------------------------------
        # Step 3: Create outcome node (match-day performance)
        # ---------------------------------------------------------------
        outcome_abs_time = absolute_time_offset + n_days + 1
        outcome_node = self._outcome_node_name(outcome_var, cycle)
        G.add_node(outcome_node, variable=outcome_var, cycle=cycle,
                   day=n_days + 1, absolute_time=outcome_abs_time,
                   role='outcome')

        # ---------------------------------------------------------------
        # Step 4: Baseline --> Day 1 covariates
        # ---------------------------------------------------------------
        for baseline_var in state_vars:
            b_node = self._node_name(baseline_var, cycle, 0)
            for cov_var in state_vars:
                c_node = self._node_name(cov_var, cycle, 1)
                G.add_edge(b_node, c_node,
                           relation='baseline_to_covariate')

        # ---------------------------------------------------------------
        # Step 5: Daily structure (days 1..N)
        # ---------------------------------------------------------------
        for day in range(1, n_days + 1):
            treat_node = self._node_name(daily_treatment_var, cycle, day)

            # 5a. Covariates --> Treatment (confounding)
            for var in state_vars:
                cov_node = self._node_name(var, cycle, day)
                G.add_edge(cov_node, treat_node,
                           relation='confounding')

            # 5b. Carry-over to next day (if not last day)
            if day < n_days:
                next_day = day + 1

                # Treatment --> next-day covariates (treatment effect)
                for var in state_vars:
                    next_cov = self._node_name(var, cycle, next_day)
                    G.add_edge(treat_node, next_cov,
                               relation='treatment_effect')

                # Covariates --> next-day covariates (state carry-over)
                for var in state_vars:
                    curr_cov = self._node_name(var, cycle, day)
                    next_cov = self._node_name(var, cycle, next_day)
                    G.add_edge(curr_cov, next_cov,
                               relation='state_carryover')

        # ---------------------------------------------------------------
        # Step 6: Final day --> Outcome
        # ---------------------------------------------------------------
        # Last treatment --> outcome
        final_treat = self._node_name(daily_treatment_var, cycle, n_days)
        G.add_edge(final_treat, outcome_node,
                   relation='treatment_to_outcome')

        # Last day covariates --> outcome
        for var in state_vars:
            final_cov = self._node_name(var, cycle, n_days)
            G.add_edge(final_cov, outcome_node,
                       relation='covariate_to_outcome')

        return {
            'baseline_nodes': baseline_nodes,
            'outcome_node': outcome_node,
            'absolute_time_end': outcome_abs_time,
        }

    # =====================================================================
    # PUBLIC API: DAG CONSTRUCTION
    # =====================================================================

    def build_dag(
        self,
        cycle_lengths: Union[int, List[int]],
        state_vars: List[str],
        daily_treatment_var: str,
        outcome_var: str,
    ) -> nx.DiGraph:
        """
        Build a causal DAG by unrolling one or more match cycles.

        Each cycle represents a training block leading up to a match. The
        outcome at the end of each cycle represents match-day performance.
        In multi-cycle DAGs, the outcome of cycle k feeds back into the
        baseline state of cycle k+1, capturing how match performance
        influences recovery and starting state for the next training block.

        Cycles can have different lengths to model the reality that players
        have variable schedules (some play every match, some skip matches,
        international breaks cause longer gaps, etc.).

        Parameters
        ----------
        cycle_lengths : int or list of int
            Length(s) of the match cycle(s) in days. A single integer creates
            one cycle (e.g., 5). A list creates multiple connected cycles
            (e.g., [5, 7, 5] creates 3 cycles of 5, 7, and 5 days).
            Each value must be >= 1.
        state_vars : list of str
            Player state variables that appear as baseline (t0 of each
            cycle) and as daily covariates (t1..tN of each cycle). These
            represent the player's condition through summarizing variables
            like wellness z-scores, physical state, etc.
        daily_treatment_var : str
            The treatment (intervention) variable name instantiated for
            each day of each cycle (e.g., 'Activity Type Today').
        outcome_var : str
            The outcome variable at the end of each cycle, representing
            match-day performance (e.g., 'Status Decrease'). In multi-cycle
            DAGs, this outcome feeds back into the next cycle's baseline.

        Returns
        -------
        nx.DiGraph
            A directed acyclic graph with time-indexed nodes and causal edges.
            Node attributes: 'variable', 'cycle', 'day', 'absolute_time', 'role'.
            Edge attributes: 'relation'.

        Raises
        ------
        ValueError
            If any cycle length < 1, variable lists are empty, or treatment
            variable overlaps with state variables.
        """
        # -----------------------------------------------------------------
        # Normalize cycle_lengths
        # -----------------------------------------------------------------
        if isinstance(cycle_lengths, int):
            cycle_lengths = [cycle_lengths]

        # -----------------------------------------------------------------
        # Input validation
        # -----------------------------------------------------------------
        if not cycle_lengths:
            raise ValueError("cycle_lengths must not be empty")
        for i, length in enumerate(cycle_lengths):
            if length < 1:
                raise ValueError(
                    f"All cycle lengths must be >= 1, got {length} "
                    f"at index {i}"
                )
        if not state_vars:
            raise ValueError("state_vars must not be empty")
        if not daily_treatment_var:
            raise ValueError("daily_treatment_var must not be empty")
        if not outcome_var:
            raise ValueError("outcome_var must not be empty")
        if daily_treatment_var in state_vars:
            raise ValueError(
                f"daily_treatment_var '{daily_treatment_var}' must not "
                f"appear in state_vars"
            )

        # -----------------------------------------------------------------
        # Build graph
        # -----------------------------------------------------------------
        G = nx.DiGraph()

        n_cycles = len(cycle_lengths)
        absolute_time_offset = 0
        cycle_info = []  # Track per-cycle metadata
        prev_outcome_node = None

        for k in range(1, n_cycles + 1):
            n_days = cycle_lengths[k - 1]

            # Build this cycle's internal structure
            result = self._build_single_cycle(
                G, cycle=k, n_days=n_days,
                state_vars=state_vars,
                daily_treatment_var=daily_treatment_var,
                outcome_var=outcome_var,
                absolute_time_offset=absolute_time_offset,
            )

            # Add feedback edges from previous cycle's outcome
            if prev_outcome_node is not None:
                for var in state_vars:
                    baseline_node = self._node_name(var, k, 0)
                    G.add_edge(prev_outcome_node, baseline_node,
                               relation='outcome_to_baseline')

            # Track metadata
            cycle_info.append({
                'cycle': k,
                'n_days': n_days,
                'n_baseline_nodes': len(state_vars),
                'n_covariate_nodes': len(state_vars) * n_days,
                'n_treatment_nodes': n_days,
                'n_outcome_nodes': 1,
            })

            prev_outcome_node = result['outcome_node']
            absolute_time_offset = result['absolute_time_end'] + 1

        # -----------------------------------------------------------------
        # Validate DAG (must be acyclic)
        # -----------------------------------------------------------------
        if not nx.is_directed_acyclic_graph(G):
            raise RuntimeError(
                "Constructed graph contains cycles — this should not happen "
                "with the temporal unrolling logic. Please report this as a bug."
            )

        # -----------------------------------------------------------------
        # Store results
        # -----------------------------------------------------------------
        n_feedback = sum(
            1 for _, _, d in G.edges(data=True)
            if d.get('relation') == 'outcome_to_baseline'
        )

        self.dag = G
        self.metadata = {
            'n_cycles': n_cycles,
            'cycle_lengths': list(cycle_lengths),
            'state_vars': list(state_vars),
            'daily_treatment_var': daily_treatment_var,
            'outcome_var': outcome_var,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_feedback_edges': n_feedback,
            'per_cycle': cycle_info,
        }

        return G

    # =====================================================================
    # QUERY METHODS
    # =====================================================================

    def _check_dag(self):
        """Raise RuntimeError if no DAG has been built yet."""
        if self.dag is None:
            raise RuntimeError("No DAG built yet. Call build_dag() first.")

    def get_nodes_by_role(self, role: str) -> List[str]:
        """
        Get all node names with a given role.

        Parameters
        ----------
        role : str
            One of 'baseline', 'covariate', 'treatment', 'outcome'.

        Returns
        -------
        list of str
            Node names matching the specified role, sorted by absolute time.
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
        """
        Get all node names at a specific absolute time index.

        Parameters
        ----------
        time : int
            The absolute time index across all cycles.

        Returns
        -------
        list of str
            Node names at the specified absolute time, sorted alphabetically.
        """
        self._check_dag()
        return sorted([
            n for n, attrs in self.dag.nodes(data=True)
            if attrs.get('absolute_time') == time
        ])

    def get_nodes_in_cycle(self, cycle: int) -> List[str]:
        """
        Get all nodes belonging to a specific cycle.

        Parameters
        ----------
        cycle : int
            The cycle index (1-based).

        Returns
        -------
        list of str
            Node names in the specified cycle, sorted by day then variable.
        """
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
        """
        Get all nodes at a specific day within a specific cycle.

        Parameters
        ----------
        cycle : int
            The cycle index (1-based).
        day : int
            The day index within the cycle (0 for baseline, 1..N for daily,
            N+1 for outcome).

        Returns
        -------
        list of str
            Node names at the specified cycle and day, sorted alphabetically.
        """
        self._check_dag()
        return sorted([
            n for n, attrs in self.dag.nodes(data=True)
            if attrs.get('cycle') == cycle and attrs.get('day') == day
        ])

    def get_cycle_outcome(self, cycle: int) -> str:
        """
        Get the outcome node name for a given cycle.

        Parameters
        ----------
        cycle : int
            The cycle index (1-based).

        Returns
        -------
        str
            The outcome node name (e.g., 'Status Decrease_c1').

        Raises
        ------
        ValueError
            If the cycle index is out of range.
        """
        self._check_dag()
        if self.metadata is None:
            raise RuntimeError("No metadata available.")
        outcome_var = self.metadata['outcome_var']
        n_cycles = self.metadata['n_cycles']
        if cycle < 1 or cycle > n_cycles:
            raise ValueError(
                f"cycle must be between 1 and {n_cycles}, got {cycle}"
            )
        return self._outcome_node_name(outcome_var, cycle)

    def get_feedback_edges(self) -> List[tuple]:
        """
        Get all outcome-to-baseline feedback edges between cycles.

        Returns
        -------
        list of tuple
            (source, target) pairs where source is an outcome node and
            target is a baseline node of the next cycle.
        """
        return self.get_edges_by_relation('outcome_to_baseline')

    def get_parents(self, node: str) -> List[str]:
        """
        Get the parent nodes (direct causes) of a given node.

        Parameters
        ----------
        node : str
            The node name.

        Returns
        -------
        list of str
            Parent node names, sorted alphabetically.
        """
        self._check_dag()
        return sorted(list(self.dag.predecessors(node)))

    def get_children(self, node: str) -> List[str]:
        """
        Get the child nodes (direct effects) of a given node.

        Parameters
        ----------
        node : str
            The node name.

        Returns
        -------
        list of str
            Child node names, sorted alphabetically.
        """
        self._check_dag()
        return sorted(list(self.dag.successors(node)))

    def get_edges_by_relation(self, relation: str) -> List[tuple]:
        """
        Get all edges with a specific relation type.

        Parameters
        ----------
        relation : str
            One of 'baseline_to_covariate', 'confounding',
            'treatment_effect', 'state_carryover',
            'treatment_to_outcome', 'covariate_to_outcome',
            'outcome_to_baseline'.

        Returns
        -------
        list of tuple
            (source, target) edge pairs matching the relation.
        """
        self._check_dag()
        return [
            (u, v) for u, v, attrs in self.dag.edges(data=True)
            if attrs.get('relation') == relation
        ]

    # =====================================================================
    # VISUALIZATION
    # =====================================================================

    # --- Color palette ---
    # Consistent across all visualization methods for recognizability
    _COLORS = {
        'baseline':   '#4CAF50',   # Green — player state at cycle start
        'covariate':  '#2196F3',   # Blue — daily player state
        'treatment':  '#FF9800',   # Orange — training intensity
        'outcome':    '#E53935',   # Red — match-day performance
        'feedback':   '#9C27B0',   # Purple — inter-cycle feedback edges
        'background': '#FAFAFA',   # Light grey — figure background
        'cycle_bg':   '#F0F0F0',   # Cycle background box
        'text':       '#1a1a2e',   # Dark text
        'edge':       '#666666',   # Default edge color
    }

    # --- Edge style mapping ---
    _EDGE_STYLES = {
        'baseline_to_covariate': {'color': '#4CAF50', 'style': '-',  'alpha': 0.5, 'width': 1.0},
        'confounding':           {'color': '#FF9800', 'style': '--', 'alpha': 0.7, 'width': 1.2},
        'treatment_effect':      {'color': '#E53935', 'style': '-',  'alpha': 0.7, 'width': 1.5},
        'state_carryover':       {'color': '#2196F3', 'style': '-',  'alpha': 0.4, 'width': 1.0},
        'treatment_to_outcome':  {'color': '#E53935', 'style': '-',  'alpha': 0.8, 'width': 1.8},
        'covariate_to_outcome':  {'color': '#2196F3', 'style': '-',  'alpha': 0.6, 'width': 1.2},
        'outcome_to_baseline':   {'color': '#9C27B0', 'style': '-',  'alpha': 0.9, 'width': 2.0},
    }

    def _get_node_color(self, role: str) -> str:
        """Map node role to color."""
        return self._COLORS.get(role, '#CCCCCC')

    def _compute_layout(
        self,
        nodes: List[str],
        cycle: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute (x, y) positions for nodes using a temporal left-to-right layout.

        Within each time step (column), nodes are stacked vertically:
        state variables on top, treatment below. The outcome node is placed
        at the far right.

        Parameters
        ----------
        nodes : list of str
            Node names to position.
        cycle : int or None
            If provided, only position nodes from this cycle.

        Returns
        -------
        dict
            {node_name: (x, y)} positions.
        """
        G = self.dag
        pos = {}

        # Filter nodes if cycle specified
        if cycle is not None:
            nodes = [n for n in nodes if G.nodes[n].get('cycle') == cycle]

        if not nodes:
            return pos

        # Group nodes by day
        day_groups: Dict[int, List[str]] = {}
        for n in nodes:
            day = G.nodes[n].get('day', 0)
            day_groups.setdefault(day, [])
            day_groups[day].append(n)

        # Sort days
        sorted_days = sorted(day_groups.keys())

        # Compute vertical positions: state vars on top, treatment below, outcome centered
        state_vars = self.metadata['state_vars']
        n_state = len(state_vars)

        x_spacing = 2.0
        y_spacing = 1.2

        for col_idx, day in enumerate(sorted_days):
            x = col_idx * x_spacing
            day_nodes = day_groups[day]

            # Separate by role
            state_nodes = [n for n in day_nodes if G.nodes[n].get('role') in ('baseline', 'covariate')]
            treat_nodes = [n for n in day_nodes if G.nodes[n].get('role') == 'treatment']
            outcome_nodes = [n for n in day_nodes if G.nodes[n].get('role') == 'outcome']

            # Sort state nodes by variable order (same order as state_vars)
            var_order = {v: i for i, v in enumerate(state_vars)}
            state_nodes.sort(key=lambda n: var_order.get(G.nodes[n].get('variable', ''), 999))

            # Position state nodes (top, stacked vertically)
            total_height = (n_state - 1) * y_spacing
            for i, n in enumerate(state_nodes):
                y = total_height / 2 - i * y_spacing
                pos[n] = (x, y)

            # Position treatment node (below state nodes)
            for i, n in enumerate(treat_nodes):
                y = -total_height / 2 - y_spacing * (1 + i)
                pos[n] = (x, y)

            # Position outcome node (vertically centered)
            for n in outcome_nodes:
                pos[n] = (x, 0)

        return pos

    def visualize(
        self,
        mode: str = 'schematic',
        cycle: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        title: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualize the causal DAG.

        Supports three visualization modes:

        1. 'schematic' (default): Collapsed view where all state variables
           are summarized into a single "Player State" node per time step.
           Treatment and outcome are single nodes. This produces the clean,
           publication-ready DAG diagram. Best for presentations and papers.

        2. 'detailed': Full expanded view showing every individual node
           (each state variable, treatment, outcome) and all edges. Can be
           very large for multi-cycle DAGs with many state variables.

        3. 'single_cycle': Shows one specific cycle in detail. Requires
           the `cycle` parameter. Useful for examining the internal
           structure of a specific training block.

        Parameters
        ----------
        mode : str, default='schematic'
            Visualization mode: 'schematic', 'detailed', or 'single_cycle'.
        cycle : int or None
            For 'single_cycle' mode: which cycle to visualize (1-based).
            For 'schematic': if provided, only show that cycle's schematic.
            Ignored for 'detailed' mode (always shows everything).
        figsize : tuple of (float, float) or None
            Figure size in inches (width, height). If None, auto-computed
            based on DAG complexity.
        save_path : str or None
            If provided, save the figure to this path (supports .png, .pdf, .svg).
        dpi : int, default=150
            Resolution for saved figures.
        title : str or None
            Custom title. If None, auto-generated from DAG metadata.
        show : bool, default=True
            Whether to call plt.show(). Set False for non-interactive use.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.

        Raises
        ------
        RuntimeError
            If no DAG has been built yet.
        ValueError
            If mode is invalid or cycle is out of range.
        """
        self._check_dag()

        if mode == 'schematic':
            fig = self._visualize_schematic(cycle=cycle, figsize=figsize, title=title)
        elif mode == 'detailed':
            fig = self._visualize_detailed(cycle=cycle, figsize=figsize, title=title)
        elif mode == 'single_cycle':
            if cycle is None:
                raise ValueError("cycle parameter is required for 'single_cycle' mode")
            fig = self._visualize_detailed(cycle=cycle, figsize=figsize, title=title)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'schematic', 'detailed', or 'single_cycle'.")

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')

        if show:
            plt.show()

        return fig

    def _visualize_detailed(
        self,
        cycle: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Detailed visualization showing every individual node and edge.

        Each state variable gets its own node. All edges from the DAG are
        drawn with colors and styles indicating the edge relation type.

        Parameters
        ----------
        cycle : int or None
            If provided, only show nodes/edges from this cycle.
        figsize : tuple or None
            Figure size. Auto-computed if None.
        title : str or None
            Plot title. Auto-generated if None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        G = self.dag
        m = self.metadata

        # Determine which nodes to show
        if cycle is not None:
            n_cycles = m['n_cycles']
            if cycle < 1 or cycle > n_cycles:
                raise ValueError(f"cycle must be between 1 and {n_cycles}, got {cycle}")
            nodes = self.get_nodes_in_cycle(cycle)
            subgraph = G.subgraph(nodes)
        else:
            nodes = list(G.nodes())
            subgraph = G

        # Compute layout
        pos = self._compute_layout(nodes, cycle=cycle)

        # Auto figure size
        if figsize is None:
            n_cols = len(set(G.nodes[n].get('day', 0) for n in nodes))
            n_rows = len(m['state_vars']) + 1  # state vars + treatment
            if cycle is None:
                # Multi-cycle: account for all cycles
                total_cols = sum(length + 2 for length in m['cycle_lengths'])
                figsize = (max(12, total_cols * 1.8), max(6, n_rows * 1.5))
            else:
                figsize = (max(10, (n_cols + 1) * 2.0), max(6, n_rows * 1.5))

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.set_facecolor(self._COLORS['background'])
        ax.set_facecolor(self._COLORS['background'])

        # --- Draw cycle background boxes ---
        if cycle is not None:
            cycles_to_draw = [cycle]
        else:
            cycles_to_draw = list(range(1, m['n_cycles'] + 1))

        for k in cycles_to_draw:
            cycle_nodes = [n for n in nodes if G.nodes[n].get('cycle') == k]
            if not cycle_nodes:
                continue
            cycle_pos = [pos[n] for n in cycle_nodes if n in pos]
            if not cycle_pos:
                continue

            xs = [p[0] for p in cycle_pos]
            ys = [p[1] for p in cycle_pos]
            pad = 0.8
            rect = FancyBboxPatch(
                (min(xs) - pad, min(ys) - pad),
                max(xs) - min(xs) + 2 * pad,
                max(ys) - min(ys) + 2 * pad,
                boxstyle="round,pad=0.3",
                facecolor=self._COLORS['cycle_bg'],
                edgecolor='#CCCCCC',
                linewidth=1.0,
                alpha=0.5,
                zorder=0,
            )
            ax.add_patch(rect)
            # Cycle label
            ax.text(
                (min(xs) + max(xs)) / 2, max(ys) + pad + 0.3,
                f"Cycle {k} ({m['cycle_lengths'][k-1]} days)",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=self._COLORS['text'], zorder=5,
            )

        # --- Draw edges ---
        for u, v, data in subgraph.edges(data=True):
            if u not in pos or v not in pos:
                continue
            relation = data.get('relation', 'unknown')
            style_info = self._EDGE_STYLES.get(relation, {
                'color': self._COLORS['edge'], 'style': '-', 'alpha': 0.3, 'width': 0.8
            })
            ax.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=dict(
                    arrowstyle='->', color=style_info['color'],
                    lw=style_info['width'], alpha=style_info['alpha'],
                    connectionstyle='arc3,rad=0.05',
                    linestyle=style_info['style'],
                ),
                zorder=1,
            )

        # --- Draw nodes ---
        for n in nodes:
            if n not in pos:
                continue
            x, y = pos[n]
            role = G.nodes[n].get('role', 'unknown')
            variable = G.nodes[n].get('variable', n)
            day = G.nodes[n].get('day', 0)
            color = self._get_node_color(role)

            # Node shape: ellipse for state/baseline, rectangle for treatment, diamond for outcome
            if role in ('baseline', 'covariate'):
                shape = 'ellipse'
                node_size = 0.55
            elif role == 'treatment':
                shape = 'rectangle'
                node_size = 0.55
            else:  # outcome
                shape = 'diamond'
                node_size = 0.7

            if shape == 'ellipse':
                ellipse = mpatches.Ellipse(
                    (x, y), node_size * 1.6, node_size,
                    facecolor=color, edgecolor='white', linewidth=1.5,
                    alpha=0.9, zorder=3,
                )
                ax.add_patch(ellipse)
            elif shape == 'rectangle':
                rect = FancyBboxPatch(
                    (x - node_size * 0.75, y - node_size * 0.4),
                    node_size * 1.5, node_size * 0.8,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor='white', linewidth=1.5,
                    alpha=0.9, zorder=3,
                )
                ax.add_patch(rect)
            else:  # diamond
                diamond = mpatches.RegularPolygon(
                    (x, y), numVertices=4, radius=node_size * 0.6,
                    orientation=0,
                    facecolor=color, edgecolor='white', linewidth=2,
                    alpha=0.9, zorder=3,
                )
                ax.add_patch(diamond)

            # Label: shortened variable name + day indicator
            if role == 'outcome':
                label = variable
            else:
                # Shorten variable names for readability
                short_name = variable.replace(' (z)', '').replace(' Yesterday', '')
                if len(short_name) > 12:
                    short_name = short_name[:11] + '.'
                label = f"{short_name}\nt{day}"

            ax.text(
                x, y, label, ha='center', va='center',
                fontsize=6, fontweight='bold', color='white', zorder=4,
            )

        # --- Title ---
        if title is None:
            if cycle is not None:
                title = f"Causal DAG — Cycle {cycle} ({m['cycle_lengths'][cycle-1]} days)"
            else:
                title = f"Causal DAG — {m['n_cycles']} cycle(s), lengths {m['cycle_lengths']}"
        ax.set_title(title, fontsize=14, fontweight='bold', color=self._COLORS['text'], pad=20)

        # --- Legend ---
        legend_elements = [
            mpatches.Patch(facecolor=self._COLORS['baseline'], label='Baseline State'),
            mpatches.Patch(facecolor=self._COLORS['covariate'], label='Daily State (Covariate)'),
            mpatches.Patch(facecolor=self._COLORS['treatment'], label='Treatment'),
            mpatches.Patch(facecolor=self._COLORS['outcome'], label='Outcome'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  framealpha=0.9, edgecolor='#CCCCCC')

        ax.set_aspect('equal')
        ax.axis('off')
        ax.margins(0.15)
        fig.tight_layout()

        return fig

    def _visualize_schematic(
        self,
        cycle: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Schematic visualization with collapsed Player State nodes.

        All state variables at each time step are collapsed into a single
        "Player State" node. This produces a clean, readable DAG suitable
        for presentations and publications.

        The visual structure per day is:

            [Player State]  ──confounding──>  [Treatment]
                 │                                │
                 │ carry-over                      │ treatment effect
                 v                                v
            [Player State]  ──confounding──>  [Treatment]
                 │          (next day)             │
                 :                                :
                 v                                v
            [Player State]  ─────────────>   [Outcome]
                 │                                │
                 │                                │ feedback
                 v                                v
            (next cycle)                    [Player State] (next cycle baseline)

        Parameters
        ----------
        cycle : int or None
            If provided, show only that cycle's schematic.
        figsize : tuple or None
            Figure size. Auto-computed if None.
        title : str or None
            Plot title. Auto-generated if None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        m = self.metadata

        # Determine cycles to draw
        if cycle is not None:
            n_cycles = m['n_cycles']
            if cycle < 1 or cycle > n_cycles:
                raise ValueError(f"cycle must be between 1 and {n_cycles}, got {cycle}")
            cycles = [cycle]
        else:
            cycles = list(range(1, m['n_cycles'] + 1))

        # Calculate layout dimensions
        total_days = sum(m['cycle_lengths'][k-1] + 2 for k in cycles)  # +2 for baseline + outcome
        if figsize is None:
            width = max(14, total_days * 1.6)
            height = max(5, 6 + (1 if len(cycles) > 1 else 0))
            figsize = (width, height)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.set_facecolor(self._COLORS['background'])
        ax.set_facecolor(self._COLORS['background'])

        # --- Build schematic nodes and edges ---
        x_offset = 0
        x_spacing = 2.2
        y_state = 1.5      # y for Player State row
        y_treatment = -1.5  # y for Treatment row

        # Track positions for drawing
        schematic_nodes = []   # (x, y, label, role, cycle_k)
        schematic_edges = []   # (x1, y1, x2, y2, relation)

        for k in cycles:
            n_days = m['cycle_lengths'][k-1]
            cycle_x_start = x_offset

            # --- Baseline node (t0) ---
            bx = x_offset
            schematic_nodes.append((bx, y_state, 'Player\nState', 'baseline', k, 't₀'))

            # --- Daily nodes (t1 .. tN) ---
            for day in range(1, n_days + 1):
                dx = x_offset + day * x_spacing

                # State node
                schematic_nodes.append((dx, y_state, 'Player\nState', 'covariate', k, f't{day}'))

                # Treatment node
                schematic_nodes.append((dx, y_treatment, 'Treatment', 'treatment', k, f't{day}'))

                # Edge: state → treatment (confounding)
                schematic_edges.append((dx, y_state, dx, y_treatment, 'confounding'))

                # Edge: previous state → current state (carry-over)
                prev_x = x_offset + (day - 1) * x_spacing
                schematic_edges.append((prev_x, y_state, dx, y_state, 'state_carryover'))

                # Edge: previous treatment → current state (treatment effect)
                if day > 1:
                    prev_tx = x_offset + (day - 1) * x_spacing
                    schematic_edges.append((prev_tx, y_treatment, dx, y_state, 'treatment_effect'))

                # Edge: baseline → day 1 state (already covered by carry-over above for day=1)
                # The baseline→covariate is the carry-over from t0 to t1

            # --- Outcome node ---
            ox = x_offset + (n_days + 1) * x_spacing
            schematic_nodes.append((ox, 0, 'Match\nOutcome', 'outcome', k, ''))

            # Edges: final state → outcome
            final_sx = x_offset + n_days * x_spacing
            schematic_edges.append((final_sx, y_state, ox, 0, 'covariate_to_outcome'))

            # Edges: final treatment → outcome
            schematic_edges.append((final_sx, y_treatment, ox, 0, 'treatment_to_outcome'))

            # --- Feedback to next cycle ---
            next_cycle_idx = cycles.index(k) + 1
            if next_cycle_idx < len(cycles):
                next_k = cycles[next_cycle_idx]
                next_x_offset = ox + x_spacing * 1.5
                # This feedback edge will be drawn after we know the next baseline position
                schematic_edges.append((ox, 0, next_x_offset, y_state, 'outcome_to_baseline'))

            # Update x_offset for next cycle
            x_offset = ox + x_spacing * 1.5

        # --- Draw cycle background boxes ---
        # Recalculate positions for backgrounds
        x_cursor = 0
        for idx, k in enumerate(cycles):
            n_days = m['cycle_lengths'][k-1]
            cycle_width = (n_days + 2) * x_spacing
            pad = 0.9

            rect = FancyBboxPatch(
                (x_cursor - pad, y_treatment - pad - 0.3),
                cycle_width - x_spacing * 0.3 + 2 * pad,
                (y_state - y_treatment) + 2 * pad + 0.6,
                boxstyle="round,pad=0.4",
                facecolor=self._COLORS['cycle_bg'],
                edgecolor='#BBBBBB',
                linewidth=1.0,
                alpha=0.4,
                zorder=0,
            )
            ax.add_patch(rect)

            # Cycle label at top
            cx = x_cursor + (cycle_width - x_spacing * 0.3) / 2
            ax.text(
                cx, y_state + pad + 0.7,
                f"Cycle {k}  ({n_days} training days)",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=self._COLORS['text'], zorder=5,
            )

            x_cursor += (n_days + 2) * x_spacing + x_spacing * 0.5

        # --- Draw edges ---
        for x1, y1, x2, y2, relation in schematic_edges:
            style_info = self._EDGE_STYLES.get(relation, {
                'color': self._COLORS['edge'], 'style': '-', 'alpha': 0.3, 'width': 0.8
            })

            # Curve radius depends on edge direction
            if abs(y2 - y1) > 0.5 and abs(x2 - x1) > 0.5:
                conn = 'arc3,rad=0.15'
            elif relation == 'outcome_to_baseline':
                conn = 'arc3,rad=-0.3'
            else:
                conn = 'arc3,rad=0.0'

            ax.annotate(
                '', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=style_info['color'],
                    lw=style_info['width'], alpha=style_info['alpha'],
                    connectionstyle=conn,
                    linestyle=style_info['style'],
                    shrinkA=18, shrinkB=18,
                ),
                zorder=1,
            )

        # --- Draw nodes ---
        for x, y, label, role, k, time_label in schematic_nodes:
            color = self._get_node_color(role)

            if role in ('baseline', 'covariate'):
                # Ellipse for Player State
                node_w, node_h = 1.6, 0.9
                ellipse = mpatches.Ellipse(
                    (x, y), node_w, node_h,
                    facecolor=color, edgecolor='white', linewidth=2,
                    alpha=0.92, zorder=3,
                )
                ax.add_patch(ellipse)
            elif role == 'treatment':
                # Rounded rectangle for Treatment
                node_w, node_h = 1.4, 0.7
                rect = FancyBboxPatch(
                    (x - node_w / 2, y - node_h / 2),
                    node_w, node_h,
                    boxstyle="round,pad=0.08",
                    facecolor=color, edgecolor='white', linewidth=2,
                    alpha=0.92, zorder=3,
                )
                ax.add_patch(rect)
            else:  # outcome
                # Diamond for Outcome
                diamond = mpatches.RegularPolygon(
                    (x, y), numVertices=4, radius=0.65,
                    orientation=0,
                    facecolor=color, edgecolor='white', linewidth=2.5,
                    alpha=0.92, zorder=3,
                )
                ax.add_patch(diamond)

            # Node text
            ax.text(
                x, y, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=4,
            )

            # Time subscript below node
            if time_label:
                sub_y = y - (0.55 if role != 'treatment' else 0.48)
                ax.text(
                    x, sub_y, time_label, ha='center', va='top',
                    fontsize=7, fontstyle='italic', color=self._COLORS['text'],
                    alpha=0.7, zorder=5,
                )

        # --- Title ---
        if title is None:
            state_names = ', '.join(m['state_vars'])
            if len(state_names) > 60:
                state_names = state_names[:57] + '...'
            if cycle is not None:
                title = f"Causal DAG (Schematic) — Cycle {cycle}"
            else:
                title = f"Causal DAG (Schematic) — {m['n_cycles']} cycle(s)"
            title += f"\nState = [{state_names}] | Treatment = {m['daily_treatment_var']} | Outcome = {m['outcome_var']}"

        ax.set_title(title, fontsize=12, fontweight='bold', color=self._COLORS['text'], pad=25)

        # --- Legend ---
        legend_elements = [
            mpatches.Patch(facecolor=self._COLORS['baseline'], label='Baseline (t₀)', edgecolor='white'),
            mpatches.Patch(facecolor=self._COLORS['covariate'], label='Player State', edgecolor='white'),
            mpatches.Patch(facecolor=self._COLORS['treatment'], label='Treatment', edgecolor='white'),
            mpatches.Patch(facecolor=self._COLORS['outcome'], label='Match Outcome', edgecolor='white'),
        ]
        # Add edge types to legend
        from matplotlib.lines import Line2D
        legend_elements.extend([
            Line2D([0], [0], color=self._EDGE_STYLES['state_carryover']['color'],
                   linewidth=1.5, label='State carry-over'),
            Line2D([0], [0], color=self._EDGE_STYLES['confounding']['color'],
                   linewidth=1.5, linestyle='--', label='Confounding (state→treatment)'),
            Line2D([0], [0], color=self._EDGE_STYLES['treatment_effect']['color'],
                   linewidth=1.5, label='Treatment effect'),
        ])
        if len(cycles) > 1:
            legend_elements.append(
                Line2D([0], [0], color=self._EDGE_STYLES['outcome_to_baseline']['color'],
                       linewidth=2, label='Inter-cycle feedback')
            )

        ax.legend(
            handles=legend_elements, loc='lower right', fontsize=7.5,
            framealpha=0.9, edgecolor='#CCCCCC', ncol=2,
        )

        ax.set_aspect('equal')
        ax.axis('off')
        ax.margins(0.08)
        fig.tight_layout()

        return fig

    def visualize_cycle(
        self,
        cycle: int,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        title: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Convenience method: visualize a single cycle in schematic mode.

        Equivalent to visualize(mode='schematic', cycle=cycle).

        Parameters
        ----------
        cycle : int
            Cycle index (1-based).
        figsize : tuple or None
            Figure size.
        save_path : str or None
            Path to save the figure.
        dpi : int
            Resolution for saved figures.
        title : str or None
            Custom title.
        show : bool
            Whether to display the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        return self.visualize(
            mode='schematic', cycle=cycle, figsize=figsize,
            save_path=save_path, dpi=dpi, title=title, show=show,
        )

    # =====================================================================
    # SUMMARY & DISPLAY
    # =====================================================================

    def summary(self) -> str:
        """
        Return a human-readable summary of the DAG structure.

        Returns
        -------
        str
            Multi-line summary string.
        """
        if self.dag is None or self.metadata is None:
            return "No DAG built yet. Call build_dag() first."

        m = self.metadata
        lines = [
            "=" * 65,
            "DAG Summary: Longitudinal Causal Graph",
            "=" * 65,
            f"  Cycles:            {m['n_cycles']} (lengths: {m['cycle_lengths']})",
            f"  State variables:   {len(m['state_vars'])} ({', '.join(m['state_vars'])})",
            f"  Daily treatment:   {m['daily_treatment_var']}",
            f"  Outcome:           {m['outcome_var']}",
            "-" * 65,
        ]

        # Per-cycle details
        for info in m['per_cycle']:
            k = info['cycle']
            n = info['n_days']
            total = (info['n_baseline_nodes'] + info['n_covariate_nodes']
                     + info['n_treatment_nodes'] + info['n_outcome_nodes'])
            lines.append(
                f"  Cycle {k} ({n} days): "
                f"{info['n_baseline_nodes']} baseline | "
                f"{info['n_covariate_nodes']} covariate | "
                f"{info['n_treatment_nodes']} treatment | "
                f"{info['n_outcome_nodes']} outcome  "
                f"= {total} nodes"
            )

        lines.append("-" * 65)
        lines.append(f"  Total nodes:       {m['n_nodes']}")
        lines.append(f"  Total edges:       {m['n_edges']}")

        # Edge type counts
        relation_counts = {}
        for _, _, attrs in self.dag.edges(data=True):
            rel = attrs.get('relation', 'unknown')
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        lines.append("  Edge types:")
        for rel, count in sorted(relation_counts.items()):
            lines.append(f"    - {rel}: {count}")

        if m['n_feedback_edges'] > 0:
            lines.append(
                f"  Feedback links:    {m['n_cycles'] - 1} inter-cycle "
                f"connections ({m['n_feedback_edges']} edges)"
            )

        lines.append("=" * 65)
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.metadata is None:
            return "DAGCreator(no DAG built)"
        m = self.metadata
        return (
            f"DAGCreator(cycles={m['n_cycles']}, "
            f"lengths={m['cycle_lengths']}, "
            f"nodes={m['n_nodes']}, edges={m['n_edges']})"
        )


# =========================================================================
# STANDALONE EXECUTION — DEMONSTRATION
# =========================================================================

if __name__ == "__main__":
    creator = DAGCreator()

    # Determine output directory for saved figures
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Demo 1: Single cycle (5 days) — text summary
    # -----------------------------------------------------------------
    print("=" * 65)
    print("DEMO 1: Single Match Cycle (5 days)")
    print("=" * 65)

    dag1 = creator.build_dag(
        cycle_lengths=5,
        state_vars=[
            'Fatigue (z)',
            'Readiness (z)',
            'Soreness (z)',
        ],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )

    print(creator.summary())

    print("\nOutcome parents (what causes match-day performance):")
    outcome = creator.get_cycle_outcome(1)
    for parent in creator.get_parents(outcome):
        print(f"  <- {parent}")

    print(f"\nIs acyclic: {nx.is_directed_acyclic_graph(dag1)}")

    # -----------------------------------------------------------------
    # Demo 2: Multi-cycle with variable lengths — text summary
    # -----------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("DEMO 2: Multi-Cycle (3 cycles: 5, 7, 5 days)")
    print("=" * 65)

    dag2 = creator.build_dag(
        cycle_lengths=[5, 7, 5],
        state_vars=[
            'Fatigue (z)',
            'Readiness (z)',
            'Soreness (z)',
        ],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )

    print(creator.summary())

    # Show feedback edges
    feedback = creator.get_feedback_edges()
    print(f"\nFeedback edges ({len(feedback)} total):")
    for src, tgt in feedback:
        print(f"  {src} --> {tgt}")

    # Show per-cycle structure
    print("\nNodes per cycle:")
    for k in range(1, 4):
        nodes = creator.get_nodes_in_cycle(k)
        print(f"  Cycle {k}: {len(nodes)} nodes")

    # Show outcome chain
    print("\nOutcome chain (match performances):")
    for k in range(1, 4):
        outcome = creator.get_cycle_outcome(k)
        parents = creator.get_parents(outcome)
        children = creator.get_children(outcome)
        print(f"  {outcome}: {len(parents)} parents, {len(children)} children")

    # Verify the feedback connects outcomes to next baseline
    print("\nCycle 1 outcome -> Cycle 2 baseline:")
    outcome_c1 = creator.get_cycle_outcome(1)
    for child in creator.get_children(outcome_c1):
        print(f"  {outcome_c1} --> {child}")

    print(f"\nIs acyclic: {nx.is_directed_acyclic_graph(dag2)}")
    print(f"Topological generations: {len(list(nx.topological_generations(dag2)))}")

    # -----------------------------------------------------------------
    # Demo 3: VISUALIZATIONS
    # -----------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("DEMO 3: DAG Visualizations")
    print("=" * 65)

    # 3a. Schematic view of a single cycle
    print("\n[3a] Schematic view — single cycle (5 days)...")
    creator.build_dag(
        cycle_lengths=5,
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )
    creator.visualize(
        mode='schematic',
        save_path=str(results_dir / "dag_schematic_single_cycle.png"),
        show=False,
    )
    print(f"  Saved to: {results_dir / 'dag_schematic_single_cycle.png'}")

    # 3b. Schematic view of multi-cycle (shows feedback edges)
    print("\n[3b] Schematic view — multi-cycle (3 cycles: 5, 7, 5)...")
    creator.build_dag(
        cycle_lengths=[5, 7, 5],
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )
    creator.visualize(
        mode='schematic',
        save_path=str(results_dir / "dag_schematic_multi_cycle.png"),
        show=False,
    )
    print(f"  Saved to: {results_dir / 'dag_schematic_multi_cycle.png'}")

    # 3c. Schematic view of just one cycle from a multi-cycle DAG
    print("\n[3c] Schematic view — cycle 2 only (from the 3-cycle DAG)...")
    creator.visualize(
        mode='schematic',
        cycle=2,
        save_path=str(results_dir / "dag_schematic_cycle_2_only.png"),
        show=False,
    )
    print(f"  Saved to: {results_dir / 'dag_schematic_cycle_2_only.png'}")

    # 3d. Detailed view of a single cycle (shows all individual variables)
    print("\n[3d] Detailed view — single cycle (5 days, all nodes)...")
    creator.build_dag(
        cycle_lengths=5,
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )
    creator.visualize(
        mode='detailed',
        save_path=str(results_dir / "dag_detailed_single_cycle.png"),
        show=False,
    )
    print(f"  Saved to: {results_dir / 'dag_detailed_single_cycle.png'}")

    # 3e. Single cycle convenience method (visualize_cycle)
    print("\n[3e] visualize_cycle() — cycle 1 schematic shorthand...")
    creator.build_dag(
        cycle_lengths=[4, 6],
        state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
        daily_treatment_var='Activity Type Today',
        outcome_var='Status Decrease',
    )
    creator.visualize_cycle(
        cycle=1,
        save_path=str(results_dir / "dag_cycle_1_shorthand.png"),
        show=False,
    )
    print(f"  Saved to: {results_dir / 'dag_cycle_1_shorthand.png'}")

    print("\n" + "=" * 65)
    print("All visualizations saved to:", results_dir)
    print("=" * 65)
