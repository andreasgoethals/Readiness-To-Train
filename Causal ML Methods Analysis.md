# Causal ML Methods Analysis — Readiness to Train

**Project:** Causal Modelling of Player Readiness to Train  
**Context:** KU Leuven & OH Leuven Partnership  
**Date:** February 2026

---

## 1. Problem Anatomy: Why This Is Unusually Hard

Before recommending methods, it's worth making the structural challenges explicit, because they jointly eliminate many off-the-shelf solutions.

**Your data-generating process has five interlocking features:**

| Challenge | What it means concretely | What it rules out |
|-----------|--------------------------|-------------------|
| **Time-varying confounding affected by prior treatment** | Fatigue at *t* is caused by training at *t−1* and causes training at *t*. ACWR, wellness z-scores, and Status are all post-treatment confounders. | Standard regression adjustment (blocks causal pathways), naive propensity score matching on time-varying covariates |
| **Sequential treatment** | Training is not a single exposure — it's a sequence of ~6 daily sessions per microcycle. The causal question is about *sequences*, not single days. | Cross-sectional CATE estimators (S/T/X-learner) applied naively to single time points |
| **Small N, Large T** | 27 players × 156 days = 4,239 obs, but only 27 independent clusters. Between-player heterogeneity is thin; within-player dynamics dominate. | High-capacity deep models without partial pooling, methods requiring large cross-sectional samples |
| **Informative censoring / selective observation** | Match-day outcomes only observed for selected, available players. Injury and non-selection are correlated with treatment history. | Complete-case analysis, methods ignoring the censoring mechanism |
| **Endogenous treatment assignment** | Coaches assign load based on the player's current state — fresh players train hard, fatigued players rest. This creates systematic confounding by indication. | Unadjusted outcome comparisons across treatment groups |

The within-row temporal structure is also critical. On any day *t*, the data contains three temporal layers: **yesterday's load** (fully observed), **morning assessment** (wellness, status — measured before activity decisions), and **Activity Type Today** (the treatment, decided *after* the morning assessment). This means morning wellness is a pre-treatment covariate for today's treatment, but a post-treatment outcome of yesterday's treatment. Any method you use must respect this ordering.

---

## 2. How to Operationalise the Causal Question

Before choosing a method, you need to lock down the treatment and outcome definitions. Different operationalisations lead to very different methods. Here are the most defensible framings for your data:

### Framing A: Single-Step Treatment Effect (Simplest Starting Point)

- **Treatment (T):** Activity Type Today, discretised as {Rest/Recovery, Low, Moderate, High}; or a continuous GPS intensity metric (e.g., Total Distance % Yesterday pulled from t+1)
- **Outcome (Y):** Status Decrease at t+1 (binary: did the player's medical status worsen the next day?)
- **Covariates (X):** Morning wellness z-scores, ACWR values, Days Since Game, Days Until Match, Position, Medical Availability, yesterday's GPS %, Comment Category
- **Confounders to worry about:** Current Status, wellness composite scores, ACWR values — all of which are simultaneously outcomes of prior treatment and causes of current treatment assignment

This framing asks: *"Given everything we observe about a player this morning, what is the causal effect of assigning training intensity level* a *vs.* a′ *on the probability of status worsening tomorrow?"*

### Framing B: Sequential / Dynamic Treatment Regime

- **Treatment sequence:** The vector of daily training intensities over a microcycle (e.g., 5–7 days between matches)
- **Outcome:** Match-day readiness, or cumulative Status Decrease events over the microcycle
- **Time-varying confounders:** Wellness z-scores, ACWR, Status — all updated daily

This framing asks: *"What sequence of training intensities over the week maximises the probability of being Available on match day?"*

### Framing C: Time-to-Event / Survival

- **Event:** Status Decrease (or transition to Injured)
- **Time:** Days since last match / start of microcycle until the status-worsening event
- **Treatment:** Daily training intensity, potentially time-varying
- **Censoring:** Players who reach match day without status decrease are right-censored; players who leave the observation window (sick, absent) are informatively censored

---

## 3. Recommended Methods — Staged by Complexity

I'm ordering these from "start here" to "ambitious PhD contribution," reflecting a natural progression where each stage builds understanding for the next.

---

### Stage 1: Establish the Causal Graph and Baseline Effects

#### 3.1 DoWhy + DAG Specification

**What it does:** Forces you to formalise the causal assumptions before estimating anything. You specify the DAG, DoWhy identifies the estimand (e.g., via the backdoor criterion), estimates it, and runs refutation tests.

**Why it fits your project:** You have strong domain knowledge about which variables are confounders, mediators, and colliders. The temporal structure within each row gives you a defensible ordering. This is the right first step before any fancy estimation.

**Concrete setup for your data:**

```python
import dowhy
from dowhy import CausalModel

# DAG edges based on your temporal structure
graph = """
digraph {
    // Yesterday's load → morning state
    ActivityTypeYesterday -> Fatigue_z;
    ActivityTypeYesterday -> Readiness_z;
    ActivityTypeYesterday -> Soreness_z;
    ActivityTypeYesterday -> StatusDecrease;
    GPS_pct_Yesterday -> Fatigue_z;
    GPS_pct_Yesterday -> ACWR;
    
    // Morning state → today's treatment (coach's decision)
    Fatigue_z -> ActivityTypeToday;
    Readiness_z -> ActivityTypeToday;
    Status -> ActivityTypeToday;
    ACWR -> ActivityTypeToday;
    DaysUntilMatch -> ActivityTypeToday;
    
    // Today's treatment → tomorrow's outcome
    ActivityTypeToday -> StatusDecrease_tomorrow;
    
    // Confounders affecting both treatment and outcome
    Fatigue_z -> StatusDecrease_tomorrow;
    ACWR -> StatusDecrease_tomorrow;
    Position -> StatusDecrease_tomorrow;
    MedicalAvailability -> StatusDecrease_tomorrow;
    MedicalAvailability -> ActivityTypeToday;
}
"""
```

**Refutation tests to run:** Random common cause, placebo treatment, data subset validation, sensitivity to unobserved confounders (E-value or Rosenbaum bounds).

**Python libraries:** `dowhy`, with `econml` or `causalml` as estimation backends.

**Limitation:** DoWhy handles the static (single time-point) case well. It does not natively handle time-varying confounding. But it disciplines your thinking, and the refutation framework is invaluable.

---

#### 3.2 Causal Meta-Learners (S/T/X/DR-Learner) for Heterogeneous Treatment Effects

**What they do:** Estimate the Conditional Average Treatment Effect (CATE) — the treatment effect as a function of covariates. This tells you *which players* respond differently to high vs. low training load.

**Why they fit (with caveats):** Your ultimate goal is a personalised policy. Meta-learners give you CATE(x), which you can directly translate to a traffic light: if CATE(x) for high intensity is strongly negative (increases status-decrease risk for this player profile), → Red. If neutral or positive, → Green. The X-Learner is particularly suited to your setting because it handles unequal group sizes well (you likely have far more "Training" days than "Rest" days), and the DR-Learner provides doubly-robust estimation combining propensity scores with outcome modeling.

**Important caveat:** These methods assume *static* treatment — they estimate the effect of a single-day treatment decision conditional on current covariates. They do NOT handle the sequential nature of the problem. However, they are a perfectly valid starting point for your Framing A, and the heterogeneity patterns they reveal (which player subgroups respond worst to high load?) are valuable input for later sequential methods.

**Concrete setup:**

```python
from econml.dml import CausalForestDML
from econml.metalearners import XLearner, SLearner, TLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# X-Learner with propensity scoring
xl = XLearner(
    models=GradientBoostingClassifier(n_estimators=100, max_depth=4),
    propensity_model=GradientBoostingClassifier(n_estimators=100),
    cate_models=GradientBoostingRegressor(n_estimators=100)
)
xl.fit(Y=y_train, T=t_train, X=x_train)
cate_estimates = xl.effect(x_test)

# Causal Forest (via EconML's DML variant)
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingClassifier(),
    n_estimators=500,
    min_samples_leaf=10,  # conservative given small N
    random_state=42
)
cf.fit(Y=y_train, T=t_train, X=x_train, W=w_train)  # W = pure confounders
```

**Key design choices for your data:**
- **Treatment discretisation:** Bin Activity Type Today into 2–3 levels (not too many given N=27). Or use continuous treatment via DML.
- **Propensity model:** This is critical. The coach's assignment policy IS the propensity score. A good propensity model tells you: "given this player's morning state, how likely were they to receive high-intensity training?" Model it carefully with the morning covariates.
- **Cluster-robust inference:** Your 27 players are the independent units, not the 4,239 observations. Use clustered standard errors or bootstrap at the player level.

**Python libraries:** `econml` (Microsoft), `causalml` (Uber), `grf` (via `rpy2` to R's `grf` package for Causal Forest with cluster-robust inference).

---

### Stage 2: Handle the Time-Varying Confounding

This is where you go beyond standard CATE estimation and address the core causal challenge of your project.

#### 3.3 Marginal Structural Models (MSMs) with Inverse Probability of Treatment Weighting (IPTW)

**What they do:** MSMs solve the time-varying confounding problem by reweighting observations so that — in the pseudo-population created by the weights — treatment is independent of confounders at each time point. Instead of conditioning on time-varying confounders (which blocks causal pathways), you weight by the inverse of the probability of receiving the observed treatment given the full history.

**Why this is central to your project:** Your fundamental problem is that fatigue at time *t* is both (a) a consequence of training at *t−1* (so it's on the causal pathway) and (b) a cause of treatment at *t* (so it's a confounder). Classical regression that adjusts for fatigue blocks the indirect effect of *t−1* training working through fatigue. MSMs avoid this by weighting rather than conditioning.

**How it works for your data:**

At each time point *t*, for each player *i*, compute stabilised weights:

$$SW_{i,t} = \prod_{k=1}^{t} \frac{P(A_{i,k} | \bar{A}_{i,k-1}, V_i)}{P(A_{i,k} | \bar{A}_{i,k-1}, \bar{L}_{i,k}, V_i)}$$

where $A_{i,k}$ is treatment at time $k$, $\bar{A}_{i,k-1}$ is treatment history, $\bar{L}_{i,k}$ is the time-varying confounder history (wellness, ACWR, status), and $V_i$ is baseline covariates (position, etc.).

```python
# Step 1: Estimate the treatment model (denominator of weights)
# P(A_t | treatment history, full covariate history)
from sklearn.linear_model import LogisticRegression
import numpy as np

# For each time step, fit propensity model
# Denominator: P(A_t | past treatments, past + current covariates)
denom_model = LogisticRegression(max_iter=1000)
denom_model.fit(X_full_history, A_t)

# Numerator (stabilised): P(A_t | past treatments, baseline covariates only)
numer_model = LogisticRegression(max_iter=1000)
numer_model.fit(X_baseline_and_past_treatment, A_t)

# Compute cumulative weights
sw = np.cumprod(numer_probs / denom_probs, axis=1)

# Step 2: Fit weighted outcome model (MSM)
# Weighted GEE or weighted pooled logistic regression
import statsmodels.api as sm
msm = sm.GEE(
    y, X_treatment_history, 
    groups=player_id,
    weights=sw,
    family=sm.families.Binomial()
)
```

**Practical concerns for your data:**
- **Weight instability:** With 27 players and multi-level treatment, some treatment histories will be rare, creating extreme weights. Use stabilised weights and truncate at the 1st/99th percentile.
- **Positivity assumption:** Every player must have non-zero probability of receiving every treatment level at every time point. This is plausible for training types but may fail for extreme scenarios (e.g., injured player receiving high-intensity training). Check the positivity diagnostic carefully.
- **Model specification:** Use flexible models (GBMs, not just logistic regression) for the propensity scores, but beware of overfitting given the small N.

**Python libraries:** `zepid` (has MSM/IPTW utilities), `causallib` (IBM), or manual implementation with `statsmodels.GEE`.

---

#### 3.4 G-Computation / G-Formula (Parametric Sequential Modelling)

**What it does:** Rather than reweighting, g-computation directly models the outcome under hypothetical treatment interventions by simulating forward through the data-generating process. You fit models for (a) each time-varying confounder as a function of history and (b) the outcome as a function of treatment and confounder history, then simulate counterfactual trajectories under different treatment strategies.

**Why it fits:** G-computation is particularly natural for your weekly microcycle framing. You can simulate: "What would this player's status trajectory look like if we assigned High-Moderate-Low-Recovery-Moderate-Game vs. Moderate-Moderate-Moderate-Recovery-Moderate-Game?" It's the parametric counterpart to the MSM approach and can be more efficient when the outcome model is well-specified.

**Algorithm sketch for your data:**

```
For each counterfactual treatment strategy a* = (a*_1, ..., a*_T):
    For each Monte Carlo iteration:
        1. Draw baseline covariates from the empirical distribution
        2. For t = 1 to T (days in microcycle):
           a. Set A_t = a*_t (intervene)
           b. Simulate L_t (wellness, ACWR) from fitted model:
              L_t ~ P(L_t | A_t, L_{t-1}, A_{t-1}, ...)
           c. Simulate Y_t (status decrease) from fitted model:
              Y_t ~ P(Y_t | A_t, L_t, history)
    3. Average Y over iterations → E[Y | do(a*)]
```

```python
# Fit confounder evolution models
from sklearn.ensemble import GradientBoostingRegressor

fatigue_model = GradientBoostingRegressor()
fatigue_model.fit(
    X=[treatment_history, confounder_history, baseline],
    y=fatigue_next
)

# Fit outcome model
outcome_model = GradientBoostingClassifier()
outcome_model.fit(
    X=[treatment_history, confounder_history],
    y=status_decrease
)

# Simulate counterfactual under strategy "moderate all week"
def g_compute(strategy, n_sims=1000):
    outcomes = []
    for _ in range(n_sims):
        # Bootstrap a player's baseline
        baseline = sample_baseline()
        L = baseline.copy()
        for t, a_t in enumerate(strategy):
            L = fatigue_model.predict(L, a_t, history)  # evolve covariates
            y_t = outcome_model.predict_proba(L, a_t, history)
        outcomes.append(y_t)
    return np.mean(outcomes)
```

**Advantage over MSMs:** No weight instability. Can naturally handle continuous treatment. The simulation approach is very intuitive for communicating results to coaching staff.

**Disadvantage:** Heavily relies on correct specification of the confounder models. If your model for "how does fatigue evolve given treatment and history?" is wrong, the counterfactual simulations are biased. With only 27 players, model misspecification is a real risk.

**Python libraries:** `zepid.causal.gformula`, or manual implementation. The `ltmle` package in R is the gold standard for targeted variants.

---

### Stage 3: Optimal Policy Learning (The PhD Contribution)

This is where the project moves from "estimate effects" to "recommend actions."

#### 3.5 Dynamic Treatment Regimes (DTR) via G-Estimation or Q-Learning

**What they do:** DTRs estimate the *optimal* treatment rule — a mapping from patient/player state to the best treatment at each time point. The two main estimation strategies are:

- **Q-learning (backward induction):** Start from the last time point, estimate the optimal treatment, then work backward. At each step, the "outcome" includes the expected future reward under optimal behaviour.
- **G-estimation (structural nested models):** Directly model the "blip function" — the additional benefit of treatment *a* vs. reference treatment *a₀* at time *t*, given that optimal treatment is followed from *t+1* onward.

**Why this is the natural endgame for your project:** Your traffic-light system IS a treatment rule. DTR methods directly produce what you need: "Given this player's state on Monday, what training intensity maximises P(Available on Saturday)?"

**Q-learning for your microcycle:**

```python
# Backward induction over the training week
# Day T (match day - 1): 
#   Q_T(H_T, A_T) = E[Y | H_T, A_T]  (just outcome regression)
# Day T-1: 
#   Q_{T-1}(H_{T-1}, A_{T-1}) = E[max_{A_T} Q_T(H_T, A_T) | H_{T-1}, A_{T-1}]

from sklearn.ensemble import GradientBoostingRegressor

# Start from last decision point
Q_models = {}
for t in reversed(range(T)):
    if t == T - 1:
        # Terminal Q-function: just predict outcome
        Q_models[t] = GradientBoostingRegressor()
        Q_models[t].fit(X=[history_t, treatment_t], y=outcome)
    else:
        # Compute pseudo-outcome: max over treatments of Q_{t+1}
        pseudo_y = []
        for treatment_level in treatment_levels:
            q_next = Q_models[t+1].predict(X=[history_t_plus_1, treatment_level])
            pseudo_y.append(q_next)
        optimal_future = np.max(pseudo_y, axis=0)
        
        Q_models[t] = GradientBoostingRegressor()
        Q_models[t].fit(X=[history_t, treatment_t], y=optimal_future)

# Extract optimal policy
def optimal_policy(player_state, day_in_week):
    best_a = None
    best_q = -np.inf
    for a in treatment_levels:
        q = Q_models[day_in_week].predict(player_state, a)
        if q > best_q:
            best_q, best_a = q, a
    return best_a
```

**Critical issue — confounding:** Standard Q-learning assumes no unmeasured confounding. With observational data, you need to modify the approach. Options include:
- Use IPTW-augmented Q-learning (doubly robust)
- Use the A-learning variant (which only models the treatment effect contrast, not the full outcome)
- Combine with propensity score weighting to deconfound

**Python libraries:** `DTRlearn2` (R, via `rpy2`), manual implementation in Python. The `d3rlpy` library supports offline RL which is closely related.

---

#### 3.6 Off-Policy Reinforcement Learning (Batch RL from Observational Data)

**What it does:** Treats the entire problem as a Markov Decision Process and learns a treatment policy from the logged (observational) data. The key challenge is *off-policy evaluation* — estimating how well a new policy would perform using data collected under the coach's historical policy.

**Why it's relevant but risky:** This is the most general framing. Your data is literally a logged MDP: state = (player wellness, ACWR, days since game, ...), action = training intensity, reward = maintaining availability while building fitness. RL methods can in principle discover complex multi-step strategies that simpler methods miss.

**Why it's risky for your setting:**
- N=27 is extremely small for RL. The state-action space is large relative to the data.
- Off-policy evaluation is notoriously unreliable with limited data — the importance weights can have enormous variance.
- RL methods are harder to interpret and validate, which matters for coaching staff adoption.

**Practical recommendation:** Use off-policy evaluation methods (not full policy optimisation) as a *validation tool* for policies learned by simpler methods (DTR, g-computation). Specifically, Doubly Robust off-policy estimators can tell you "if we had followed policy π instead of the coach's policy, the expected status-decrease rate would have been X."

```python
# Off-policy evaluation via Doubly Robust estimator
def dr_off_policy_eval(new_policy, behaviour_probs, outcomes, Q_hat):
    """
    new_policy: P(a|s) under the proposed policy
    behaviour_probs: P(a|s) under the coach's observed policy
    outcomes: observed rewards
    Q_hat: estimated Q-function
    """
    rho = new_policy / behaviour_probs  # importance ratio
    dr_estimate = np.mean(
        Q_hat + rho * (outcomes - Q_hat)
    )
    return dr_estimate
```

**Python libraries:** `d3rlpy`, `OPE` libraries, or custom implementation.

---

#### 3.7 SurvITE (Survival-Based Heterogeneous Treatment Effects)

**What it does:** SurvITE (Curth et al., 2021 — already in your project references) estimates treatment-specific hazard functions using balanced representations. It handles three types of covariate shift simultaneously: selection bias (confounded treatment), informative censoring, and event-induced shift.

**Why it fits your Framing C:** If you reframe the outcome as time-to-status-decrease (a survival outcome), SurvITE directly estimates: "Given this player's profile, what is the survival curve (probability of remaining Available) under treatment A vs. treatment B over the next 7 days?" The difference in survival curves IS your treatment effect.

**Mapping to your data:**
- **Event time T:** Days from start of microcycle until Status Decrease = 1
- **Censoring C:** Players who reach match day without status decrease, or who leave observation (sick, absent)
- **Treatment A:** Training intensity (binary or multi-level)
- **Covariates X:** Morning wellness, ACWR, position, days since game, etc.

**Why it addresses your key challenges:**
- The IPM (Integral Probability Metric) regularisation in SurvITE explicitly handles confounding by balancing representations across treatment groups.
- It models the time-to-event nature directly, so you get survival curves, not just point estimates.
- The discrete-time hazard formulation handles the daily granularity of your data naturally.

**Architecture (from the paper):** Shared representation Φ(x), with treatment-specific hazard heads h_{a,τ}, trained with a loss combining risk (cross-entropy for each time step) and IPM balance toward the baseline population.

**Limitation for your setting:** SurvITE handles static treatment — one treatment assigned at baseline affecting a subsequent survival curve. It does NOT handle time-varying treatment. So it fits Framing C when treatment is defined as the training type at the start of a microcycle, but not for day-by-day dynamic treatment.

**Implementation:** Available at `github.com/chl8856/survITE`.

---

### Stage 4: Methods That Address Multiple Challenges Simultaneously

#### 3.8 Longitudinal Targeted Maximum Likelihood Estimation (LTMLE)

**What it does:** LTMLE is the most principled method for longitudinal causal inference with time-varying treatments and confounders. It combines the virtues of g-computation (outcome modelling) and IPTW (propensity weighting) into a doubly robust estimator that is consistent if *either* the outcome model *or* the treatment model is correctly specified. Additionally, it uses targeted regularisation (TMLE) to optimise the bias-variance tradeoff specifically for the causal parameter of interest, rather than for prediction accuracy.

**Why this is arguably the best-suited method for your full problem:**
- Handles time-varying confounding affected by prior treatment ✓
- Doubly robust (protects against model misspecification) ✓
- Provides valid inference (confidence intervals, p-values) for the causal effect ✓
- Can estimate both static regimes ("always moderate") and dynamic regimes ("train hard if ACWR < 1.2, rest otherwise") ✓

**The challenge:** LTMLE is primarily implemented in R (`ltmle` package by Lendle, Schwab, van der Laan). The Python ecosystem is less mature, though `zepid` has some TMLE functionality and `tmle3` is under development.

```r
# R implementation (consider using via rpy2)
library(ltmle)

result <- ltmle(
  data = player_data,
  Anodes = c("Activity_Day1", "Activity_Day2", ..., "Activity_Day5"),
  Lnodes = c("Fatigue_Day1", "ACWR_Day1", ..., "Fatigue_Day5", "ACWR_Day5"),
  Ynodes = "StatusDecrease_MatchDay",
  abar = list(
    treatment = c("High", "Moderate", "Low", "Recovery", "Moderate"),
    control   = c("Moderate", "Moderate", "Moderate", "Recovery", "Moderate")
  ),
  SL.library = c("SL.glm", "SL.randomForest", "SL.xgboost"),
  variance.method = "ic"  # influence-curve based variance
)
```

**Python alternative:** Use `zepid.causal.doublyrobust` for point-in-time TMLE, or invest in wrapping the R `ltmle` package via `rpy2`.

---

#### 3.9 Causal Forests for Panel Data (with Fixed Effects)

**What it does:** The standard Causal Forest (Wager & Athey, 2018) is designed for cross-sectional data. Recent extensions handle panel data by incorporating player fixed effects or within-player estimation — subtracting player-level means before estimating heterogeneous effects.

**Why it fits your small-N-large-T setting:** By leveraging within-player variation, you're effectively using each player as their own control. A player who trained hard on Monday and rested on Wednesday provides within-player variation that identifies the treatment effect without needing to compare across players. This dramatically increases your effective sample size from 27 to ~4,000.

**Implementation approach:**

```python
from econml.dml import CausalForestDML

# Option 1: Include player fixed effects as covariates
# (player dummies or player embeddings)
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingClassifier(),
    n_estimators=1000,
    min_samples_leaf=20,  # regularise heavily
)
# Add player ID dummies to W (instruments/controls)
cf.fit(Y=y, T=t, X=x_effect_modifiers, W=np.hstack([x_confounders, player_dummies]))

# Option 2: Within-player demeaning (panel fixed effects)
# Subtract player means before fitting
for col in continuous_features:
    df[col] = df.groupby('Player ID')[col].transform(lambda x: x - x.mean())
```

**Key advantage:** You get CATE estimates — treatment effect heterogeneity — which directly feeds the traffic-light system. You can identify subgroups: "Central defenders with high ACWR and low readiness respond worst to high-intensity training."

---

## 4. Recommended Progression for the PhD

Based on the data structure, sample size, and research objectives, here is a concrete staging:

### Phase 1: Foundation (Months 1–2)

**Goal:** Establish the causal graph, validate assumptions, get baseline CATE estimates.

1. **Specify the DAG in DoWhy.** Use domain knowledge to encode the causal structure. Run identification and refutation checks.
2. **Estimate propensity scores.** Model P(high-intensity | morning state). This is valuable in itself — it characterises the coach's decision policy.
3. **Fit meta-learners (X-Learner, DR-Learner via EconML).** Get initial CATE estimates for the single-step treatment effect. Cluster-bootstrap confidence intervals.
4. **Fit a Causal Forest (CausalForestDML).** Compare CATE estimates. Check heterogeneity: which covariates drive differential treatment effects?

### Phase 2: Sequential Methods (Months 3–5)

**Goal:** Address time-varying confounding and estimate effects of treatment sequences.

5. **Implement MSMs with IPTW.** Build the sequential propensity model. Estimate the effect of "always high" vs. "always moderate" training strategies.
6. **Implement G-computation.** Simulate counterfactual microcycle trajectories. Compare with MSM results.
7. **Fit Q-learning DTR.** Estimate the optimal treatment rule at each day of the microcycle. Validate with off-policy evaluation.

### Phase 3: Novel Contribution (Months 5–8)

**Goal:** Combine survival framing with sequential treatment, or develop a domain-adapted method.

8. **Adapt SurvITE to your microcycle framing.** Define the "event" as status-decrease within the microcycle. Estimate treatment-specific survival curves.
9. **Implement LTMLE for the full sequential problem.** This is the methodologically strongest option. Use R's `ltmle` via `rpy2` if needed.
10. **Off-policy evaluation of the learned policy.** Compare your recommended policy against the coach's historical policy. Quantify expected improvement.

### Phase 4: Practical System (Months 8–10)

**Goal:** Build the traffic-light tool.

11. **Translate CATE/DTR output to Green/Orange/Red.** Map the estimated treatment effects or optimal policy to actionable categories based on clinically meaningful thresholds.
12. **Develop the decision-support interface.** Daily input: morning assessment data. Output: recommended training intensity per player.

---

## 5. Method Comparison Matrix

| Method | Handles time-varying confounding | Handles sequential treatment | Heterogeneous effects | Small-N safe | Python implementation | Interpretability |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| DoWhy + DAG | — | — | — | ✓ | `dowhy` | Very high |
| Meta-learners (X/DR) | ✗ | ✗ | ✓ | ○ | `econml`, `causalml` | High |
| Causal Forest | ✗ | ✗ | ✓ | ○ | `econml` | High |
| MSM / IPTW | ✓ | ✓ | ✗ | ○ | `zepid`, custom | Medium |
| G-computation | ✓ | ✓ | ○ | ○ | `zepid`, custom | Medium |
| Q-learning DTR | ✓ | ✓ | ✓ | ✗ | Custom, `d3rlpy` | Medium |
| SurvITE | ○ | ✗ | ✓ | ○ | `survITE` (GitHub) | Medium |
| LTMLE | ✓ | ✓ | ○ | ○ | R's `ltmle` via `rpy2` | Low-Medium |
| Off-policy RL | ✓ | ✓ | ✓ | ✗ | `d3rlpy` | Low |

*✓ = handles well, ○ = partially, ✗ = does not handle, — = not applicable*

---

## 6. Critical Assumptions to Check

No matter which method you choose, these assumptions must be checked or acknowledged:

1. **No unmeasured confounding (sequential ignorability):** Conditional on observed history, treatment is independent of potential outcomes. This is the strongest and least testable assumption. Your rich morning assessment data (6 wellness dimensions + ACWR + status + medical availability) makes it more plausible than in most observational studies, but it can never be verified. Use sensitivity analysis (E-values, Rosenbaum bounds) extensively.

2. **Positivity:** Every player must have non-zero probability of receiving every treatment level at every time point, conditional on their history. Check this empirically: are there covariate regions where treatment is deterministic? (e.g., injured players ALWAYS rest — this is a positivity violation).

3. **Consistency:** A player's potential outcome under treatment *a* is the same regardless of how *a* was assigned. This is generally plausible for training intensity.

4. **Correct model specification (for g-computation, Q-learning):** The parametric models for outcome and confounder evolution must be correctly specified. Use cross-validation, but remember that predictive accuracy ≠ causal accuracy.

5. **Stable unit treatment value (SUTVA):** One player's treatment doesn't affect another player's outcome. This may be violated in team sports (e.g., tactical training requires specific player combinations). A reasonable starting assumption, but worth acknowledging.

---

## 7. Python Ecosystem Summary

| Library | What it provides | Install |
|---------|-----------------|---------|
| `dowhy` | Causal graph specification, identification, refutation | `pip install dowhy` |
| `econml` | Meta-learners, Causal Forest, DML, Double Robust | `pip install econml` |
| `causalml` | Meta-learners, uplift modelling | `pip install causalml` |
| `zepid` | IPTW, MSM, G-formula, TMLE, diagnostics | `pip install zepid` |
| `causallib` | IPW, standardisation, doubly robust (IBM) | `pip install causallib` |
| `d3rlpy` | Offline RL, batch policy learning | `pip install d3rlpy` |
| `lifelines` | Survival analysis (Cox PH, KM, AFT) | `pip install lifelines` |
| `scikit-survival` | Survival forests, survival SVM | `pip install scikit-survival` |
| R `ltmle` | Longitudinal TMLE (via `rpy2`) | R package |
| R `grf` | Causal Forest with cluster-robust inference | R package |
