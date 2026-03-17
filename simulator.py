import argparse
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Backtest foundation (historical calibration)
# ---------------------------------------------------------------------------
# Intended flow once historical data is available:
#   config = BacktestConfig(season="2024", bracket=load_historical_bracket(2024), ...)
#   predicted = run_backtest(config, num_simulations=100_000, rng=rng)
#   actual = load_historical_results(2024)
#   metrics = evaluate_backtest(predicted, actual, num_simulations)
#   # Compare: champion_correct, final_four_hit_rate, upset calibration
#
# TODO: Add load_historical_bracket(year) -> dict
# TODO: Add load_historical_ratings(year) -> dict[str, float]
# TODO: Add load_historical_results(year) -> HistoricalResults


@dataclass
class BacktestConfig:
    """
    Configuration for running the simulator on a specific season.

    Swap in historical bracket and ratings to backtest model calibration.
    """
    season: str  # e.g. "2024", "2025"
    bracket: dict[str, list[tuple[str, int]]]
    overall_overrides: dict[str, float]
    metrics_overrides: dict[str, dict[str, Any]] | None = None

    # TODO: Load from CSV/JSON, e.g.:
    #   config = BacktestConfig(
    #       season="2024",
    #       bracket=load_historical_bracket(2024),
    #       overall_overrides=load_historical_ratings(2024),
    #       metrics_overrides=load_historical_metrics(2024),
    #   )


@dataclass
class HistoricalResults:
    """
    Actual tournament outcomes for a season. Used to evaluate predictions.

    TODO: Populate from historical data (Wikipedia, NCAA, etc.).
    """

    season: str
    champion: str
    final_four: list[str]
    # TODO: Add elite_eight, sweet_sixteen, round_of_32 for finer evaluation
    # TODO: Add first_round_upsets: list[tuple[str, str]]  # (winner, loser) for upset calibration


def run_backtest(
    config: BacktestConfig,
    num_simulations: int = 100_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run the simulator on a backtest config (historical bracket + ratings).

    Returns the same structure as simulate_tournament for evaluation.
    """
    rng = rng or np.random.default_rng()
    registry = build_team_metrics_registry(
        config.bracket,
        overall_overrides=config.overall_overrides,
        metrics_overrides=config.metrics_overrides,
    )
    local_team_stats = {name: m.to_win_factors() for name, m in registry.items()}

    return simulate_tournament(
        config.bracket,
        num_simulations=num_simulations,
        rng=rng,
        progress_every=0,
        _team_stats=local_team_stats,
        _bracket=config.bracket,
    )


def evaluate_backtest(
    predicted: dict,
    actual: HistoricalResults,
    num_simulations: int,
) -> dict[str, Any]:
    """
    Compare predicted results to actual outcomes.

    Returns metrics for:
    - champion_pick_quality: did top-predicted champion match actual?
    - final_four_hit_rate: how many actual Final Four teams were in top-4 predicted?
    - upset_calibration: TODO - compare predicted upset probs to actual upset rates by seed
    """
    # Predicted champion = most frequent in championship_wins
    pred_champion = max(predicted["championship_wins"].items(), key=lambda x: x[1])[0]
    champion_correct = pred_champion == actual.champion

    # Predicted Final Four = top 4 by final_four count
    pred_ff = sorted(
        predicted["final_four"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:4]
    pred_ff_teams = {t for t, _ in pred_ff}
    actual_ff_set = set(actual.final_four)
    final_four_hits = len(pred_ff_teams & actual_ff_set)
    final_four_hit_rate = final_four_hits / 4.0 if len(actual_ff_set) == 4 else 0.0

    # TODO: Upset calibration by seed
    #   - For each seed matchup (1v16, 2v15, ...), compare predicted higher-seed win %
    #     to actual historical win rate. Requires first_round_upsets in HistoricalResults.

    return {
        "champion_correct": champion_correct,
        "predicted_champion": pred_champion,
        "actual_champion": actual.champion,
        "final_four_hits": final_four_hits,
        "final_four_hit_rate": final_four_hit_rate,
        "predicted_final_four": list(pred_ff_teams),
        "actual_final_four": actual.final_four,
    }


# ---------------------------------------------------------------------------
# Bracket pool expected value (optional)
# ---------------------------------------------------------------------------
# Optimizes for pool scoring, not raw prediction accuracy.
# Extensible: custom round scoring, contrarian pick value, chalk vs variance.


@dataclass
class BracketPoolConfig:
    """
    Scoring rules for a bracket pool. Points per correct pick by round.

    Default mimics common pools (1-2-4-8-16-32). Override for custom formats.
    """
    round_64: int = 1
    round_32: int = 2
    sweet_16: int = 4
    elite_8: int = 8
    final_four: int = 16
    championship: int = 32

    def points_for_round(self, round_key: str) -> int:
        """Round key: 'round_32'|'sweet_sixteen'|'elite_eight'|'final_four'|'championship_wins'."""
        mapping = {
            "round_32": self.round_64,  # R64 winners advance to R32
            "sweet_sixteen": self.round_32,
            "elite_eight": self.sweet_16,
            "final_four": self.elite_8,
            "championship_wins": self.championship,
        }
        return mapping.get(round_key, 0)


def _build_first_round_matchups(bracket: dict) -> list[tuple[str, str]]:
    """
    Build the 32 first-round matchups from the bracket.
    Returns [(team_a, team_b), ...] for each game.
    """
    games = []
    for region, teams in bracket.items():
        matchups = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        for i, j in matchups:
            t1, t2 = teams[i][0], teams[j][0]
            games.append((t1, t2))
    return games


def expected_bracket_score_r64(
    simulation_results: dict,
    num_simulations: int,
    bracket: dict,
    pool_config: BracketPoolConfig,
    picks: dict[tuple[str, str], str] | None = None,
) -> float:
    """
    Expected pool score for first-round (R64) picks only.

    picks: optional {(team_a, team_b): winner} for each game. When None, uses chalk
    (most likely winner from round_32 advancement counts).

    Extensible: add expected_bracket_score_full() for R32–Champ once bracket
    resolution is implemented.
    """
    n = num_simulations
    r32 = simulation_results["round_32"]
    games = _build_first_round_matchups(bracket)
    ev = 0.0

    for t1, t2 in games:
        c1, c2 = r32.get(t1, 0), r32.get(t2, 0)
        p1 = c1 / n if n else 0
        p2 = c2 / n if n else 0

        if picks is not None and (t1, t2) in picks:
            winner = picks[(t1, t2)]
            prob = p1 if winner == t1 else p2
        else:
            # Chalk: pick higher probability
            prob = max(p1, p2)

        ev += prob * pool_config.round_64

    return ev


def compare_chalk_vs_contrarian(
    simulation_results: dict,
    num_simulations: int,
    bracket: dict,
    pool_config: BracketPoolConfig | None = None,
    upset_game: tuple[str, str] | None = None,
) -> dict[str, float]:
    """
    Proof-of-concept: compare expected R64 score for chalk vs one contrarian pick.

    upset_game: (favorite, underdog) - pick underdog in this game, chalk elsewhere.
    If None, picks the first 12v5 upset by seed (if present in bracket).
    """
    pool_config = pool_config or BracketPoolConfig()
    games = _build_first_round_matchups(bracket)

    chalk_ev = expected_bracket_score_r64(
        simulation_results, num_simulations, bracket, pool_config, picks=None
    )

    if upset_game is None:
        # Find a 12v5 game (5-seed favorite, 12-seed underdog)
        seed_index = _build_seed_index(bracket)
        for t1, t2 in games:
            s1, s2 = seed_index.get(t1, 99), seed_index.get(t2, 99)
            if (s1, s2) in [(5, 12), (12, 5)]:
                favorite = t1 if s1 < s2 else t2
                underdog = t2 if s1 < s2 else t1
                upset_game = (favorite, underdog)
                break
        if upset_game is None:
            return {"chalk_ev": chalk_ev, "contrarian_ev": chalk_ev, "contrarian_game": None}

    fav, underdog = upset_game
    picks = {}
    for t1, t2 in games:
        if (t1, t2) == (fav, underdog) or (t1, t2) == (underdog, fav):
            picks[(t1, t2)] = underdog
        else:
            r32 = simulation_results["round_32"]
            c1, c2 = r32.get(t1, 0), r32.get(t2, 0)
            picks[(t1, t2)] = t1 if c1 >= c2 else t2

    contrarian_ev = expected_bracket_score_r64(
        simulation_results, num_simulations, bracket, pool_config, picks=picks
    )

    return {
        "chalk_ev": chalk_ev,
        "contrarian_ev": contrarian_ev,
        "contrarian_game": upset_game,
    }


def _team_advance_prob(team: str, r32: dict, n: int) -> float:
    """P(team advances to R32). Handles play-in by summing both constituent teams."""
    if _is_play_in_team(team):
        a, b = _split_play_in(team)
        return (r32.get(a, 0) + r32.get(b, 0)) / n if n else 0
    return r32.get(team, 0) / n if n else 0


def _build_chalk_picks(
    simulation_results: dict,
    num_simulations: int,
    bracket: dict,
) -> dict[tuple[str, str], str]:
    """Build chalk picks: {(t1, t2): winner} for each R64 game."""
    r32 = simulation_results["round_32"]
    n = num_simulations
    games = _build_first_round_matchups(bracket)
    picks = {}
    for t1, t2 in games:
        p1 = _team_advance_prob(t1, r32, n)
        p2 = _team_advance_prob(t2, r32, n)
        picks[(t1, t2)] = t1 if p1 >= p2 else t2
    return picks


def _build_picks_with_upsets(
    chalk_picks: dict[tuple[str, str], str],
    upset_games: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Take chalk picks and flip specified games to underdog. upset_games: [(fav, underdog), ...]."""
    picks = dict(chalk_picks)
    for fav, underdog in upset_games:
        for (t1, t2) in list(picks.keys()):
            if (t1, t2) == (fav, underdog) or (t1, t2) == (underdog, fav):
                picks[(t1, t2)] = underdog
                break
    return picks


def _leverage_score(
    p_underdog: float,
    p_fav: float,
    hi_seed: int,
    lo_seed: int,
) -> float:
    """
    Simple leverage score for an upset pick: balances win probability, seed spread,
    and contrarian value (popularity proxy = p_fav, higher = more chalk-heavy = more leverage when we deviate).

    leverage = p_underdog * seed_diff * p_fav
    - p_underdog: chance upset hits (we want non-trivial)
    - seed_diff: magnitude of upset (12v5 > 9v8)
    - p_fav: popularity proxy (high = most pick favorite = we differentiate more)
    """
    seed_diff = lo_seed - hi_seed
    return p_underdog * seed_diff * p_fav


def rank_single_upset_swaps(
    simulation_results: dict,
    num_simulations: int,
    bracket: dict,
    pool_config: BracketPoolConfig,
    seed_index: dict[str, int] | None = None,
    top_n: int = 10,
) -> list[tuple[tuple[str, str], float, float, tuple[int, int], float, float, float]]:
    """
    Rank R64 games by EV impact of picking underdog instead of chalk.

    Returns list of (game, ev_with_upset, ev_change, (hi_seed, lo_seed), p_underdog, p_fav, leverage)
    sorted by ev_change descending (least-bad upsets first). top_n limits results.
    """
    r32 = simulation_results["round_32"]
    n = num_simulations
    games = _build_first_round_matchups(bracket)
    seed_idx = seed_index or _build_seed_index(bracket)

    chalk_picks = _build_chalk_picks(simulation_results, n, bracket)
    chalk_ev = expected_bracket_score_r64(
        simulation_results, n, bracket, pool_config, picks=chalk_picks
    )

    swaps = []
    for t1, t2 in games:
        p1 = _team_advance_prob(t1, r32, n)
        p2 = _team_advance_prob(t2, r32, n)
        fav = t1 if p1 >= p2 else t2
        underdog = t2 if p1 >= p2 else t1
        p_fav = max(p1, p2)
        p_underdog = min(p1, p2)

        # Only consider true upsets (underdog has lower prob)
        if p_underdog >= p_fav:
            continue

        picks_upset = _build_picks_with_upsets(chalk_picks, [(fav, underdog)])
        ev_upset = expected_bracket_score_r64(
            simulation_results, n, bracket, pool_config, picks=picks_upset
        )
        ev_change = ev_upset - chalk_ev  # negative; we want least negative

        s1 = seed_idx.get(_split_play_in(t1)[0] if _is_play_in_team(t1) else t1, 99)
        s2 = seed_idx.get(_split_play_in(t2)[0] if _is_play_in_team(t2) else t2, 99)
        hi_seed, lo_seed = (s1, s2) if s1 < s2 else (s2, s1)

        leverage = _leverage_score(p_underdog, p_fav, hi_seed, lo_seed)
        swaps.append(((fav, underdog), ev_upset, ev_change, (hi_seed, lo_seed), p_underdog, p_fav, leverage))

    swaps.sort(key=lambda x: -x[2])  # highest ev_change first (least bad)
    return swaps[:top_n]


# Weight for leverage when computing leverage-adjusted ranking (EV cost vs differentiation).
LEVERAGE_WEIGHT = 0.4  # leverage_adjusted = ev_change + LEVERAGE_WEIGHT * leverage


def rank_upsets_by_leverage(
    swaps: list[tuple[tuple[str, str], float, float, tuple[int, int], float, float, float]],
    leverage_weight: float = LEVERAGE_WEIGHT,
) -> list[tuple[tuple[str, str], float, float, tuple[int, int], float, float, float]]:
    """
    Re-rank upset swaps by leverage-adjusted value: ev_change + leverage_weight * leverage.
    Balances expected points cost with bracket differentiation potential.
    """
    scored = [(s, s[2] + leverage_weight * s[6]) for s in swaps]
    scored.sort(key=lambda x: -x[1])  # highest combined score first
    return [s[0] for s in scored]


def compare_bracket_variants(
    simulation_results: dict,
    num_simulations: int,
    bracket: dict,
    pool_config: BracketPoolConfig | None = None,
) -> dict[str, float]:
    """
    Compare expected R64 scores for chalk, light, medium, and high variance brackets.

    - pure chalk: all favorites
    - light contrarian: 1 upset (best single swap by EV)
    - medium contrarian: top 3 upsets
    - high variance: top 5 upsets
    """
    pool_config = pool_config or BracketPoolConfig()
    chalk_picks = _build_chalk_picks(simulation_results, num_simulations, bracket)
    ranked = rank_single_upset_swaps(
        simulation_results, num_simulations, bracket, pool_config, top_n=10
    )

    results = {}
    # Pure chalk
    results["chalk"] = expected_bracket_score_r64(
        simulation_results, num_simulations, bracket, pool_config, picks=chalk_picks
    )

    # Light: 1 best upset
    if ranked:
        light_picks = _build_picks_with_upsets(chalk_picks, [ranked[0][0]])
        results["light_contrarian"] = expected_bracket_score_r64(
            simulation_results, num_simulations, bracket, pool_config, picks=light_picks
        )
    else:
        results["light_contrarian"] = results["chalk"]

    # Medium: top 3 upsets
    if len(ranked) >= 3:
        medium_picks = _build_picks_with_upsets(chalk_picks, [r[0] for r in ranked[:3]])
        results["medium_contrarian"] = expected_bracket_score_r64(
            simulation_results, num_simulations, bracket, pool_config, picks=medium_picks
        )
    elif ranked:
        results["medium_contrarian"] = results["light_contrarian"]
    else:
        results["medium_contrarian"] = results["chalk"]

    # High variance: top 5 upsets
    if len(ranked) >= 5:
        high_picks = _build_picks_with_upsets(chalk_picks, [r[0] for r in ranked[:5]])
        results["high_variance"] = expected_bracket_score_r64(
            simulation_results, num_simulations, bracket, pool_config, picks=high_picks
        )
    elif ranked:
        results["high_variance"] = results["medium_contrarian"]
    else:
        results["high_variance"] = results["chalk"]

    return results


from collections import defaultdict


# ---------------------------------------------------------------------------
# Team metrics data structure
# ---------------------------------------------------------------------------

@dataclass
class TeamMetrics:
    """
    Canonical team input for the simulation model.

    All fields are used by win-probability and/or score-prediction logic.
    Replace fallback-generated values with real data (e.g. KenPom, custom CSV)
    by passing a metrics_overrides dict to build_team_metrics_registry().
    """
    name: str
    seed: int
    overall: float
    offensive_efficiency: float
    defensive_efficiency: float
    tempo: float
    experience: float
    coaching: float

    def to_win_factors(self) -> list[float]:
        """[overall, offense, defense, experience, coach] for calculate_win_probability."""
        return [
            self.overall,
            self.offensive_efficiency,
            self.defensive_efficiency,
            self.experience,
            self.coaching,
        ]

    def to_tempo_dict(self) -> dict[str, float]:
        """{offense, defense, tempo} for predict_game_score."""
        return {
            "offense": float(np.clip(self.offensive_efficiency, 98.0, 126.0)),
            "defense": float(np.clip(self.defensive_efficiency, 85.0, 112.0)),
            "tempo": float(np.clip(self.tempo, 62.0, 72.0)),
        }

    @classmethod
    def from_dict(cls, name: str, seed: int, d: dict[str, Any]) -> "TeamMetrics":
        """
        Build from a dict (e.g. CSV row or JSON). Missing keys use defaults.
        Use this when loading real data from external sources.
        """
        return cls(
            name=name,
            seed=seed,
            overall=float(d.get("overall", 75.0)),
            offensive_efficiency=float(d.get("offensive_efficiency", d.get("offense", 105.0))),
            defensive_efficiency=float(d.get("defensive_efficiency", d.get("defense", 100.0))),
            tempo=float(d.get("tempo", 67.0)),
            experience=float(d.get("experience", 6.5)),
            coaching=float(d.get("coaching", 6.5)),
        )

# 2026 NCAA Tournament bracket data (official first-round matchups)
bracket_data = {
    # Note: some 11/16 seeds come from the First Four and are represented as "A/B"
    "East": [
        ("Duke", 1), ("Siena", 16),
        ("Ohio St.", 8), ("TCU", 9),
        ("St. John's", 5), ("Northern Iowa", 12),
        ("Kansas", 4), ("Cal Baptist", 13),
        ("Louisville", 6), ("South Florida", 11),
        ("Michigan St.", 3), ("North Dakota St.", 14),
        ("UCLA", 7), ("UCF", 10),
        ("UConn", 2), ("Furman", 15)
    ],
    "West": [
        ("Arizona", 1), ("Long Island", 16),
        ("Villanova", 8), ("Utah St.", 9),
        ("Wisconsin", 5), ("High Point", 12),
        ("Arkansas", 4), ("Hawaii", 13),
        ("BYU", 6), ("Texas/NC State", 11),
        ("Gonzaga", 3), ("Kennesaw St.", 14),
        ("Miami (FL)", 7), ("Missouri", 10),
        ("Purdue", 2), ("Queens (N.C.)", 15)
    ],
    "South": [
        ("Florida", 1), ("Prairie View A&M/Lehigh", 16),
        ("Clemson", 8), ("Iowa", 9),
        ("Vanderbilt", 5), ("McNeese", 12),
        ("Nebraska", 4), ("Troy", 13),
        ("North Carolina", 6), ("VCU", 11),
        ("Illinois", 3), ("Penn", 14),
        ("Saint Mary's", 7), ("Texas A&M", 10),
        ("Houston", 2), ("Idaho", 15)
    ],
    "Midwest": [
        ("Michigan", 1), ("UMBC/Howard", 16),
        ("Georgia", 8), ("Saint Louis", 9),
        ("Texas Tech", 5), ("Akron", 12),
        ("Alabama", 4), ("Hofstra", 13),
        ("Tennessee", 6), ("Miami (Ohio)/SMU", 11),
        ("Virginia", 3), ("Wright St.", 14),
        ("Kentucky", 7), ("Santa Clara", 10),
        ("Iowa St.", 2), ("Tennessee St.", 15)
    ]
}

# Baseline team ratings (optional overrides). Teams not listed will be seeded-based defaults.
team_ratings = {
    "Duke": 86.8,
    "Arizona": 89.7,
    "Michigan": 84.0,
    "Florida": 85.9,
    "Houston": 93.8,
    "Purdue": 92.1,
    "UConn": 91.4,
    "Illinois": 86.7,
    "Gonzaga": 85.3,
    "Kansas": 85.7,
    "Tennessee": 90.2,
    "Kentucky": 87.9,
    "Alabama": 88.4,
    "Iowa St.": 89.2,
    "Michigan St.": 82.1,
    "Wisconsin": 83.5,
    "St. John's": 83.7,
    "Louisville": 76.2,
    "UCLA": 82.4,
    "North Carolina": 84.0,
    "Clemson": 84.3,
    "Vanderbilt": 75.6,
    "Saint Mary's": 84.1,
    "Texas Tech": 84.8,
    "BYU": 87.2,
    "Arkansas": 78.3,
}

def _is_play_in_team(team_name: str) -> bool:
    return "/" in team_name

def _split_play_in(team_name: str) -> tuple[str, str]:
    left, right = team_name.split("/", 1)
    return left.strip(), right.strip()

def _seed_based_overall(seed: int) -> float:
    seed = max(1, min(16, int(seed)))
    return 92.0 - (seed - 1) * 3.0  # 1-seed ~92, 16-seed ~47

# Seed hierarchy: ensure overrides don't invert seed-line strength (e.g. 2-seed > 1-seed).
# Applied when building metrics: 1-seeds floored, higher seeds capped below lower-seed baseline.
# Larger gap = stronger monotonicity; 1.5 keeps 2-seeds below 1-seeds even with off/def variation.
SEED_HIERARCHY_GAP = 1.5  # minimum gap between adjacent seed lines

def _iter_bracket_teams(bracket: dict) -> list[tuple[str, int]]:
    teams: list[tuple[str, int]] = []
    for region_matchups in bracket.values():
        for team, seed in region_matchups:
            teams.append((team, seed))
    return teams

def _team_seed_from_name(name: str) -> int:
    """
    Stable, deterministic seed derived from team name.

    This lets us generate synthetic attributes that are:
    - reproducible across runs
    - different across teams
    """
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _build_fallback_metrics(team: str, seed: int, overall_override: float | None = None) -> TeamMetrics:
    """
    Generate synthetic TeamMetrics when real data is unavailable.

    Uses deterministic per-team variation so similar overall ratings can still
    differ in offense/defense profile, experience, and coaching.

    When overall_override is provided, applies seed hierarchy bounds so that
    average seed-line strength is monotonic (1-seeds >= 2-seeds >= ...).
    """
    raw_overall = float(overall_override if overall_override is not None else _seed_based_overall(seed))
    # Enforce monotonic seed hierarchy: floor 1-seeds, cap higher seeds below lower-seed baseline
    seed_int = max(1, min(16, int(seed)))
    if seed_int == 1:
        base_overall = max(raw_overall, _seed_based_overall(1))
    else:
        cap = _seed_based_overall(seed_int - 1) - SEED_HIERARCHY_GAP
        base_overall = min(raw_overall, cap)
    rng = np.random.default_rng(_team_seed_from_name(team))

    style = rng.normal(0.0, 0.7)
    off_base = 100.0 + (base_overall - 70.0) * 0.8
    def_base = 105.0 - (base_overall - 70.0) * 0.6
    off_eff = float(np.clip(off_base + style * 3.0 + rng.normal(0.0, 2.0), 100.0, 125.0))
    def_eff = float(np.clip(def_base - style * 2.5 + rng.normal(0.0, 2.0), 88.0, 112.0))

    exp_base = 6.3 + (8 - min(seed, 16)) * 0.08
    experience = float(np.clip(exp_base + rng.normal(0.0, 0.7), 4.0, 9.5))

    coach_base = 6.0 + (base_overall - 75.0) / 12.0
    coaching = float(np.clip(coach_base + rng.normal(0.0, 0.6), 4.0, 9.5))

    tempo = 66.0 + (base_overall - 70.0) * 0.12 + (8 - seed) * 0.04
    tempo = float(np.clip(tempo, 62.0, 72.0))

    return TeamMetrics(
        name=team,
        seed=seed,
        overall=base_overall,
        offensive_efficiency=off_eff,
        defensive_efficiency=def_eff,
        tempo=tempo,
        experience=experience,
        coaching=coaching,
    )


def build_team_metrics_registry(
    bracket: dict,
    *,
    overall_overrides: dict[str, float] | None = None,
    metrics_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, TeamMetrics]:
    """
    Build the canonical team metrics registry from bracket + optional overrides.

    - overall_overrides: team_name -> overall rating (e.g. team_ratings). Used when
      generating fallback metrics.
    - metrics_overrides: team_name -> full metrics dict. When present, uses
      TeamMetrics.from_dict() instead of fallback. Use this to inject real data
      from CSV, KenPom, etc.

    Example for future real data (CSV/dict):
        import csv
        overrides = {}
        with open("kenpom_2026.csv") as f:
            for row in csv.DictReader(f):
                overrides[row["team"]] = {
                    "overall": float(row["rating"]),
                    "offensive_efficiency": float(row["adj_o"]),
                    "defensive_efficiency": float(row["adj_d"]),
                    "tempo": float(row["tempo"]),
                    "experience": 6.5,  # if not in CSV
                    "coaching": 6.5,
                }
        registry = build_team_metrics_registry(bracket_data, metrics_overrides=overrides)
    """
    overall_overrides = overall_overrides or {}
    metrics_overrides = metrics_overrides or {}
    registry: dict[str, TeamMetrics] = {}

    for team, seed in _iter_bracket_teams(bracket):
        teams_to_add = [team]
        if _is_play_in_team(team):
            teams_to_add = list(_split_play_in(team))

        for t in teams_to_add:
            if t in registry:
                continue
            if t in metrics_overrides:
                registry[t] = TeamMetrics.from_dict(t, seed, metrics_overrides[t])
            else:
                registry[t] = _build_fallback_metrics(t, seed, overall_overrides.get(t))

    return registry


# Build registry from bracket + team_ratings. Pass metrics_overrides for real data.
team_metrics_registry = build_team_metrics_registry(bracket_data, overall_overrides=team_ratings)

# Backward-compatible views (used by calculate_win_probability and predict_game_score)
team_stats = {name: m.to_win_factors() for name, m in team_metrics_registry.items()}
team_tempo_stats = {name: m.to_tempo_dict() for name, m in team_metrics_registry.items()}

# Win-probability calibration constants.
# Tune to balance chalk vs upsets: reduce overall / SEED_EDGE for more upsets.
WIN_WEIGHTS = {
    "overall": 0.48,   # was 0.55; lower reduces favorite dominance
    "offense": 0.20,
    "defense": 0.19,
    "experience": 0.06,
    "coach": 0.07,
}

# Logistic scale: larger → flatter curve → more upsets. Smaller → more chalk.
LOGISTIC_SCALE = 14.0  # was 12.0; modest increase for realistic early-round upset frequency

# Seed-based nudge: better seeds get a small bump. Lower = less chalk amplification.
SEED_EDGE_STRENGTH = 0.65  # was 0.8; reduced to avoid over-amplifying favorites, but keep 1v16 strong

# Extra nudge for 1v16 so average stays above 2v15 despite play-in 16 variance.
SEED_1v16_BONUS = 2.0  # added to seed_edge when higher seed is 1 and lower is 16

def _composite_rating(stats: list[float]) -> float:
    """
    Map raw team factors -> a single composite rating.

    This keeps the combination logic explicit and easy to tune.
    """
    overall, off_eff, def_eff, exp, coach = stats

    # Normalise offense/defense into rough 0–1 scales.
    off_component = (off_eff - 105.0) / 12.0  # ~[-1, +1] for typical teams
    def_component = (100.0 - def_eff) / 10.0  # better defense → larger positive

    exp_component = (exp - 6.5) / 3.0
    coach_component = (coach - 6.5) / 3.0

    rating = (
        WIN_WEIGHTS["overall"] * overall
        + WIN_WEIGHTS["offense"] * off_component * 10.0
        + WIN_WEIGHTS["defense"] * def_component * 10.0
        + WIN_WEIGHTS["experience"] * exp_component * 10.0
        + WIN_WEIGHTS["coach"] * coach_component * 10.0
    )
    return rating

def calculate_win_probability(
    team1: str,
    team2: str,
    _team_stats: dict[str, list[float]] | None = None,
    _bracket: dict | None = None,
) -> float:
    """
    Calculate win probability using team factors and a logistic model.

    Tuning notes:
    - Increase LOGISTIC_SCALE to make matchups flatter (more upsets).
    - Decrease LOGISTIC_SCALE to make ratings more decisive (more chalk).

    _team_stats, _bracket: optional overrides for backtesting; when None, use globals.
    """
    stats_src = _team_stats if _team_stats is not None else team_stats
    bracket_src = _bracket if _bracket is not None else bracket_data

    stats1 = stats_src[team1]
    stats2 = stats_src[team2]

    rating1 = _composite_rating(stats1)
    rating2 = _composite_rating(stats2)

    # Optional small seed-based nudge: better seeds get a *tiny* bump,
    # but this should never dominate the actual rating.
    seed1 = seed2 = None
    for region, matchups in bracket_src.items():
        for name, seed in matchups:
            if name == team1:
                seed1 = seed
            if name == team2:
                seed2 = seed
        if seed1 is not None and seed2 is not None:
            break

    if seed1 is not None and seed2 is not None:
        seed_edge = (seed2 - seed1) * SEED_EDGE_STRENGTH
        # Ensure 1v16 average stays above 2v15 (play-in 16s can have high variance)
        if (seed1, seed2) in ((1, 16), (16, 1)):
            seed_edge += SEED_1v16_BONUS if seed1 == 1 else -SEED_1v16_BONUS
    else:
        seed_edge = 0.0

    rating_diff = rating1 - rating2 + seed_edge

    # Logistic transform into (0,1). Centered at 0 → 50/50.
    x = rating_diff / LOGISTIC_SCALE
    prob = 1.0 / (1.0 + np.exp(-x))

    # Guard against exact 0 or 1 from extreme diffs / numeric limits.
    return float(np.clip(prob, 1e-4, 1.0 - 1e-4))


def _canonical_seed_win_prob(hi_seed: int, lo_seed: int) -> float:
    """
    Win probability for higher seed when both teams use seed-based ratings only.
    Uses _seed_based_overall and deterministic off/def/exp/coach (no per-team noise).
    Ensures monotonic seed-line expectation: 1v16 >= 2v15 >= ... >= 8v9.
    """
    def _seed_stats(s: int) -> list[float]:
        overall = _seed_based_overall(s)
        off_base = 100.0 + (overall - 70.0) * 0.8
        def_base = 105.0 - (overall - 70.0) * 0.6
        exp_base = 6.3 + (8 - min(s, 16)) * 0.08
        coach_base = 6.0 + (overall - 75.0) / 12.0
        return [overall, off_base, def_base, exp_base, coach_base]

    r1 = _composite_rating(_seed_stats(hi_seed))
    r2 = _composite_rating(_seed_stats(lo_seed))
    seed_edge = (lo_seed - hi_seed) * SEED_EDGE_STRENGTH
    if (hi_seed, lo_seed) == (1, 16):
        seed_edge += SEED_1v16_BONUS
    rating_diff = r1 - r2 + seed_edge
    x = rating_diff / LOGISTIC_SCALE
    return float(np.clip(1.0 / (1.0 + np.exp(-x)), 1e-4, 1.0 - 1e-4))


def print_seed_matchup_diagnostics(bracket: dict | None = None):
    """
    Calibration helper: prints implied upset probabilities for common seed matchups.

    Two tables:
    1. Raw: average higher-seed win prob from actual field (team-specific ratings)
    2. Normalized: seed-line expectation from model inputs only (_seed_based_overall, no overrides)

    bracket: optional override; when None, uses bracket_data.
    """
    br = bracket if bracket is not None else bracket_data
    pairs = [
        (1, 16),
        (2, 15),
        (3, 14),
        (4, 13),
        (5, 12),
        (6, 11),
        (7, 10),
        (8, 9),
    ]

    # Build (seed, region) -> team name
    seed_index: dict[tuple[int, str], str] = {}
    for region, teams in br.items():
        for name, seed in teams:
            key = (seed, region)
            if key not in seed_index:
                if _is_play_in_team(name):
                    rep, _ = _split_play_in(name)
                    seed_index[key] = rep
                else:
                    seed_index[key] = name

    regions = ["South", "West", "East", "Midwest"]

    # 1) Raw: actual field (team overrides, synthetic factors)
    print("\nSeed matchup diagnostics (higher-seed win probability):")
    print("\n  Raw (actual field, avg over 4 regions):")
    raw_probs = {}
    for hi, lo in pairs:
        probs = []
        for region in regions:
            hi_team = seed_index.get((hi, region))
            lo_team = seed_index.get((lo, region))
            if hi_team and lo_team:
                p = calculate_win_probability(hi_team, lo_team, _bracket=br)
                probs.append(p)
        if probs:
            avg = sum(probs) / len(probs)
            raw_probs[(hi, lo)] = avg
            print(f"    {hi} vs {lo}: {avg:.3f}")

    # 2) Normalized: seed-line expectation (model inputs only, monotonic by design)
    print("\n  Normalized (seed-line expectation, no team overrides):")
    for hi, lo in pairs:
        canon = _canonical_seed_win_prob(hi, lo)
        print(f"    {hi} vs {lo}: {canon:.3f}")

def simulate_game(
    team1: str,
    team2: str,
    rng: np.random.Generator,
    _team_stats: dict[str, list[float]] | None = None,
    _bracket: dict | None = None,
) -> str:
    """Simulate a single game between two teams."""
    prob_team1 = calculate_win_probability(team1, team2, _team_stats=_team_stats, _bracket=_bracket)
    return team1 if rng.random() < prob_team1 else team2

def simulate_round(
    matchups: list[tuple[str, int]],
    rng: np.random.Generator,
    _team_stats: dict[str, list[float]] | None = None,
    _bracket: dict | None = None,
) -> list[tuple[str, int]]:
    """Simulate one round of games."""
    winners = []
    for i in range(0, len(matchups), 2):
        team1, seed1 = matchups[i]
        team2, seed2 = matchups[i + 1]
        winner = simulate_game(team1, team2, rng, _team_stats=_team_stats, _bracket=_bracket)
        winner_seed = seed1 if winner == team1 else seed2
        winners.append((winner, winner_seed))
    return winners

def validate_bracket(bracket: dict) -> None:
    expected_regions = {"South", "West", "East", "Midwest"}
    got_regions = set(bracket.keys())
    if got_regions != expected_regions:
        raise ValueError(f"Bracket regions must be {sorted(expected_regions)}; got {sorted(got_regions)}")

    for region, matchups in bracket.items():
        if len(matchups) != 16:
            raise ValueError(f"{region} must have 16 team entries (8 games); got {len(matchups)}")

        seeds = [seed for _, seed in matchups]
        if set(seeds) != set(range(1, 17)):
            raise ValueError(f"{region} seeds must be 1-16 exactly once; got {sorted(seeds)}")

def resolve_first_four(
    bracket: dict,
    rng: np.random.Generator,
    _team_stats: dict[str, list[float]] | None = None,
    _bracket: dict | None = None,
) -> dict:
    """
    Replace any play-in placeholders like 'A/B' with a simulated winner (A vs B).
    This is done per-simulation, so play-in uncertainty is reflected in outcomes.
    """
    resolved: dict = {}
    for region, matchups in bracket.items():
        new_matchups = []
        for team, seed in matchups:
            if _is_play_in_team(team):
                a, b = _split_play_in(team)
                winner = simulate_game(a, b, rng, _team_stats=_team_stats, _bracket=_bracket)
                new_matchups.append((winner, seed))
            else:
                new_matchups.append((team, seed))
        resolved[region] = new_matchups
    return resolved

def simulate_tournament(
    bracket_data,
    num_simulations=1000000,
    *,
    rng: np.random.Generator,
    progress_every: int = 0,
    _team_stats: dict[str, list[float]] | None = None,
    _bracket: dict | None = None,
):
    """Simulate the entire tournament multiple times."""
    results = {
        'championship_wins': defaultdict(int),
        'final_four': defaultdict(int),
        'elite_eight': defaultdict(int),
        'sweet_sixteen': defaultdict(int),
        'round_32': defaultdict(int)
    }
    stats = _team_stats
    br = _bracket

    for sim in range(num_simulations):
        if progress_every and sim % progress_every == 0:
            print(f"Running simulation {sim}/{num_simulations}")
            
        current_bracket = resolve_first_four(bracket_data, rng, _team_stats=stats, _bracket=br)
        region_winners: dict[str, tuple[str, int]] = {}
        
        for region in current_bracket:
            # Round of 32
            current_bracket[region] = simulate_round(current_bracket[region], rng, _team_stats=stats, _bracket=br)
            for team, _ in current_bracket[region]:
                results['round_32'][team] += 1
            
            # Sweet 16
            current_bracket[region] = simulate_round(current_bracket[region], rng, _team_stats=stats, _bracket=br)
            for team, _ in current_bracket[region]:
                results['sweet_sixteen'][team] += 1
            
            # Elite 8
            current_bracket[region] = simulate_round(current_bracket[region], rng, _team_stats=stats, _bracket=br)
            for team, _ in current_bracket[region]:
                results['elite_eight'][team] += 1
            
            # Final 4
            regional_winner = simulate_round(current_bracket[region], rng, _team_stats=stats, _bracket=br)[0]
            region_winners[region] = regional_winner
            results['final_four'][regional_winner[0]] += 1
        
        # Final Four pairings (2026): East vs South, West vs Midwest
        final_four = [
            region_winners["East"],
            region_winners["South"],
            region_winners["West"],
            region_winners["Midwest"],
        ]

        # Championship
        championship_game = simulate_round(final_four, rng, _team_stats=stats, _bracket=br)
        champion = simulate_round(championship_game, rng, _team_stats=stats, _bracket=br)[0][0]
        results['championship_wins'][champion] += 1
    
    return results

def print_results(results, num_simulations):
    """Print formatted simulation results."""
    rounds = [
        ("Championship", 'championship_wins', 10),
        ("Final Four", 'final_four', 12),
        ("Elite Eight", 'elite_eight', 16),
        ("Sweet Sixteen", 'sweet_sixteen', 20),
        ("Round of 32", 'round_32', 32)
    ]
    
    for title, key, num_teams in rounds:
        print(f"\n{title} Odds:")
        sorted_teams = sorted(results[key].items(), key=lambda x: x[1], reverse=True)[:num_teams]
        for team, appearances in sorted_teams:
            prob = (appearances/num_simulations)*100
            print(f"{team}: {prob:.1f}% ({appearances} times)")

def _build_seed_index(bracket: dict) -> dict[str, int]:
    """
    Map team name -> original seed using the bracket definition.
    Play-in placeholders are expanded so that both underlying teams share the seed.
    """
    index: dict[str, int] = {}
    for region_teams in bracket.values():
        for name, seed in region_teams:
            if _is_play_in_team(name):
                a, b = _split_play_in(name)
                index.setdefault(a, seed)
                index.setdefault(b, seed)
            else:
                index.setdefault(name, seed)
    return index

def run_sanity_diagnostics(results, num_simulations: int, bracket: dict):
    """
    Post-simulation sanity diagnostics by seed.

    Summarises odds by seed for key rounds and emits light warnings if patterns
    look obviously strange (too many early exits for top seeds, too much chalk, etc.).
    """
    seed_index = _build_seed_index(bracket)

    # Aggregate probabilities by seed.
    def _agg(bucket_key: str) -> dict[int, float]:
        bucket = results[bucket_key]
        seed_probs: dict[int, float] = {}
        for team, count in bucket.items():
            seed = seed_index.get(team)
            if seed is None:
                continue
            seed_probs.setdefault(seed, 0.0)
            seed_probs[seed] += count / num_simulations
        return seed_probs

    champ_by_seed = _agg("championship_wins")
    ff_by_seed = _agg("final_four")
    s16_by_seed = _agg("sweet_sixteen")
    r32_by_seed = _agg("round_32")

    print("\n=== Seed-level sanity diagnostics ===")
    print("Seed | Champ% | Final4% | Sweet16% | R32%")
    for seed in range(1, 17):
        c = champ_by_seed.get(seed, 0.0) * 100
        f4 = ff_by_seed.get(seed, 0.0) * 100
        s16 = s16_by_seed.get(seed, 0.0) * 100
        r32 = r32_by_seed.get(seed, 0.0) * 100
        if c + f4 + s16 + r32 == 0:
            continue
        print(f"{seed:>4} | {c:6.2f} | {f4:7.2f} | {s16:9.2f} | {r32:5.2f}")

    # Simple heuristics for suspicious patterns.
    warnings = []

    # Top seeds exiting too early.
    top_seed_r32 = sum(r32_by_seed.get(s, 0.0) for s in (1, 2)) / 2 or 0.0
    if top_seed_r32 < 0.9:
        warnings.append("Top seeds (1–2) reach Round of 32 less than 90% on average.")

    # Too many double-digit seeds in Final Four.
    low_seed_ff = sum(ff_by_seed.get(s, 0.0) for s in range(10, 17))
    if low_seed_ff > 0.25:
        warnings.append("Double-digit seeds account for more than 25% of Final Four appearances.")

    # Too much chalk: 1–4 seeds dominate Final Four.
    high_seed_ff = sum(ff_by_seed.get(s, 0.0) for s in range(1, 5))
    if high_seed_ff > 0.95:
        warnings.append("Seeds 1–4 account for more than 95% of Final Four appearances (very chalky).")

    if warnings:
        print("\nSanity warnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nSanity checks: no obvious red flags detected.")

def print_bracket_predictions(results, bracket_data):
    """
    Print visual bracket representations based on simulation results.

    Two views:
      1. Coherent matchup-based bracket (uses head-to-head win probabilities).
      2. Marginal-probability bracket (uses advancement frequencies by round).
    """
    def get_most_likely_winner(team1, team2, round_results):
        """Helper for marginal bracket: decide winner from advancement counts."""
        prob1 = round_results.get(team1[0], 0)
        prob2 = round_results.get(team2[0], 0)
        return team1 if prob1 > prob2 else team2

    def deterministic_winner(pair1, pair2):
        """Helper for coherent bracket: decide winner from head-to-head probability."""
        (t1, s1), (t2, s2) = pair1, pair2
        # Handle play-in placeholders by using the first named team for rating purposes.
        name1 = _split_play_in(t1)[0] if _is_play_in_team(t1) else t1
        name2 = _split_play_in(t2)[0] if _is_play_in_team(t2) else t2
        p = calculate_win_probability(name1, name2)
        if p > 0.5:
            return (t1, s1)
        if p < 0.5:
            return (t2, s2)
        # Exact tie: choose better seed.
        return (t1, s1) if s1 < s2 else (t2, s2)

    print("\n=== 2026 NCAA Tournament Predictions ===\n")

    # ------------------------------------------------------------------
    # 1) Coherent matchup-based bracket
    # ------------------------------------------------------------------
    print("=== Coherent matchup-based bracket (game win probabilities) ===")
    matchup_region_winners = {}

    for region in ['South', 'West', 'East', 'Midwest']:
        print(f"\n{region} Region:")
        print("=" * 50)

        # First Round
        current_round = bracket_data[region][:]
        print("\nFirst Round:")
        for i in range(0, len(current_round), 2):
            team1 = current_round[i]
            team2 = current_round[i + 1]
            winner = deterministic_winner(team1, team2)
            print(f"({team1[1]}) {team1[0]} vs ({team2[1]}) {team2[0]} → ({winner[1]}) {winner[0]}")
        # Advance winners
        next_round = []
        for i in range(0, len(current_round), 2):
            team1 = current_round[i]
            team2 = current_round[i + 1]
            winner = deterministic_winner(team1, team2)
            next_round.append(winner)
        current_round = next_round

        # Round of 32
        print("\nRound of 32:")
        next_round = []
        for i in range(0, len(current_round), 2):
            team1 = current_round[i]
            team2 = current_round[i + 1]
            winner = deterministic_winner(team1, team2)
            print(f"({team1[1]}) {team1[0]} vs ({team2[1]}) {team2[0]} → ({winner[1]}) {winner[0]}")
            next_round.append(winner)
        current_round = next_round

        # Sweet 16
        print("\nSweet 16:")
        next_round = []
        for i in range(0, len(current_round), 2):
            team1 = current_round[i]
            team2 = current_round[i + 1]
            winner = deterministic_winner(team1, team2)
            print(f"({team1[1]}) {team1[0]} vs ({team2[1]}) {team2[0]} → ({winner[1]}) {winner[0]}")
            next_round.append(winner)
        current_round = next_round

        # Regional Final (Elite 8)
        team1, team2 = current_round[0], current_round[1]
        winner = deterministic_winner(team1, team2)
        matchup_region_winners[region] = winner
        print(f"\nRegional Final: ({team1[1]}) {team1[0]} vs ({team2[1]}) {team2[0]} → ({winner[1]}) {winner[0]}")

    # Final Four (coherent)
    print("\n=== Final Four (coherent bracket) ===")
    east_south = deterministic_winner(matchup_region_winners['East'], matchup_region_winners['South'])
    west_midwest = deterministic_winner(matchup_region_winners['West'], matchup_region_winners['Midwest'])
    print(
        f"East/South: ({matchup_region_winners['East'][1]}) {matchup_region_winners['East'][0]} vs "
        + f"({matchup_region_winners['South'][1]}) {matchup_region_winners['South'][0]} → ({east_south[1]}) {east_south[0]}"
    )
    print(
        f"West/Midwest: ({matchup_region_winners['West'][1]}) {matchup_region_winners['West'][0]} vs "
        + f"({matchup_region_winners['Midwest'][1]}) {matchup_region_winners['Midwest'][0]} → ({west_midwest[1]}) {west_midwest[0]}"
    )

    coherent_champ = deterministic_winner(east_south, west_midwest)
    print("\nChampionship Game (coherent bracket):")
    print(
        f"({east_south[1]}) {east_south[0]} vs ({west_midwest[1]}) {west_midwest[0]} → "
        f"({coherent_champ[1]}) {coherent_champ[0]}"
    )

    # ------------------------------------------------------------------
    # 2) Marginal-probability bracket
    # ------------------------------------------------------------------
    print("\n=== Marginal-probability bracket (advancement frequencies) ===")

    # Store predictions for each round (by region + slot).
    predictions = {
        'round_32': {},
        'sweet_16': {},
        'elite_8': {},
    }

    for region in ['South', 'West', 'East', 'Midwest']:
        print(f"\n{region} Region:")
        print("=" * 50)
        
        first_round = bracket_data[region]
        matchups = [
            (0,1),   # 1 vs 16
            (2,3),   # 8 vs 9
            (4,5),   # 5 vs 12
            (6,7),   # 4 vs 13
            (8,9),   # 6 vs 11
            (10,11), # 3 vs 14
            (12,13), # 7 vs 10
            (14,15)  # 2 vs 15
        ]
        
        print("\nFirst Round:")
        for i, (idx1, idx2) in enumerate(matchups):
            team1, team2 = first_round[idx1], first_round[idx2]
            winner = get_most_likely_winner(team1, team2, results['round_32'])
            predictions['round_32'][f"{region}_{i}"] = winner
            print(f"({team1[1]}) {team1[0]} vs ({team2[1]}) {team2[0]} → ({winner[1]}) {winner[0]}")
        
        # Round of 32
        print("\nRound of 32:")
        r32_matchups = [
            (0,1),  # 1/16 vs 8/9
            (2,3),  # 5/12 vs 4/13
            (4,5),  # 6/11 vs 3/14
            (6,7)   # 7/10 vs 2/15
        ]
        
        for i, (idx1, idx2) in enumerate(r32_matchups):
            team1 = predictions['round_32'][f"{region}_{idx1}"]
            team2 = predictions['round_32'][f"{region}_{idx2}"]
            winner = get_most_likely_winner(team1, team2, results['sweet_sixteen'])
            predictions['sweet_16'][region + str(i)] = winner
            print(f"({team1[1]}) {team1[0]} vs ({team2[1]}) {team2[0]} → ({winner[1]}) {winner[0]}")
        
        # Sweet 16
        print("\nSweet 16:")
        top_winner = get_most_likely_winner(
            predictions['sweet_16'][region + '0'],
            predictions['sweet_16'][region + '1'],
            results['elite_eight']
        )
        bottom_winner = get_most_likely_winner(
            predictions['sweet_16'][region + '2'],
            predictions['sweet_16'][region + '3'],
            results['elite_eight']
        )
        
        winner = get_most_likely_winner(top_winner, bottom_winner, results['final_four'])
        predictions['elite_8'][region] = winner
        print(f"Regional Final: ({top_winner[1]}) {top_winner[0]} vs ({bottom_winner[1]}) {bottom_winner[0]} → ({winner[1]}) {winner[0]}")
        
    # Final Four (marginal)
    print("\n=== Final Four (marginal bracket) ===")
    east_south = get_most_likely_winner(
        predictions['elite_8']['East'],
        predictions['elite_8']['South'],
        results['final_four']
    )
    west_midwest = get_most_likely_winner(
        predictions['elite_8']['West'],
        predictions['elite_8']['Midwest'],
        results['final_four']
    )
    
    print(
        f"East/South: ({predictions['elite_8']['East'][1]}) {predictions['elite_8']['East'][0]} vs "
        + f"({predictions['elite_8']['South'][1]}) {predictions['elite_8']['South'][0]} → ({east_south[1]}) {east_south[0]}"
    )
    print(
        f"West/Midwest: ({predictions['elite_8']['West'][1]}) {predictions['elite_8']['West'][0]} vs "
        + f"({predictions['elite_8']['Midwest'][1]}) {predictions['elite_8']['Midwest'][0]} → ({west_midwest[1]}) {west_midwest[0]}"
    )
    
    # Championship (marginal)
    print("\nChampionship Game (marginal bracket):")
    champion = get_most_likely_winner(east_south, west_midwest, results['championship_wins'])
    print(f"({east_south[1]}) {east_south[0]} vs ({west_midwest[1]}) {west_midwest[0]} → ({champion[1]}) {champion[0]}")
    
    print("\n=== 2026 Champion (marginal bracket) ===")
    print(f"({champion[1]}) {champion[0]}")


# Score model calibration constants.
# These are intentionally easy to tune.
# NCAA tournament games tend slightly lower than regular-season D1 averages.
LEAGUE_AVG_PPP = 1.02  # was 1.06; modest reduction for neutral-court tourney totals
LEAGUE_AVG_TEMPO = 67.0  # possessions per game (40 minutes)

# How strongly offense/defense affect PPP (bounded; see `_bounded_ppp_adjustment`).
PPP_ADJ_STRENGTH = 0.06  # was 0.075; slightly less boost to cap high-scoring games
EFFICIENCY_SCALE = 15.0  # points/100 possessions scale for "one strong deviation"

# Randomness controls (preserve variance for realistic spread).
TEMPO_GAME_SD = 2.0  # game-to-game possessions variability
PPP_GAME_SD = 0.032  # was 0.035; slight reduction in shared upward drift
PPP_TEAM_SD = 0.042  # was 0.045; slight reduction, preserves matchup variance

# Guardrails for realism.
PPP_MIN = 0.85
PPP_MAX = 1.22  # was 1.25; 160+ totals possible but less common
POSS_MIN = 58.0  # possession bounds (tourney games often slightly slower)
POSS_MAX = 72.0  # was 60-75
SCORE_MIN = 45
SCORE_MAX = 95
SCORE_NOISE_SD = 2.5  # final rounding/noise per team (preserves variance)

def _bounded_ppp_adjustment(off_eff: float, opp_def_eff: float) -> float:
    """
    Convert (offense, opponent defense) efficiencies (pts/100 poss) into a *modest*
    additive PPP adjustment around league average.

    We use tanh() to bound the adjustment so no matchup explodes totals.
    """
    # Higher off_eff is better; lower opp_def_eff is better defense.
    off_z = (off_eff - 110.0) / EFFICIENCY_SCALE
    def_z = (100.0 - opp_def_eff) / EFFICIENCY_SCALE
    return PPP_ADJ_STRENGTH * (np.tanh(off_z) + np.tanh(def_z))

def predict_game_score(team1, team2, rng: np.random.Generator):
    """
    Predict the score for a game between two teams.

    Calibration intent:
    - Typical totals ~130–150 for neutral-court NCAA tournament games
    - Low-scoring games are possible; high-scoring (160+) possible but less common
    """
    if team1 not in team_tempo_stats or team2 not in team_tempo_stats:
        raise ValueError(f"Missing stats for {team1} or {team2}")
    
    t1 = team_tempo_stats[team1]
    t2 = team_tempo_stats[team2]

    # 1) Possessions: average both tempos + a small game-level tempo wobble.
    base_possessions = (t1["tempo"] + t2["tempo"]) / 2
    possessions = base_possessions + rng.normal(0, TEMPO_GAME_SD)
    possessions = float(np.clip(possessions, POSS_MIN, POSS_MAX))

    # 2) PPP: start at league average, then apply bounded offense/defense adjustments.
    game_env = rng.normal(0, PPP_GAME_SD)  # shared shift affecting both teams

    t1_adj = _bounded_ppp_adjustment(t1["offense"], t2["defense"])
    t2_adj = _bounded_ppp_adjustment(t2["offense"], t1["defense"])

    team1_ppp = LEAGUE_AVG_PPP + game_env + t1_adj + rng.normal(0, PPP_TEAM_SD)
    team2_ppp = LEAGUE_AVG_PPP + game_env + t2_adj + rng.normal(0, PPP_TEAM_SD)

    team1_ppp = float(np.clip(team1_ppp, PPP_MIN, PPP_MAX))
    team2_ppp = float(np.clip(team2_ppp, PPP_MIN, PPP_MAX))

    # 3) Convert to points. A small final rounding noise approximates end-game fouling,
    #    late threes, and the fact that PPP isn't perfectly stationary.
    team1_score = round(team1_ppp * possessions + rng.normal(0, SCORE_NOISE_SD))
    team2_score = round(team2_ppp * possessions + rng.normal(0, SCORE_NOISE_SD))
    
    # Ensure reasonable scores
    team1_score = max(SCORE_MIN, min(SCORE_MAX, team1_score))
    team2_score = max(SCORE_MIN, min(SCORE_MAX, team2_score))
    
    return team1_score, team2_score

def _resolve_team_for_win_prob(name: str) -> str:
    """Use first team of play-in placeholder for calculate_win_probability lookup."""
    return _split_play_in(name)[0] if _is_play_in_team(name) else name


def _build_projected_matchups(bracket: dict) -> tuple[list[tuple[tuple[str, int], tuple[str, int]]], list[tuple[tuple[str, int], tuple[str, int]]], tuple[tuple[str, int], tuple[str, int]] | None]:
    """
    Build projected Elite Eight, Final Four, and Championship matchups from the
    coherent (deterministic win-probability) bracket.
    Returns (elite_eight_matchups, final_four_matchups, championship_matchup).
    """
    def winner(pair1, pair2):
        (t1, s1), (t2, s2) = pair1, pair2
        n1 = _resolve_team_for_win_prob(t1)
        n2 = _resolve_team_for_win_prob(t2)
        p = calculate_win_probability(n1, n2)
        if p > 0.5:
            return (t1, s1)
        if p < 0.5:
            return (t2, s2)
        return (t1, s1) if s1 < s2 else (t2, s2)

    elite_eight = []
    region_winners = {}

    for region in ["South", "West", "East", "Midwest"]:
        current = bracket[region][:]
        for _ in range(3):  # R64 -> R32 -> S16
            next_round = []
            for i in range(0, len(current), 2):
                w = winner(current[i], current[i + 1])
                next_round.append(w)
            current = next_round
        team1, team2 = current[0], current[1]
        elite_eight.append((team1, team2))
        region_winners[region] = winner(team1, team2)

    east_south = winner(region_winners["East"], region_winners["South"])
    west_midwest = winner(region_winners["West"], region_winners["Midwest"])
    final_four = [
        (region_winners["East"], region_winners["South"]),
        (region_winners["West"], region_winners["Midwest"]),
    ]
    championship = (east_south, west_midwest)
    return elite_eight, final_four, championship


def print_matchup_win_probabilities(bracket: dict):
    """
    Print head-to-head win probabilities for projected Elite Eight, Final Four,
    and Championship matchups (from the coherent bracket).
    """
    elite_eight, final_four, championship = _build_projected_matchups(bracket)

    print("\n=== Matchup Win Probabilities ===")

    print("\nElite Eight (Regional Finals):")
    for (t1, s1), (t2, s2) in elite_eight:
        n1 = _resolve_team_for_win_prob(t1)
        n2 = _resolve_team_for_win_prob(t2)
        p1 = calculate_win_probability(n1, n2)
        p2 = 1.0 - p1
        print(f"  {t1} vs {t2}: {t1} {p1*100:.1f}%, {t2} {p2*100:.1f}%")

    print("\nFinal Four:")
    for (t1, s1), (t2, s2) in final_four:
        n1 = _resolve_team_for_win_prob(t1)
        n2 = _resolve_team_for_win_prob(t2)
        p1 = calculate_win_probability(n1, n2)
        p2 = 1.0 - p1
        print(f"  {t1} vs {t2}: {t1} {p1*100:.1f}%, {t2} {p2*100:.1f}%")

    if championship:
        (t1, s1), (t2, s2) = championship
        n1 = _resolve_team_for_win_prob(t1)
        n2 = _resolve_team_for_win_prob(t2)
        p1 = calculate_win_probability(n1, n2)
        p2 = 1.0 - p1
        print("\nChampionship:")
        print(f"  {t1} vs {t2}: {t1} {p1*100:.1f}%, {t2} {p2*100:.1f}%")


def analyze_championship_scoring(final_four_teams, rng: np.random.Generator, *, sims_per_matchup: int = 1000):
    """Analyze potential scoring outcomes for all possible championship matchups"""
    print("\n=== Championship Game Scoring Analysis ===")
    
    # Generate all possible championship matchups
    matchups = []
    for i in range(len(final_four_teams)):
        for j in range(i + 1, len(final_four_teams)):
            team1 = final_four_teams[i][0]
            team2 = final_four_teams[j][0]
            matchups.append((team1, team2))
    
    # Analyze each potential matchup
    for team1, team2 in matchups:
        print(f"\n{team1} vs {team2}")
        print("-" * 40)
        
        scores = []
        for _ in range(sims_per_matchup):
            score1, score2 = predict_game_score(team1, team2, rng)
            scores.append((score1, score2))
        
        # Calculate statistics
        total_scores = [s1 + s2 for s1, s2 in scores]
        avg_total = sum(total_scores) / len(total_scores)
        avg_score1 = sum(s[0] for s in scores) / len(scores)
        avg_score2 = sum(s[1] for s in scores) / len(scores)
        
        # Find most common scores
        from collections import Counter
        score_counter = Counter(scores)
        
        print(f"Average Score: {avg_score1:.1f}-{avg_score2:.1f}")
        print(f"Average Total: {avg_total:.1f}")
        print("\nMost Likely Scores:")
        for score, count in score_counter.most_common(3):
            percentage = (count / sims_per_matchup) * 100
            print(f"{score[0]}-{score[1]}: {percentage:.1f}%")
        
        # Calculate over/under probabilities
        print("\nOver/Under Probabilities:")
        for points in [140, 150, 160]:
            over_prob = sum(1 for s in total_scores if s > points) / len(total_scores) * 100
            print(f"Over {points}: {over_prob:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="March Madness 2026 tournament simulator")
    parser.add_argument("--num-simulations", type=int, default=1_000_000, help="Number of tournament simulations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N simulations (0 = off)")
    parser.add_argument("--no-scoring", action="store_true", help="Skip championship scoring analysis")
    parser.add_argument("--score-sims", type=int, default=1000, help="Simulations per matchup in scoring analysis")
    parser.add_argument("--sanity-checks", action="store_true", help="Print seed-level sanity diagnostics after simulations")
    parser.add_argument("--pool-ev", action="store_true", help="Print bracket pool EV: chalk vs contrarian (R64 proof-of-concept)")
    args = parser.parse_args()

    validate_bracket(bracket_data)

    rng = np.random.default_rng(args.seed)
    print("Running 2026 tournament simulations...")
    simulation_results = simulate_tournament(
        bracket_data,
        num_simulations=args.num_simulations,
        rng=rng,
        progress_every=args.progress_every,
    )
    print_results(simulation_results, args.num_simulations)
    print_seed_matchup_diagnostics(bracket_data)
    print_bracket_predictions(simulation_results, bracket_data)

    if args.sanity_checks:
        run_sanity_diagnostics(simulation_results, args.num_simulations, bracket_data)

    if args.pool_ev:
        pool_cfg = BracketPoolConfig()
        # Proof-of-concept: chalk vs one contrarian
        comp = compare_chalk_vs_contrarian(
            simulation_results, args.num_simulations, bracket_data, pool_config=pool_cfg
        )
        print("\n=== Bracket Pool EV (R64 proof-of-concept) ===")
        print(f"Chalk expected score: {comp['chalk_ev']:.2f}")
        print(f"Contrarian (1 upset) expected score: {comp['contrarian_ev']:.2f}")
        if comp.get("contrarian_game"):
            fav, underdog = comp["contrarian_game"]
            print(f"Contrarian game: {underdog} over {fav}")

        # Bracket variants: chalk, light, medium, high variance
        variants = compare_bracket_variants(
            simulation_results, args.num_simulations, bracket_data, pool_config=pool_cfg
        )
        print("\n=== Bracket Variants (R64 expected score) ===")
        for name, ev in variants.items():
            label = name.replace("_", " ").title()
            print(f"  {label}: {ev:.2f}")

        # Best single-upset swaps: by EV (least cost) and by leverage-adjusted (differentiation)
        all_swaps = rank_single_upset_swaps(
            simulation_results, args.num_simulations, bracket_data, pool_cfg, top_n=20
        )
        print("\n=== Best Single-Upset Swaps (by EV impact) ===")
        print("  Pick underdog for least EV cost (lev = leverage: P(underdog)×seed_diff×P(fav)):")
        for i, ((fav, underdog), ev_upset, ev_change, (hi, lo), p_underdog, p_fav, leverage) in enumerate(all_swaps[:8], 1):
            print(f"  {i}. {lo}v{hi}: {underdog} over {fav}  (EV={ev_upset:.2f}, Δ={ev_change:+.2f}, lev={leverage:.2f})")

        by_leverage = rank_upsets_by_leverage(all_swaps)
        print("\n=== Best Single-Upset Swaps (by leverage-adjusted) ===")
        print("  Pick underdog for best EV + differentiation (adj = Δ + 0.4×lev):")
        for i, ((fav, underdog), ev_upset, ev_change, (hi, lo), p_underdog, p_fav, leverage) in enumerate(by_leverage[:8], 1):
            adj = ev_change + LEVERAGE_WEIGHT * leverage
            print(f"  {i}. {lo}v{hi}: {underdog} over {fav}  (EV={ev_upset:.2f}, Δ={ev_change:+.2f}, lev={leverage:.2f}, adj={adj:+.2f})")

    if not args.no_scoring:
        # Matchup win probabilities for projected Elite Eight, Final Four, Championship
        print_matchup_win_probabilities(bracket_data)

        # Get Final Four teams (most frequent by region) and analyze scoring
        final_four_teams = []
        for region in ["South", "West", "East", "Midwest"]:
            teams_in_region = [
                (team, count)
                for team, count in simulation_results["final_four"].items()
                if team in [t[0] for t in bracket_data[region]]
            ]
            if teams_in_region:
                top_team = max(teams_in_region, key=lambda x: x[1])
                final_four_teams.append((top_team[0], None))
        analyze_championship_scoring(final_four_teams, rng, sims_per_matchup=args.score_sims)