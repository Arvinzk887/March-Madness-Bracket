import argparse
import numpy as np
from collections import defaultdict

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

def _iter_bracket_teams(bracket: dict) -> list[tuple[str, int]]:
    teams: list[tuple[str, int]] = []
    for region_matchups in bracket.values():
        for team, seed in region_matchups:
            teams.append((team, seed))
    return teams

def _ensure_team_factors(bracket: dict) -> dict[str, list[float]]:
    """
    Returns per-team factors:
      [overall, offense_eff, defense_eff, experience, coach]
    Ensures all teams in the bracket have entries, using `team_ratings` + seed defaults.
    """
    factors: dict[str, list[float]] = {}

    for team, seed in _iter_bracket_teams(bracket):
        # If this is a play-in placeholder (e.g., "A/B"), ensure *both* A and B get stats.
        teams_to_add = [team]
        if _is_play_in_team(team):
            teams_to_add = list(_split_play_in(team))

        for t in teams_to_add:
            base_overall = float(team_ratings.get(t, _seed_based_overall(seed)))
            offense = 98.0 + (base_overall - 50.0) * 0.55
            defense = 110.0 - (base_overall - 50.0) * 0.45  # lower is better
            experience = 6.0 + (base_overall - 70.0) / 20.0
            coach = 6.0 + (base_overall - 70.0) / 25.0

            factors[t] = [
                base_overall,
                float(np.clip(offense, 95.0, 125.0)),
                float(np.clip(defense, 85.0, 115.0)),
                float(np.clip(experience, 4.0, 9.5)),
                float(np.clip(coach, 4.0, 9.5)),
            ]

    return factors

team_stats = _ensure_team_factors(bracket_data)

def calculate_win_probability(team1, team2):
    """Calculate win probability using multiple factors"""
    stats1 = team_stats[team1]
    stats2 = team_stats[team2]
    
    # Weights for different factors
    weights = {
        'overall': 0.4,
        'offense': 0.2,
        'defense': 0.2,
        'experience': 0.1,
        'coach': 0.1
    }
    
    # Calculate weighted score for each team
    score1 = (
        weights['overall'] * stats1[0] +
        weights['offense'] * (stats1[1]/120) * 100 +
        weights['defense'] * (1 - stats1[2]/120) * 100 +
        weights['experience'] * stats1[3] * 10 +
        weights['coach'] * stats1[4] * 10
    )
    
    score2 = (
        weights['overall'] * stats2[0] +
        weights['offense'] * (stats2[1]/120) * 100 +
        weights['defense'] * (1 - stats2[2]/120) * 100 +
        weights['experience'] * stats2[3] * 10 +
        weights['coach'] * stats2[4] * 10
    )
    
    # Calculate win probability using logistic function
    diff = score1 - score2
    return 1 / (1 + np.exp(-diff / 15))

def simulate_game(team1, team2, rng: np.random.Generator):
    """Simulate a single game between two teams."""
    prob_team1 = calculate_win_probability(team1, team2)
    return team1 if rng.random() < prob_team1 else team2

def simulate_round(matchups, rng: np.random.Generator):
    """Simulate one round of games."""
    winners = []
    for i in range(0, len(matchups), 2):
        team1, seed1 = matchups[i]
        team2, seed2 = matchups[i + 1]
        winner = simulate_game(team1, team2, rng)
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

def resolve_first_four(bracket: dict, rng: np.random.Generator) -> dict:
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
                winner = simulate_game(a, b, rng)
                new_matchups.append((winner, seed))
            else:
                new_matchups.append((team, seed))
        resolved[region] = new_matchups
    return resolved

def simulate_tournament(bracket_data, num_simulations=1000000, *, rng: np.random.Generator, progress_every: int = 0):
    """Simulate the entire tournament multiple times."""
    results = {
        'championship_wins': defaultdict(int),
        'final_four': defaultdict(int),
        'elite_eight': defaultdict(int),
        'sweet_sixteen': defaultdict(int),
        'round_32': defaultdict(int)
    }
    
    for sim in range(num_simulations):
        if progress_every and sim % progress_every == 0:
            print(f"Running simulation {sim}/{num_simulations}")
            
        current_bracket = resolve_first_four(bracket_data, rng)
        region_winners: dict[str, tuple[str, int]] = {}
        
        for region in current_bracket:
            # Round of 32
            current_bracket[region] = simulate_round(current_bracket[region], rng)
            for team, _ in current_bracket[region]:
                results['round_32'][team] += 1
            
            # Sweet 16
            current_bracket[region] = simulate_round(current_bracket[region], rng)
            for team, _ in current_bracket[region]:
                results['sweet_sixteen'][team] += 1
            
            # Elite 8
            current_bracket[region] = simulate_round(current_bracket[region], rng)
            for team, _ in current_bracket[region]:
                results['elite_eight'][team] += 1
            
            # Final 4
            regional_winner = simulate_round(current_bracket[region], rng)[0]
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
        championship_game = simulate_round(final_four, rng)
        champion = simulate_round(championship_game, rng)[0][0]
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

def print_bracket_predictions(results, bracket_data):
    """
    Print a visual representation of the predicted bracket based on simulation results.
    """
    def get_most_likely_winner(team1, team2, round_results):
        """Helper function to determine the most likely winner between two teams"""
        prob1 = round_results.get(team1[0], 0)
        prob2 = round_results.get(team2[0], 0)
        return team1 if prob1 > prob2 else team2

    print("\n=== 2026 NCAA Tournament Predictions ===\n")

    # Store predictions for each round
    predictions = {
        'round_32': {},
        'sweet_16': {},
        'elite_8': {},
    }

    # Print each region
    for region in ['South', 'West', 'East', 'Midwest']:
        print(f"\n{region} Region:")
        print("=" * 50)
        
        # First Round matchups (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
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
        # Top bracket game
        top_winner = get_most_likely_winner(
            predictions['sweet_16'][region + '0'],
            predictions['sweet_16'][region + '1'],
            results['elite_eight']
        )
        # Bottom bracket game
        bottom_winner = get_most_likely_winner(
            predictions['sweet_16'][region + '2'],
            predictions['sweet_16'][region + '3'],
            results['elite_eight']
        )
        
        # Elite 8 game for the region
        winner = get_most_likely_winner(top_winner, bottom_winner, results['final_four'])
        predictions['elite_8'][region] = winner
        print(f"Regional Final: ({top_winner[1]}) {top_winner[0]} vs ({bottom_winner[1]}) {bottom_winner[0]} → ({winner[1]}) {winner[0]}")

    # Final Four
    print("\n=== Final Four ===")
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
    
    # Championship
    print("\nChampionship Game:")
    champion = get_most_likely_winner(east_south, west_midwest, results['championship_wins'])
    print(f"({east_south[1]}) {east_south[0]} vs ({west_midwest[1]}) {west_midwest[0]} → ({champion[1]}) {champion[0]}")
    
    print("\n=== 2026 Champion ===")
    print(f"({champion[1]}) {champion[0]}")

def _ensure_team_tempo_stats(bracket: dict) -> dict[str, dict[str, float]]:
    """
    Build a tempo-based stats dict for score predictions:
      {team: {offense, defense, tempo}}
    Ensures all bracket teams are present, derived from `team_stats`.
    """
    tempo_stats: dict[str, dict[str, float]] = {}
    for team, seed in _iter_bracket_teams(bracket):
        # For play-in placeholders, store stats for the underlying teams too.
        teams_to_add = [team]
        if _is_play_in_team(team):
            teams_to_add = list(_split_play_in(team))

        for t in teams_to_add:
            overall, off_eff, def_eff, _, _ = team_stats[t]
            tempo = 66.0 + (overall - 70.0) * 0.12 + (8 - seed) * 0.04
            tempo_stats[t] = {
                "offense": float(np.clip(off_eff, 98.0, 126.0)),
                "defense": float(np.clip(def_eff, 85.0, 112.0)),
                "tempo": float(np.clip(tempo, 62.0, 72.0)),
            }
    return tempo_stats

team_tempo_stats = _ensure_team_tempo_stats(bracket_data)

def predict_game_score(team1, team2, rng: np.random.Generator):
    """Predict the score for a game between two teams"""
    if team1 not in team_tempo_stats or team2 not in team_tempo_stats:
        raise ValueError(f"Missing stats for {team1} or {team2}")
    
    # Average the tempo of both teams to estimate possessions
    possessions = (team_tempo_stats[team1]['tempo'] + team_tempo_stats[team2]['tempo']) / 2
    
    # Calculate points per possession
    team1_ppp = team_tempo_stats[team1]['offense'] / 100
    team2_ppp = team_tempo_stats[team2]['offense'] / 100
    
    # Adjust for opponent's defense
    team1_ppp *= (100 / team_tempo_stats[team2]['defense'])
    team2_ppp *= (100 / team_tempo_stats[team1]['defense'])
    
    # Calculate base scores
    team1_base = team1_ppp * possessions
    team2_base = team2_ppp * possessions
    
    # Add random variation
    team1_score = round(team1_base + rng.normal(0, 3.5))
    team2_score = round(team2_base + rng.normal(0, 3.5))
    
    # Ensure reasonable scores
    team1_score = max(40, min(105, team1_score))
    team2_score = max(40, min(105, team2_score))
    
    return team1_score, team2_score

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
    print_bracket_predictions(simulation_results, bracket_data)

    if not args.no_scoring:
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