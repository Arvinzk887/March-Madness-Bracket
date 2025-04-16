import random
import numpy as np
from collections import defaultdict

# 2025 NCAA Tournament bracket data
bracket_data = {
    "South": [
        ("Auburn", 1), ("Alabama State/St. Francis (PA)", 16),
        ("Louisville", 8), ("Creighton", 9),
        ("Michigan", 5), ("UC San Diego", 12),
        ("Texas A&M", 4), ("Yale", 13),
        ("Ole Miss", 6), ("San Diego State/UNC", 11),
        ("Iowa State", 3), ("Lipscomb", 14),
        ("Marquette", 7), ("New Mexico", 10),
        ("Michigan State", 2), ("Bryant", 15)
        
    ],
    "West": [
        ("Florida", 1), ("Norfolk State", 16),
        ("UConn", 8), ("Oklahoma", 9),
        ("Memphis", 5), ("Colorado State", 12),
        ("Maryland", 4), ("Grand Canyon", 13),
        ("Missouri", 6), ("Drake", 11),
        ("Texas Tech", 3), ("UNC Wilmington", 14),
         ("Kansas", 7), ("Arkansas", 10),
        ("St. John's", 2), ("Omaha", 15)
     
    ],
    "East": [
        ("Duke", 1), ("American/Mount St. Mary's", 16),
        ("Mississippi State", 8), ("Baylor", 9),
        ("Oregon", 5), ("Liberty", 12),
        ("Arizona", 4), ("Akron", 13),
        ("BYU", 6), ("VCU", 11),
        ("Wisconsin", 3), ("Montana", 14),
        ("Saint Mary's", 7), ("Vanderbilt", 10),
        ("Alabama", 2), ("Robert Morris", 15)

    ],
    "Midwest": [
        ("Houston", 1), ("SIU Edwardsville", 16),
        ("Gonzaga", 8), ("Georgia", 9),
        ("Clemson", 5), ("McNeese", 12),
        ("Purdue", 4), ("High Point", 13),
        ("Illinois", 6), ("Texas/Xavier", 11),
        ("Kentucky", 3), ("Troy", 14),
        ("UCLA", 7), ("Utah State", 10),
        ("Tennessee", 2), ("Wofford", 15) 

    ]
    
}

# Original team ratings
team_ratings = {
    # South Region
    "Auburn": 87.7, "Alabama State/St. Francis (PA)": 45.2,
    "Michigan State": 82.1, "Bryant": 51.3,
    "Iowa State": 89.2, "Lipscomb": 53.8,
    "Texas A&M": 81.4, "Yale": 67.2,
    "Michigan": 78.9, "UC San Diego": 58.4,
    "Ole Miss": 80.3, "San Diego State/UNC": 82.5,
    "Marquette": 85.6, "New Mexico": 79.8,
    "Louisville": 76.2, "Creighton": 86.9,
    
    # East Region
    "Duke": 86.8, "American/Mount St. Mary's": 46.1,
    "Alabama": 88.4, "Robert Morris": 50.7,
    "Wisconsin": 83.5, "Montana": 54.2,
    "Arizona": 89.7, "Akron": 65.8,
    "Oregon": 82.3, "Liberty": 63.4,
    "BYU": 87.2, "VCU": 77.3,
    "Saint Mary's": 84.1, "Vanderbilt": 75.6,
    "Mississippi State": 81.8, "Baylor": 83.2,
    
    # Midwest Region
    "Houston": 93.8, "SIU Edwardsville": 44.9,
    "Tennessee": 90.2, "Wofford": 52.1,
    "Kentucky": 87.9, "Troy": 55.3,
    "Purdue": 92.1, "High Point": 54.7,
    "Clemson": 84.3, "McNeese": 71.2,
    "Illinois": 86.7, "Texas/Xavier": 81.9,
    "UCLA": 82.4, "Utah State": 80.1,
    "Gonzaga": 85.3, "Georgia": 77.8,
    
    # West Region
    "Florida": 85.9, "Norfolk State": 45.8,
    "St. John's": 83.7, "Omaha": 49.9,
    "Texas Tech": 84.8, "UNC Wilmington": 66.3,
    "Maryland": 79.6, "Grand Canyon": 68.4,
    "Memphis": 83.9, "Colorado State": 78.5,
    "Missouri": 77.4, "Drake": 79.2,
    "Kansas": 85.7, "Arkansas": 78.3,
    "UConn": 91.4, "Oklahoma": 83.6,
}

# Enhanced team statistics
team_stats = {
    # South Region
    "Auburn": [87.7, 115.2, 95.3, 7.8, 8.5],  # Strong offense, Bruce Pearl coaching
    "Alabama State/St. Francis (PA)": [45.2, 98.1, 108.4, 5.0, 4.0],  # Lower division stats
    "Michigan State": [82.1, 111.3, 97.8, 7.5, 9.5],  # Tom Izzo coaching boost
    "Bryant": [51.3, 101.2, 105.6, 4.5, 4.5],
    "Iowa State": [89.2, 114.5, 94.2, 7.8, 8.0],  # Strong defensive team
    "Lipscomb": [53.8, 102.3, 104.8, 5.0, 4.8],
    "Texas A&M": [81.4, 110.8, 98.2, 7.2, 7.8],
    "Yale": [67.2, 106.5, 101.2, 6.5, 6.0],  # Solid fundamentals
    "Michigan": [78.9, 109.7, 99.1, 6.8, 7.5],
    "UC San Diego": [58.4, 103.4, 103.9, 5.2, 5.0],
    "Ole Miss": [80.3, 110.2, 98.7, 7.0, 7.2],
    "San Diego State/UNC": [82.5, 111.5, 97.6, 7.3, 8.0],  # Tournament experience
    "Marquette": [85.6, 113.8, 96.2, 7.6, 8.2],  # Strong offensive team
    "New Mexico": [79.8, 109.9, 98.9, 6.9, 7.0],
    "Louisville": [76.2, 108.4, 100.1, 6.5, 7.0],
    "Creighton": [86.9, 114.2, 95.8, 7.7, 8.3],  # Efficient offense

    # East Region
    "Duke": [86.8, 114.0, 96.0, 7.8, 8.8],  # Traditional power
    "American/Mount St. Mary's": [46.1, 98.5, 107.8, 4.8, 4.2],
    "Alabama": [88.4, 115.8, 95.5, 7.6, 8.4],  # Fast-paced offense
    "Robert Morris": [50.7, 100.5, 106.2, 4.7, 4.5],
    "Wisconsin": [83.5, 111.2, 96.8, 7.4, 8.2],  # Strong fundamentals
    "Montana": [54.2, 102.8, 104.5, 5.1, 4.9],
    "Arizona": [89.7, 116.2, 94.8, 7.9, 8.6],  # Elite offense
    "Akron": [65.8, 105.8, 102.5, 5.8, 5.5],
    "Oregon": [82.3, 111.8, 97.2, 7.2, 8.0],
    "Liberty": [63.4, 104.8, 103.2, 5.6, 5.4],
    "BYU": [87.2, 114.5, 95.6, 7.5, 8.1],  # Strong shooting team
    "VCU": [77.3, 108.8, 99.5, 6.8, 7.2],  # Pressure defense
    "Saint Mary's": [84.1, 112.5, 96.5, 7.3, 8.0],
    "Vanderbilt": [75.6, 108.2, 100.2, 6.6, 7.0],
    "Mississippi State": [81.8, 110.5, 97.8, 7.1, 7.6],
    "Baylor": [83.2, 112.2, 96.9, 7.4, 8.3],  # Scott Drew coaching

    # Midwest Region
    "Houston": [93.8, 117.5, 92.1, 8.5, 9.0],  # Elite defense, Kelvin Sampson
    "SIU Edwardsville": [44.9, 97.8, 108.6, 4.6, 4.0],
    "Tennessee": [90.2, 115.6, 93.8, 8.0, 8.7],  # Strong defense
    "Wofford": [52.1, 101.8, 105.2, 4.9, 4.7],
    "Kentucky": [87.9, 115.4, 95.2, 7.7, 9.2],  # Calipari coaching boost
    "Troy": [55.3, 103.2, 104.2, 5.2, 5.0],
    "Purdue": [92.1, 116.8, 93.0, 8.3, 8.5],  # Zach Edey effect
    "High Point": [54.7, 102.5, 104.8, 5.0, 4.8],
    "Clemson": [84.3, 112.8, 96.4, 7.4, 7.8],
    "McNeese": [71.2, 107.2, 101.5, 6.2, 6.0],
    "Illinois": [86.7, 114.2, 95.5, 7.6, 8.1],
    "Texas/Xavier": [81.9, 111.2, 97.5, 7.2, 7.8],
    "UCLA": [82.4, 111.5, 97.2, 7.3, 8.4],  # Mick Cronin coaching
    "Utah State": [80.1, 110.2, 98.4, 7.0, 7.4],
    "Gonzaga": [85.3, 113.8, 96.2, 7.5, 8.8],  # Few coaching boost
    "Georgia": [77.8, 109.2, 99.2, 6.8, 7.2],

    # West Region
    "Florida": [85.9, 113.5, 96.0, 7.6, 8.2],
    "Norfolk State": [45.8, 98.2, 108.2, 4.7, 4.2],
    "St. John's": [83.7, 112.4, 96.8, 7.3, 7.8],
    "Omaha": [49.9, 100.2, 106.5, 4.6, 4.4],
    "Texas Tech": [84.8, 112.8, 96.2, 7.4, 8.0],
    "UNC Wilmington": [66.3, 105.5, 102.8, 5.8, 5.6],
    "Maryland": [79.6, 109.8, 98.8, 7.0, 7.6],
    "Grand Canyon": [68.4, 106.8, 102.0, 6.0, 5.8],
    "Memphis": [83.9, 112.5, 96.5, 7.3, 7.8],
    "Colorado State": [78.5, 109.5, 99.0, 6.9, 7.2],
    "Missouri": [77.4, 109.0, 99.5, 6.8, 7.0],
    "Drake": [79.2, 109.8, 98.6, 6.9, 7.2],
    "Kansas": [85.7, 113.8, 96.1, 7.5, 9.0],  # Bill Self coaching boost
    "Arkansas": [78.3, 109.4, 99.2, 6.8, 7.4],
    "UConn": [91.4, 116.5, 93.2, 8.2, 8.7],  # Defending champs boost
    "Oklahoma": [83.6, 112.2, 96.8, 7.3, 7.8]
}

for team in team_ratings.keys():
    if team not in team_stats:
        base_rating = team_ratings[team]
        team_stats[team] = [
            base_rating,
            base_rating + 20,  # Offensive rating
            120 - base_rating,  # Defensive rating (lower is better)
            6.0,  # Default experience rating
            6.0   # Default coach rating
        ]

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
    return 1 / (1 + np.exp(-diff/15))

def simulate_game(team1, team2):
    """Simulate a single game between two teams."""
    prob_team1 = calculate_win_probability(team1, team2)
    return team1 if random.random() < prob_team1 else team2

def simulate_round(matchups):
    """Simulate one round of games."""
    winners = []
    for i in range(0, len(matchups), 2):
        team1, seed1 = matchups[i]
        team2, seed2 = matchups[i + 1]
        winner = simulate_game(team1, team2)
        winner_seed = seed1 if winner == team1 else seed2
        winners.append((winner, winner_seed))
    return winners

def simulate_tournament(bracket_data, num_simulations=1000000): #Running 1,000,000 simulations
    """Simulate the entire tournament multiple times."""
    results = {
        'championship_wins': defaultdict(int),
        'final_four': defaultdict(int),
        'elite_eight': defaultdict(int),
        'sweet_sixteen': defaultdict(int),
        'round_32': defaultdict(int)
    }
    
    for sim in range(num_simulations):
        if sim % 100 == 0:  # Progress indicator
            print(f"Running simulation {sim}/{num_simulations}")
            
        current_bracket = {region: matchups[:] for region, matchups in bracket_data.items()}
        final_four = []
        
        for region in current_bracket:
            # Round of 32
            current_bracket[region] = simulate_round(current_bracket[region])
            for team, _ in current_bracket[region]:
                results['round_32'][team] += 1
            
            # Sweet 16
            current_bracket[region] = simulate_round(current_bracket[region])
            for team, _ in current_bracket[region]:
                results['sweet_sixteen'][team] += 1
            
            # Elite 8
            current_bracket[region] = simulate_round(current_bracket[region])
            for team, _ in current_bracket[region]:
                results['elite_eight'][team] += 1
            
            # Final 4
            regional_winner = simulate_round(current_bracket[region])[0]
            final_four.append(regional_winner)
            results['final_four'][regional_winner[0]] += 1
        
        # Championship
        championship_game = simulate_round(final_four)
        champion = simulate_round(championship_game)[0][0]
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

# Run the simulation
print("Running 2025 tournament simulations...")
simulation_results = simulate_tournament(bracket_data, num_simulations=1000000)
print_results(simulation_results, 1000000)

def print_bracket_predictions(results, bracket_data):
    """
    Print a visual representation of the predicted bracket based on simulation results.
    """
    def get_most_likely_winner(team1, team2, round_results):
        """Helper function to determine the most likely winner between two teams"""
        prob1 = round_results.get(team1[0], 0)
        prob2 = round_results.get(team2[0], 0)
        return team1 if prob1 > prob2 else team2

    print("\n=== 2025 NCAA Tournament Predictions ===\n")

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
    south_west = get_most_likely_winner(predictions['elite_8']['South'], predictions['elite_8']['West'], results['final_four'])
    east_midwest = get_most_likely_winner(predictions['elite_8']['East'], predictions['elite_8']['Midwest'], results['final_four'])
    
    print(f"South/West: ({predictions['elite_8']['South'][1]}) {predictions['elite_8']['South'][0]} vs " + 
          f"({predictions['elite_8']['West'][1]}) {predictions['elite_8']['West'][0]} → ({south_west[1]}) {south_west[0]}")
    print(f"East/Midwest: ({predictions['elite_8']['East'][1]}) {predictions['elite_8']['East'][0]} vs " + 
          f"({predictions['elite_8']['Midwest'][1]}) {predictions['elite_8']['Midwest'][0]} → ({east_midwest[1]}) {east_midwest[0]}")
    
    # Championship
    print("\nChampionship Game:")
    champion = get_most_likely_winner(south_west, east_midwest, results['championship_wins'])
    print(f"({south_west[1]}) {south_west[0]} vs ({east_midwest[1]}) {east_midwest[0]} → ({champion[1]}) {champion[0]}")
    
    print("\n=== 2025 Champion ===")
    print(f"({champion[1]}) {champion[0]}")

# Run the simulation
print("Running 2025 tournament simulations...")
simulation_results = simulate_tournament(bracket_data, num_simulations=1000000)
print_results(simulation_results, 1000000)
print_bracket_predictions(simulation_results, bracket_data)
# Team statistics dictionary (offensive efficiency, defensive efficiency, tempo)
team_stats = {
    'UConn': {'offense': 124.5, 'defense': 94.2, 'tempo': 67.5},
    'Houston': {'offense': 116.8, 'defense': 87.3, 'tempo': 65.8},
    'Purdue': {'offense': 121.7, 'defense': 94.8, 'tempo': 66.2},
    'Tennessee': {'offense': 115.8, 'defense': 90.5, 'tempo': 67.1},
    'Arizona': {'offense': 120.3, 'defense': 96.5, 'tempo': 71.4},
    'Marquette': {'offense': 118.9, 'defense': 95.8, 'tempo': 68.9},
    'North Carolina': {'offense': 119.2, 'defense': 96.7, 'tempo': 69.8},
    'Iowa State': {'offense': 114.5, 'defense': 91.2, 'tempo': 66.4},
    'Duke': {'offense': 117.8, 'defense': 95.6, 'tempo': 67.8},
    'Illinois': {'offense': 117.2, 'defense': 97.8, 'tempo': 69.5},
    'Kentucky': {'offense': 121.5, 'defense': 99.2, 'tempo': 70.2},
    'Auburn': {'offense': 118.2, 'defense': 92.1, 'tempo': 69.8},
    'Creighton': {'offense': 116.9, 'defense': 93.4, 'tempo': 67.2},
    'Baylor': {'offense': 117.5, 'defense': 98.2, 'tempo': 68.7},
    'Kansas': {'offense': 115.8, 'defense': 96.8, 'tempo': 67.9},
    'Gonzaga': {'offense': 119.4, 'defense': 97.5, 'tempo': 70.1},
    'Texas': {'offense': 115.2, 'defense': 95.8, 'tempo': 68.4},
    'Alabama': {'offense': 118.8, 'defense': 98.4, 'tempo': 71.8},
    'Wisconsin': {'offense': 112.5, 'defense': 95.2, 'tempo': 64.8},
    'San Diego St': {'offense': 111.8, 'defense': 94.1, 'tempo': 65.2},
    'Washington St': {'offense': 113.5, 'defense': 95.8, 'tempo': 66.4},
    'Florida': {'offense': 115.8, 'defense': 97.2, 'tempo': 69.5},
    'Texas A&M': {'offense': 113.2, 'defense': 96.8, 'tempo': 68.7},
    'Colorado': {'offense': 114.5, 'defense': 97.5, 'tempo': 68.2},
    'Dayton': {'offense': 113.8, 'defense': 95.4, 'tempo': 65.8},
    'Nevada': {'offense': 112.5, 'defense': 96.2, 'tempo': 66.5},
    'Texas Tech': {'offense': 112.8, 'defense': 96.5, 'tempo': 67.8},
    'FAU': {'offense': 112.2, 'defense': 96.8, 'tempo': 67.2},
    'Nebraska': {'offense': 113.5, 'defense': 97.2, 'tempo': 68.4},
    'Northwestern': {'offense': 111.8, 'defense': 96.5, 'tempo': 65.8},
    'Mississippi St': {'offense': 111.5, 'defense': 95.8, 'tempo': 66.2},
    'Utah St': {'offense': 114.2, 'defense': 98.5, 'tempo': 68.5},
    'New Mexico': {'offense': 114.8, 'defense': 98.2, 'tempo': 69.2},
    'Michigan St': {'offense': 112.5, 'defense': 97.5, 'tempo': 67.2},
    'St. Mary\'s': {'offense': 113.2, 'defense': 94.8, 'tempo': 64.5},
    'BYU': {'offense': 115.5, 'defense': 97.2, 'tempo': 68.8},
    'Drake': {'offense': 111.8, 'defense': 96.5, 'tempo': 65.8},
    'Grand Canyon': {'offense': 110.5, 'defense': 97.2, 'tempo': 66.4},
    'South Carolina': {'offense': 111.2, 'defense': 97.8, 'tempo': 67.5},
    'Oregon': {'offense': 113.5, 'defense': 98.4, 'tempo': 68.2},
    'Vermont': {'offense': 110.2, 'defense': 98.5, 'tempo': 64.8},
    'Yale': {'offense': 109.8, 'defense': 98.2, 'tempo': 65.2},
    'Samford': {'offense': 111.5, 'defense': 99.5, 'tempo': 70.5},
    'Charleston': {'offense': 110.8, 'defense': 98.8, 'tempo': 69.2},
    'Colgate': {'offense': 109.5, 'defense': 99.2, 'tempo': 65.8},
    'UAB': {'offense': 111.2, 'defense': 99.5, 'tempo': 68.5},
    'Western Ky': {'offense': 110.5, 'defense': 99.8, 'tempo': 67.8},
    'Montana St': {'offense': 109.2, 'defense': 99.5, 'tempo': 66.5},
    'Longwood': {'offense': 108.8, 'defense': 99.8, 'tempo': 66.2},
    'Morehead St': {'offense': 108.5, 'defense': 100.2, 'tempo': 65.8},
    'Howard': {'offense': 108.2, 'defense': 100.5, 'tempo': 67.2},
    'Wagner': {'offense': 107.8, 'defense': 100.8, 'tempo': 64.5},
    'Stetson': {'offense': 108.5, 'defense': 101.2, 'tempo': 66.8},
    'Grambling': {'offense': 107.5, 'defense': 101.5, 'tempo': 65.5}
}

def predict_game_score(team1, team2):
    """Predict the score for a game between two teams"""
    if team1 not in team_stats or team2 not in team_stats:
        raise ValueError(f"Missing stats for {team1} or {team2}")
    
    # Average the tempo of both teams to estimate possessions
    possessions = (team_stats[team1]['tempo'] + team_stats[team2]['tempo']) / 2
    
    # Calculate points per possession
    team1_ppp = team_stats[team1]['offense'] / 100
    team2_ppp = team_stats[team2]['offense'] / 100
    
    # Adjust for opponent's defense
    team1_ppp *= (100 / team_stats[team2]['defense'])
    team2_ppp *= (100 / team_stats[team1]['defense'])
    
    # Calculate base scores
    team1_base = team1_ppp * possessions
    team2_base = team2_ppp * possessions
    
    # Add random variation
    team1_score = round(team1_base + np.random.normal(0, 4))
    team2_score = round(team2_base + np.random.normal(0, 4))
    
    # Ensure reasonable scores
    team1_score = max(55, min(95, team1_score))
    team2_score = max(55, min(95, team2_score))
    
    return team1_score, team2_score

def analyze_championship_scoring(final_four_teams):
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
        for _ in range(1000):  # Simulate each matchup 1000 times
            score1, score2 = predict_game_score(team1, team2)
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
            percentage = (count/1000) * 100
            print(f"{score[0]}-{score[1]}: {percentage:.1f}%")
        
        # Calculate over/under probabilities
        print("\nOver/Under Probabilities:")
        for points in [140, 150, 160]:
            over_prob = sum(1 for s in total_scores if s > points) / len(total_scores) * 100
            print(f"Over {points}: {over_prob:.1f}%")

# Get Final Four teams from the simulation results
final_four_teams = []
for region in ['South', 'West', 'East', 'Midwest']:
    teams_in_region = [(team, count) for team, count in simulation_results['final_four'].items() 
                       if team in [t[0] for t in bracket_data[region]]]
    if teams_in_region:
        top_team = max(teams_in_region, key=lambda x: x[1])
        final_four_teams.append((top_team[0], None))

# Run the scoring analysis
analyze_championship_scoring(final_four_teams)