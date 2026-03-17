# March Madness Tournament Simulator (2026)

A Python-based simulator that predicts NCAA March Madness outcomes using statistical modeling, Monte Carlo simulation, and bracket pool expected-value analysis.

## Features

- **Statistical modeling**: Team ratings, offensive/defensive efficiency, tempo, experience, and coaching factors
- **Monte Carlo simulation**: Configurable simulations (default 1M) for advancement probabilities
- **Win probability model**: Logistic model with tunable chalk/upset calibration; seed matchup diagnostics
- **Score prediction**: PPP-based model for game totals (calibrated for neutral-court tournament games)
- **Bracket visualization**: Coherent and marginal-probability brackets
- **Championship analysis**: Win probabilities and scoring outcomes for Elite Eight, Final Four, and title game
- **Bracket pool EV**: Compare chalk vs contrarian brackets; rank best single-upset swaps; light/medium/high-variance variants
- **Backtest foundation**: Structure for historical calibration (bracket + ratings + results)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/March-Madness-Bracket.git
cd March-Madness-Bracket
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulator (2026 field):
```bash
python simulator.py
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-simulations` | 1,000,000 | Number of tournament simulations |
| `--seed` | None | Random seed for reproducibility |
| `--progress-every` | 0 | Print progress every N simulations (0 = off) |
| `--no-scoring` | False | Skip championship scoring analysis |
| `--score-sims` | 1000 | Simulations per matchup in scoring analysis |
| `--sanity-checks` | False | Print seed-level diagnostics after simulations |
| `--pool-ev` | False | Print bracket pool EV: chalk vs contrarian, variants, best upset swaps |

### Example commands

**Faster run while iterating:**
```bash
python simulator.py --num-simulations 20000 --seed 123 --no-scoring
```

**Bracket pool strategy (best upset picks):**
```bash
python simulator.py --num-simulations 50000 --pool-ev --no-scoring
```

**Full analysis with sanity checks:**
```bash
python simulator.py --num-simulations 200000 --sanity-checks --score-sims 500
```

## How It Works

1. **Team statistics**: Each team is rated on overall strength, offensive/defensive efficiency, tempo, experience, and coaching. Optional overrides (e.g. KenPom-style ratings) can be supplied.

2. **Win probability**: Logistic model with composite rating and optional seed nudge. Tunable via `LOGISTIC_SCALE`, `SEED_EDGE_STRENGTH`, and `WIN_WEIGHTS`.

3. **Game simulation**: Each game sampled from win probability; scores from PPP × possessions with bounded noise.

4. **Tournament simulation**: Full bracket resolution per run; aggregates advancement counts across simulations.

5. **Bracket pool EV**: R64 expected scores for chalk vs contrarian picks; ranks single-upset swaps by EV impact; compares chalk, light, medium, and high-variance variants.

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Improve Statistical Models**: Enhance the team rating system
2. **Add Historical Data**: Incorporate historical tournament results
3. **Create Visualizations**: Add data visualization capabilities
4. **Optimize Performance**: Improve simulation speed
5. **Add Features**: Implement new analysis tools

To contribute:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NCAA for tournament data
- KenPom for statistical inspiration
- All contributors who help improve this project

## Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/arvindha/) to discuss this project or potential collaborations!
