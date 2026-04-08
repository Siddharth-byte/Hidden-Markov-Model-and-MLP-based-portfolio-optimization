# Hidden-Markov-Model-and-MLP-based-portfolio-optimization


This project implements a sophisticated quantitative finance pipeline that combines unsupervised learning (Hidden Markov Models) with deep learning (Multi-Layer Perceptrons) to outperform traditional static allocation strategies. By decoding latent market "regimes" (Bull, Bear, and Sideways), the system dynamically reweights a multi-asset portfolio to maximize risk-adjusted returns, achieving a 1.54 Sharpe Ratio in walk-forward testing.

#Technical Architecture
The strategy operates through a two-stage hierarchical model:

Regime Detection (The "HMM" Layer): * Utilizes a Gaussian Hidden Markov Model to analyze rolling market returns and volatility.

Unlike simple moving averages, the HMM identifies the underlying "latent state" of the market, effectively filtering noise from actionable structural shifts.

Asset Allocation (The "MLP" Layer): * A Multi-Layer Perceptron (Neural Network) acts as the decision engine.

It is trained as a classifier to predict the "winning" asset for the subsequent period based on the detected regime.

The model outputs a Softmax probability distribution, which serves as the optimal portfolio weights for the given market state.

#Key Features
Adaptive Intelligence: The model doesn't just follow a trend; it "reads the room." It automatically shifts from aggressive growth (e.g., NVDA, QQQ) to defensive stores of value (e.g., GLD, TLT) as market regimes flip.

Walk-Forward Validation: Employs a rigorous TimeSeriesSplit methodology to ensure the model generalizes to unseen data, preventing the "look-ahead bias" common in many trading backtests.

Interactive Streamlit Dashboard: A production-ready web interface allowing users to input custom tickers, adjust date ranges, and visualize live performance metrics against an Equal-Weight (EW) benchmark.

High-Conviction Weighting: Features an amplified Softmax function to ensure the portfolio takes meaningful positions in predicted outperformers rather than defaulting to over-diversification.
