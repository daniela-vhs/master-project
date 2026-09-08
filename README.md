# PnL Explain for Interest Rate Options: Hull-White vs. Forward Market Model

MSc thesis, Quantitative Finance & Risk Management (MAFINRISK), Università Bocconi, 2026.

## What this does

Compares two interest rate models — the one-factor Hull-White short-rate model and the
Forward Market Model — on their ability to explain the daily PnL of a portfolio of interest
rate options (Caps). The question is not which model prices better in isolation, but which one
produces Greeks that actually reconcile realised PnL against predicted PnL: the test a
front-office or model validation team applies before trusting a model in production.

## Method

1. **Calibration.** Both models are calibrated to ATM cap volatility surface.
2. **Greeks.** Delta, vega, gamma, volga, theta, vanna.
3. **PnL attribution.** Daily PnL is decomposed into delta, vega, gamma, theta, volga, vanna, and an unexplained residual.
4. **Comparison.** Models are ranked on the size and stability of the unexplained residual over 3 years.

## Repository structure

```
src/          # model implementations and calibration routines
notebooks/    # analysis and figures
data/         # market data inputs — see note below
tests/        # unit tests
```

## Running it

```bash
pip install -r requirements.txt
python src/[entry point].py
```

## Data

Bloomberg.

## Author

Daniela Hidalgo Soto — [linkedin.com/in/daniela-hidalgo-soto](https://linkedin.com/in/daniela-hidalgo-soto)