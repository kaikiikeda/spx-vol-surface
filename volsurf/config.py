"""Snapshot setup: SPX close, Friday 13 February 2026.

All market inputs here are illustrative. The option chain is generated from a
hidden "true" Heston model so every later stage (SVI fit, VIX replication,
Heston calibration) can be checked against a known answer.
"""
from datetime import datetime

SNAPSHOT = datetime(2026, 2, 13, 16, 0)   # 4:00 pm ET close
SPOT = 6836.17                            # illustrative SPX level
R = 0.0365                                # flat risk-free rate
Q = 0.0118                                # flat dividend yield

# Hidden data-generating model (calibration must recover this)
TRUE_HESTON = dict(v0=0.0262, kappa=2.10, theta=0.0405, sigma=0.78, rho=-0.71)

# (settlement time, settlement style, label)
# Standard monthlies settle AM (SOQ at 9:30); SPXW weeklies settle PM (4:00).
# June 19 2026 is Juneteenth (market holiday), so June monthlies expire Thu 18th.
EXPIRIES = [
    (datetime(2026, 2, 20, 9, 30), "AM", "Feb 20 (monthly)"),
    (datetime(2026, 2, 27, 16, 0), "PM", "Feb 27 (weekly)"),
    (datetime(2026, 3, 13, 16, 0), "PM", "Mar 13 (weekly)"),
    (datetime(2026, 3, 20, 9, 30), "AM", "Mar 20 (monthly)"),
    (datetime(2026, 4, 17, 9, 30), "AM", "Apr 17 (monthly)"),
    (datetime(2026, 5, 15, 9, 30), "AM", "May 15 (monthly)"),
    (datetime(2026, 6, 18, 9, 30), "AM", "Jun 18 (monthly, Thu)"),
    (datetime(2026, 9, 18, 9, 30), "AM", "Sep 18 (monthly)"),
    (datetime(2026, 12, 18, 9, 30), "AM", "Dec 18 (monthly)"),
]

MIN_PER_YEAR = 525600.0


def year_frac(expiry: datetime) -> float:
    """Calendar-minute year fraction, the convention the VIX white paper uses."""
    return (expiry - SNAPSHOT).total_seconds() / 60.0 / MIN_PER_YEAR
