import numpy as np
import pandas as pd
from src.network_analysis.scores import calculate_average_betweenness, get_crime_data, aggregate_lsoas_to_wards


def within_between_variance(df, value_col = 'avg_betweenness', unit_col='WD21CD', time_col='period'):
    """
    Decompose variance of value_col into between-unit and within-unit (temporal).
    within_share near 0  -> regressor is essentially a ward characteristic;
                            Prophet's per-ward intercept absorbs it, no forecasting help.
    within_share large   -> there is temporal movement worth trying to exploit.
    """
    d = df[[unit_col, time_col, value_col]].dropna().copy()

    grand_mean  = d[value_col].mean()
    unit_means  = d.groupby(unit_col)[value_col].transform('mean')

    within_dev  = d[value_col] - unit_means        # deviation from ward's own mean over time
    between_dev = unit_means - grand_mean          # ward mean vs grand mean

    within_var  = (within_dev  ** 2).mean()
    between_var = (between_dev ** 2).mean()
    within_share = within_var / (within_var + between_var)

    per_unit_temporal_std = d.groupby(unit_col)[value_col].std(ddof=0)
    ward_mean_spread      = d.groupby(unit_col)[value_col].mean().std(ddof=0)

    return {
        'within_share':            within_share,             # the headline number
        'within_var':              within_var,
        'between_var':             between_var,
        'median_within_ward_std':  per_unit_temporal_std.median(),
        'cross_ward_mean_std':     ward_mean_spread,
    }


def demeaned_lead_lag(df, reg_col, target_col, unit_col='WD21CD', time_col='period', lag=1):
    """
    Strip each ward's own mean from BOTH series (the cross-sectional part Prophet
    already captures), then ask: does the regressor, lagged by `lag` months,
    correlate with the target's temporal movement?

    r ~ 0 at lag>=1  -> no leading temporal content; cannot help Prophet.
    r meaningfully != 0 -> there is something to exploit (even 0.05-0.15 can be
                           real at noisy ward-month resolution).
    Tip: also run with lag=0. A big lag-0 r but ~0 lag-1 r is the signature of
    contemporaneous leakage, not forecasting skill.
    """
    d = df[[unit_col, time_col, reg_col, target_col]].dropna().copy()
    d = d.sort_values([unit_col, time_col])

    d['reg_w'] = d[reg_col]    - d.groupby(unit_col)[reg_col].transform('mean')
    d['tgt_w'] = d[target_col] - d.groupby(unit_col)[target_col].transform('mean')
    d['reg_w_lag'] = d.groupby(unit_col)['reg_w'].shift(lag)

    sub = d.dropna(subset=['reg_w_lag', 'tgt_w'])
    r = np.corrcoef(sub['reg_w_lag'], sub['tgt_w'])[0, 1]
    return r, len(sub)

def _demean(d, col, unit_col, moy_col):
    s = d[col]
    return (s - d.groupby(unit_col)[col].transform('mean')
              - d.groupby(moy_col)[col].transform('mean') + s.mean())

def _resid(y, x):
    X = np.c_[np.ones(len(x)), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def strict_lead_lag(df, reg_col, target_col, unit_col='WD21CD', time_col='period', lag=1):
    d = df[[unit_col, time_col, reg_col, target_col]].dropna().copy()
    d = d.sort_values([unit_col, time_col])
    d['moy'] = d[time_col] % 12
    d['reg_w'] = _demean(d, reg_col, unit_col, 'moy')
    d['tgt_w'] = _demean(d, target_col, unit_col, 'moy')
    d['reg_lag'] = d.groupby(unit_col)['reg_w'].shift(lag)
    d['tgt_lag'] = d.groupby(unit_col)['tgt_w'].shift(lag)
    d = d.dropna(subset=['reg_lag', 'tgt_lag', 'tgt_w'])
    reg_r = _resid(d['reg_lag'].values, d['tgt_lag'].values)
    tgt_r = _resid(d['tgt_w'].values,  d['tgt_lag'].values)
    return np.corrcoef(reg_r, tgt_r)[0, 1], len(d)

if __name__ == "__main__":
    # Build the ward-period counts panel (counts, NOT shares)
    raw = aggregate_lsoas_to_wards(get_crime_data())          # long: WD21CD, period, crime_type, count
    panel = (raw.pivot_table(index=['WD21CD', 'period'],
                            columns='crime_type', values='count', fill_value=0)
                .reset_index())

    for reg in ['Robbery', 'Theft from the person', 'Possession of weapons']:
        r, n = strict_lead_lag(panel, reg, 'Violence and sexual offences')
        print(f"{reg:28s} strict lag1 r={r:+.3f} (n={n})")