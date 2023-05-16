import numpy as np
import pandas as pd


def cusum(df, direction="p", delta=4, C_thr=3, est_length="3 days"):
    """
    Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318
       https://www.amazon.com/dp/0470169923

    Parameters
    ----------
    df        :  data to analyze
    direction :  negative or positive? 'n' or 'p'
    delta     :  parameter to calculate slack value K =>  K = (delta/2)*sigma
    K         :  reference value, allowance, slack value for each pipe
    C_thr     :  threshold for raising flag
    est_length:  Window for estimating distribution parameters mu and sigma, needs to be longer than one timestep (int=hours, str=timedelta, auto=first timestep)
    leak_det  :  Leaks detected
    """
    if est_length == "auto":
        distribution_window = pd.Timedelta(df.index[1] - df.index[0])
    elif type(est_length) == int or type(est_length) == float:
        distribution_window = pd.Timedelta(hours=est_length)
    else:
        distribution_window = pd.Timedelta(est_length)

    if df.index[1] - df.index[0] > distribution_window:
        raise ValueError(
            f"est_length ({distribution_window}) needs to be longer than one timestep ({df.index[1] - df.index[0]})"
        )

    ar_mean = np.zeros(df.shape[1])
    ar_sigma = np.zeros(df.shape[1])
    for i, col in enumerate(df.columns):
        traj_ = df[col].copy()
        # BUG: This sets the first timeseries index to first non zero value
        # TODO Test
        non_zero_traj = traj_[traj_ != 0]
        if non_zero_traj.empty:
            ar_mean[i] = 0
            ar_sigma[i] = 0
        else:
            traj_ = non_zero_traj
            ar_mean[i] = traj_.loc[: (traj_.index[0] + distribution_window)].mean()
            ar_sigma[i] = traj_.loc[: (traj_.index[0] + distribution_window)].std()

    ar_K = (delta / 2) * ar_sigma

    sumlm = np.frompyfunc(lambda a, b: 0 if a + b < 0 else a + b, 2, 1)
    df_cs = sumlm.accumulate(df - ar_mean - ar_K, dtype=object)
    df_cs.iloc[0] = 0

    leak_det = pd.Series(dtype=object)
    for i, pipe in enumerate(df_cs):
        C_thr_abs = C_thr * ar_sigma[i]
        if any(df_cs[pipe] > C_thr_abs):
            leak_det[pipe] = df_cs.index[(df_cs[pipe] > C_thr_abs).values][0]

    return leak_det, df_cs


def cusum_old(df, direction="p", delta=4, C_thr=3, est_length="3 days"):
    """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318
       https://www.amazon.com/dp/0470169923
    df        :  data to analyze
    direction :  negative or positive? 'n' or 'p'
    delta     :  parameter to calculate slack value K =>  K = (delta/2)*sigma
    K         :  reference value, allowance, slack value for each pipe
    C_thr     :  threshold for raising flag
    est_length:  Window for estimating distribution parameters mu and sigma, needs to be longer than one timestep
    leak_det  :  Leaks detected
    """
    if est_length == "auto":
        distribution_window = pd.Timedelta(df.index[1] - df.index[0])
    elif type(est_length) == int or type(est_length) == float:
        distribution_window = pd.Timedelta(hours=est_length)
    else:
        distribution_window = pd.Timedelta(est_length)

    if df.index[1] - df.index[0] > distribution_window:
        raise ValueError(
            f"est_length ({distribution_window}) needs to be longer than one timestep ({df.index[1] - df.index[0]})"
        )

    ar_mean = np.zeros(df.shape[1])
    ar_sigma = np.zeros(df.shape[1])
    for i, col in enumerate(df.columns):
        traj_ = df[col].copy()
        non_zero_traj = traj_[traj_ != 0]
        if non_zero_traj.empty:
            ar_mean[i] = 0
            ar_sigma[i] = 0
        else:
            traj_ = non_zero_traj
            ar_mean[i] = traj_.loc[: (traj_.index[0] + distribution_window)].mean()
            ar_sigma[i] = traj_.loc[: (traj_.index[0] + distribution_window)].std()

    ar_K = (delta / 2) * ar_sigma

    cumsum = np.zeros(df.shape)

    if direction == "p":
        for i in range(1, df.shape[0]):
            cumsum[i, :] = [
                max(0, j) for j in df.iloc[i, :] - ar_mean + cumsum[i - 1, :] - ar_K
            ]
    elif direction == "n":
        for i in range(1, df.shape[0]):
            cumsum[i] = [
                max(0, j) for j in -df.iloc[i, :] + ar_mean + cumsum[i - 1, :] - ar_K
            ]

    # df_cs     :  pd.DataFrame containing cusum-values for each df column
    df_cs = pd.DataFrame(cumsum, columns=df.columns, index=df.index)

    leak_det = pd.Series(dtype=object)
    for i, pipe in enumerate(df_cs):
        C_thr_abs = C_thr * ar_sigma[i]
        if any(df_cs[pipe] > C_thr_abs):
            leak_det[pipe] = df_cs.index[(df_cs[pipe] > C_thr_abs).values][0]

    return leak_det, df_cs
