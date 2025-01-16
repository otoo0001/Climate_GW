# Main script to perform ANOVA and significance analysis for groundwater data.
import os
import pandas as pd
import numpy as np
import xarray as xr
import warnings
import time
import concurrent.futures
from tqdm import tqdm
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=FutureWarning, message="verbose is deprecated since functions should not print results")

# ----------------------------
# Configuration
# ----------------------------
input_file = "/scratch-shared/otoo0001/data/Jarno_dataset/ransient_Gw_top_30arcsec_monthly_1958-2015_Australia_detrend_deseasoned.nc"
nino_file = "/scratch-shared/otoo0001/data/paper_2/Nino/iersst_nino3.4a_detrend_seasmean.nc"
time_start = "1979-01-01"
time_end   = "2015-12-31"
CHUNK_SIZE = 10_000
MAX_WORKERS = 100
test_mode = True
num_test_grids = 100 if test_mode else None

output_folder = "/scratch-shared/otoo0001/GLOBGDES/Paper2_mapping_wetlands/groundwater_enso_analysis_v2025_01_09/all_v_2025_01_09_test/"
os.makedirs(output_folder, exist_ok=True)
log_file = os.path.join(output_folder, "processing_log.txt")

def log_message(msg, level='info'):
    """
    Logs a message to both the console and a log file.
    
    Parameters:
        msg (str): The message to log.
        level (str): The logging level ('info', 'warning', 'error').
    """
    if level == 'info':
        print(f"[INFO]: {msg}")
    elif level == 'warning':
        print(f"[WARNING]: {msg}")
    elif level == 'error':
        print(f"[ERROR]: {msg}")
    with open(log_file, "a") as f:
        f.write(f"{time.ctime()}: [{level.upper()}]: {msg}\n")

def perform_anova(gw_values, enso_values):
    """
    Performs ANOVA across 3 phases: El Niño, La Niña, Neutral.
    Returns (F_stat, df_between, df_within, P_value, significant_boolean).
    """
    from scipy import stats
    try:
        phase_data = {}
        for phase in ["El Niño","La Niña","Neutral"]:
            phase_data[phase] = gw_values[enso_values == phase]

        valid_phases = [vals for vals in phase_data.values() if len(vals)>1]
        if len(valid_phases)<2:
            return np.nan, np.nan, np.nan, np.nan, False

        f_stat, p_val = stats.f_oneway(*valid_phases)
        k = len(valid_phases)
        N = sum(len(arr) for arr in valid_phases)
        df_between = k - 1
        df_within = N - k

        return f_stat, df_between, df_within, p_val, (p_val<0.05)
    except Exception as e:
        log_message(f"ANOVA error: {e}", level='error')
        return np.nan, np.nan, np.nan, np.nan, False

def dummy_regression_phase_effect(df):
    """
    Dummy-coded regression to see which phase (El Niño, La Niña) 
    has the strongest effect vs. Neutral baseline.
    
    df must have columns: 'GW_Data','ENSO_Phase'
    Returns dict with:
      'coef_ElNino','pval_ElNino','coef_LaNina','pval_LaNina','phase_winner'
    """
    df_ = df.dropna(subset=["GW_Data", "ENSO_Phase"])
    if len(df_) < 3:
        return {
            "coef_ElNino": np.nan, "pval_ElNino": np.nan,
            "coef_LaNina": np.nan, "pval_LaNina": np.nan,
            "phase_winner": "No Significant Effect"
        }

    df_["ElNino_dummy"] = (df_["ENSO_Phase"] == "El Niño").astype(int)
    df_["LaNina_dummy"] = (df_["ENSO_Phase"] == "La Niña").astype(int)

    y = df_["GW_Data"].values
    X = df_[["ElNino_dummy", "LaNina_dummy"]].values
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X, missing="drop").fit()
    except Exception as e:
        log_message(f"OLS Regression error: {e}", level='error')
        return {
            "coef_ElNino": np.nan, "pval_ElNino": np.nan,
            "coef_LaNina": np.nan, "pval_LaNina": np.nan,
            "phase_winner": "No Significant Effect"
        }

    if len(model.params) < 3:
        # Handle cases with insufficient parameters
        b0 = model.params[0] if len(model.params) > 0 else np.nan
        b_elnino = model.params[1] if len(model.params) > 1 else np.nan
        b_lanina = np.nan
    else:
        b0, b_elnino, b_lanina = model.params
    pvals = model.pvalues
    p_elnino = pvals[1] if len(pvals) > 1 else np.nan
    p_lanina = pvals[2] if len(pvals) > 2 else np.nan

    # Determine the phase winner based on coefficients and p-values
    winner = "No Significant Effect"
    if not np.isnan(b_elnino) and p_elnino < 0.05:
        direction = "Increases" if b_elnino > 0 else "Decreases"
        winner = f"El Niño {direction}"
    if not np.isnan(b_lanina) and p_lanina < 0.05:
        direction = "Increases" if b_lanina > 0 else "Decreases"
        # Compare absolute coefficients if both phases are significant
        if winner != "No Significant Effect":
            if abs(b_lanina) > abs(b_elnino):
                winner = f"La Niña {direction}"
        else:
            winner = f"La Niña {direction}"

    return {
        "coef_ElNino": b_elnino, "pval_ElNino": p_elnino,
        "coef_LaNina": b_lanina, "pval_LaNina": p_lanina,
        "phase_winner": winner
    }

def analyze_significant_grid(lat, lon, combined_df, max_lag=12):
    """
    For each significant grid: R-squared, Cross-corr, Granger, dummy-coded regression.
    Also determines the phase winner indicating if El Niño or La Niña causes an increase or decrease in GW data.
    """
    from sklearn.linear_model import LinearRegression
    from scipy.signal import correlate
    from statsmodels.tsa.stattools import grangercausalitytests

    df_sub = combined_df[(combined_df['lat'] == lat) & (combined_df['lon'] == lon)]
    if len(df_sub) < (max_lag + 1):
        return {
            'lat': lat, 'lon': lon,
            'R_squared': np.nan, 'Cross_Correlation': np.nan, 'Max_Lag': np.nan,
            'Granger_p_value': np.nan, 'Granger_Lag': np.nan, 'Granger_Significant': False,
            'coef_ElNino': np.nan, 'pval_ElNino': np.nan,
            'coef_LaNina': np.nan, 'pval_LaNina': np.nan,
            'phase_winner': "No Significant Effect"
        }

    X = df_sub['Nino34_Index'].values.reshape(-1, 1)
    y = df_sub['GW_Data'].values

    r_squared = np.nan
    cross_corr = np.nan
    max_lag_value = np.nan
    granger_pval = np.nan
    granger_lag = np.nan
    granger_significant = False

    # R-squared
    if not np.any(np.isnan(X)) and not np.any(np.isnan(y)):
        try:
            lin_mod = LinearRegression()
            lin_mod.fit(X, y)
            r_squared = lin_mod.score(X, y)
        except Exception as e:
            log_message(f"Error in LinearRegression grid({lat},{lon}): {e}", level='error')

    # Cross-correlation
    gw = y
    nino34 = df_sub['Nino34_Index'].values
    gw_std = gw.std()
    nino34_std = nino34.std()
    if gw_std != 0 and nino34_std != 0:
        corr = correlate(gw - gw.mean(), nino34 - nino34.mean(), mode='full')
        lags = np.arange(-len(gw) + 1, len(gw))
        mask = (lags >= 0) & (lags <= max_lag)
        corr = corr[mask]
        lags = lags[mask]
        corr_norm = corr / (gw_std * nino34_std * len(gw))
        if len(corr_norm) > 0:
            idx = np.argmax(np.abs(corr_norm))
            cross_corr = corr_norm[idx]
            max_lag_value = lags[idx]

    # Granger Causality
    try:
        data = pd.DataFrame({'y': y, 'x': nino34})
        gr_result = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
        pvals = {}
        for lag in range(1, max_lag + 1):
            pvals[lag] = gr_result[lag][0]['ssr_ftest'][1]
        valid_pvals = {lg: pv for lg, pv in pvals.items() if not np.isnan(pv)}
        if valid_pvals:
            best_lg = min(valid_pvals, key=valid_pvals.get)
            granger_pval = valid_pvals[best_lg]
            granger_lag = best_lg
            granger_significant = (granger_pval < 0.05)
    except Exception as e:
        log_message(f"Error Granger causality grid({lat},{lon}): {e}", level='error')

    # Dummy-coded regression
    dummy_res = dummy_regression_phase_effect(df_sub)
    coef_elnino = dummy_res["coef_ElNino"]
    pval_elnino = dummy_res["pval_ElNino"]
    coef_lanina = dummy_res["coef_LaNina"]
    pval_lanina = dummy_res["pval_LaNina"]
    phase_winner = dummy_res["phase_winner"]

    return {
        'lat': lat, 'lon': lon,
        'R_squared': r_squared,
        'Cross_Correlation': cross_corr, 'Max_Lag': max_lag_value,
        'Granger_p_value': granger_pval, 'Granger_Lag': granger_lag, 'Granger_Significant': granger_significant,
        'coef_ElNino': coef_elnino, 'pval_ElNino': pval_elnino,
        'coef_LaNina': coef_lanina, 'pval_LaNina': pval_lanina,
        'phase_winner': phase_winner
    }

def main():
    log_message("Processing started.")

    # 1) LOAD & ALIGN
    log_message("Loading GW_Data...")
    try:
        ds_gw = xr.open_dataset(input_file)
        gw_data = ds_gw["groundwater_heads_top"].sel(time=slice(time_start, time_end))
        gw_data["time"] = pd.to_datetime(gw_data["time"].values)
    except Exception as e:
        log_message(f"Error loading GW_Data: {e}", level='error')
        sys.exit(1)

    log_message("Loading Nino3.4 data...")
    try:
        ds_nino = xr.open_dataset(nino_file, decode_times=False)
        ref_date_str = ds_nino["time"].attrs.get("units", "").split("since")[-1].strip()
        ref_date = pd.Timestamp(ref_date_str)
        decoded_time = [ref_date + pd.DateOffset(months=int(t)) for t in ds_nino["time"].values]
        ds_nino["time"] = xr.DataArray(decoded_time, dims="time")
        nino_data = ds_nino["Nino3.4"].sel(time=slice(time_start, time_end))
        nino_data["time"] = pd.to_datetime(nino_data["time"].values) + pd.offsets.MonthEnd(0)
    except Exception as e:
        log_message(f"Error loading Nino3.4 data: {e}", level='error')
        sys.exit(1)

    # Align times
    log_message("Aligning times for GW_Data & Nino3.4 data.")
    try:
        common_times = np.intersect1d(gw_data["time"].values, nino_data["time"].values)
        gw_data = gw_data.sel(time=common_times)
        nino_data = nino_data.sel(time=common_times)
    except Exception as e:
        log_message(f"Error aligning data: {e}", level='error')
        sys.exit(1)

    # ENSO phases
    log_message("Classifying ENSO Phases...")
    try:
        enso_phases = xr.where(nino_data >= 0.5, "El Niño",
                               xr.where(nino_data <= -0.5, "La Niña", "Neutral"))
        enso_phases = enso_phases.broadcast_like(gw_data)
    except Exception as e:
        log_message(f"Error classifying ENSO phases: {e}", level='error')
        sys.exit(1)

    # 2) VALID GRIDS
    log_message("Finding non-zero grids across time.")
    try:
        valid_mask = (gw_data != 0).all(dim="time")  # shape [lat, lon]
        lat_idx, lon_idx = np.where(valid_mask.values)
        lat_vals = gw_data["lat"].values[lat_idx]
        lon_vals = gw_data["lon"].values[lon_idx]

        valid_grids = list(zip(lat_vals, lon_vals))
        if num_test_grids and len(valid_grids) > num_test_grids:
            valid_grids = valid_grids[:num_test_grids]
        total_grids = len(valid_grids)
        log_message(f"Found {total_grids} valid grids.")
        if not valid_grids:
            log_message("No valid grids => exit.", level='warning')
            return
    except Exception as e:
        log_message(f"Error identifying valid grids: {e}", level='error')
        sys.exit(1)

    anova_files = []          # Store chunk ANOVA partial CSV paths
    significance_files = []   # Store chunk significance partial CSV paths

    # Helper to gather data for significance
    def gather_data_for_grid(lat_, lon_):
        try:
            times = gw_data.sel(lat=lat_, lon=lon_)["time"].values
            gw_arr = gw_data.sel(lat=lat_, lon=lon_).values
            enso_arr = enso_phases.sel(lat=lat_, lon=lon_).values
            nino_arr = nino_data.sel(time=times).values
            df_sub = pd.DataFrame({
                "lat": lat_, "lon": lon_,
                "time": times,
                "GW_Data": gw_arr,
                "Nino34_Index": nino_arr,
                "ENSO_Phase": enso_arr
            })
            df_sub.dropna(subset=["GW_Data", "Nino34_Index", "ENSO_Phase"], inplace=True)
            return df_sub
        except Exception as e:
            log_message(f"Data gathering error for grid lat={lat_}, lon={lon_}: {e}", level='error')
            return pd.DataFrame()

    # 3) CHUNK PROCESS
    start_time_ = time.time()
    for chunk_start in range(0, total_grids, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_grids)
        chunk_grids = valid_grids[chunk_start:chunk_end]
        log_message(f"Processing chunk {chunk_start}:{chunk_end} (#grids={len(chunk_grids)})...")

        # 3A) ANOVA
        tasks = []
        for (lat_, lon_) in chunk_grids:
            gw_arr = gw_data.sel(lat=lat_, lon=lon_).values
            enso_arr = enso_phases.sel(lat=lat_, lon=lon_).values
            tasks.append((lat_, lon_, gw_arr, enso_arr))

        partial_anova = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(perform_anova, t[2], t[3]): (t[0], t[1]) for t in tasks}
            for fut in tqdm(concurrent.futures.as_completed(future_map),
                            total=len(tasks),
                            desc=f"ANOVA chunk {chunk_start}-{chunk_end}", ncols=100):
                lat_, lon_ = future_map[fut]
                try:
                    f_stat, df_b, df_w, p_val, sig = fut.result()
                    partial_anova.append({
                        "lat": lat_, "lon": lon_,
                        "F_statistic": f_stat,
                        "df_between": df_b, "df_within": df_w,
                        "P_value": p_val, "significant": sig
                    })
                except Exception as e:
                    log_message(f"ANOVA error lat={lat_}, lon={lon_}: {e}", level='error')
                    partial_anova.append({
                        "lat": lat_, "lon": lon_,
                        "F_statistic": np.nan, "df_between": np.nan,
                        "df_within": np.nan, "P_value": np.nan,
                        "significant": False
                    })

        anova_df = pd.DataFrame(partial_anova)

        # 3B) Save ANOVA chunk results
        anova_chunk_file = f"anova_chunk_{chunk_start}_{chunk_end}.csv"
        anova_chunk_path = os.path.join(output_folder, anova_chunk_file)
        anova_df.to_csv(anova_chunk_path, index=False)
        anova_files.append(anova_chunk_path)
        log_message(f"Saved chunk ANOVA => {anova_chunk_path}")

        # 3C) Filter significant and do significance analysis
        sig_df = anova_df[anova_df["significant"] == True]
        if sig_df.empty:
            # No significance => create an empty DataFrame with the required columns
            significance_chunk_df = pd.DataFrame(columns=[
                "lat", "lon", "R_squared", "Cross_Correlation", "Max_Lag",
                "Granger_p_value", "Granger_Lag", "Granger_Significant",
                "coef_ElNino", "pval_ElNino",
                "coef_LaNina", "pval_LaNina",
                "phase_winner"
            ])
        else:
            # Gather data and run 'analyze_significant_grid' for each
            significance_results = []
            for row in sig_df.itertuples(index=False):
                lat_, lon_ = row.lat, row.lon
                df_sub = gather_data_for_grid(lat_, lon_)
                res_dict = analyze_significant_grid(lat_, lon_, df_sub)
                significance_results.append(res_dict)

            significance_chunk_df = pd.DataFrame(significance_results)

        # 3D) Save significance chunk
        significance_chunk_file = f"signif_chunk_{chunk_start}_{chunk_end}.csv"
        significance_chunk_path = os.path.join(output_folder, significance_chunk_file)
        significance_chunk_df.to_csv(significance_chunk_path, index=False)
        significance_files.append(significance_chunk_path)
        log_message(f"Saved chunk significance => {significance_chunk_path}")

        # 3E) Clear references from memory
        del tasks, partial_anova, anova_df, sig_df, significance_chunk_df
        log_message(f"Memory cleared for chunk {chunk_start}:{chunk_end}...")

    # End chunk loop
    elapsed = (time.time() - start_time_) / 60
    log_message(f"All chunks processed. Elapsed={elapsed:.2f} min")

    # 4) Combine partial ANOVA CSVs
    log_message("Combining partial ANOVA CSVs => final_overall_anova.xlsx")
    try:
        anova_dfs = []
        for fpath in tqdm(sorted(anova_files), desc="Combining ANOVA CSVs", ncols=100):
            anova_dfs.append(pd.read_csv(fpath))
        final_anova_df = pd.concat(anova_dfs, ignore_index=True)
        final_anova_xlsx = os.path.join(output_folder, "final_overall_anova.xlsx")
        final_anova_df.to_excel(final_anova_xlsx, index=False)
        log_message(f"Final ANOVA => {final_anova_xlsx}")
    except Exception as e:
        log_message(f"Error combining ANOVA CSVs: {e}", level='error')

    # 5) Combine partial significance CSVs
    log_message("Combining partial significance CSVs => final_significance.xlsx")
    try:
        signif_dfs = []
        for fpath in tqdm(sorted(significance_files), desc="Combining Significance CSVs", ncols=100):
            tmpdf = pd.read_csv(fpath)
            signif_dfs.append(tmpdf)
        final_signif_df = pd.concat(signif_dfs, ignore_index=True)
        final_signif_xlsx = os.path.join(output_folder, "final_significance.xlsx")
        final_signif_df.to_excel(final_signif_xlsx, index=False)
        log_message(f"Final significance => {final_signif_xlsx}")
    except Exception as e:
        log_message(f"Error combining Significance CSVs: {e}", level='error')

    # 6) (Optional) Delete partial chunk CSVs
    log_message("Deleting partial chunk CSVs now that finals are created...")
    try:
        for fpath in anova_files + significance_files:
            if os.path.exists(fpath):
                os.remove(fpath)
        log_message("All partial chunk CSVs deleted. Done!")
    except Exception as e:
        log_message(f"Error deleting partial chunk CSVs: {e}", level='error')

if __name__ == "__main__":
    main()
