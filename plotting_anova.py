#!/usr/bin/env python3
# plot_anova_results.py

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm  # For colormap handling
from matplotlib.patches import Patch
import folium
from folium.plugins import MarkerCluster
import xarray as xr
from tqdm import tqdm
from scipy.spatial import cKDTree

def print_and_flush(msg, log_file):
    """
    Prints a message and appends it to a log file.
    """
    print(msg, flush=True)
    with open(log_file, "a") as f:
        f.write(f"{pd.Timestamp.now()}: {msg}\n")

def plot_anova_results(data_folder, plots_folder, groundwater_heads_nc, groundwater_variable):
    """
    Plots ANOVA results including F-statistic map, p-value map,
    scatter plot of p-values, significant grids map, histogram of p-values,
    joint plot of F-statistic vs. p-value, interactive map, and groundwater heads histogram.
    
    Parameters:
    - data_folder: Directory where ANOVA chunk CSV files are stored.
    - plots_folder: Directory where plots will be saved.
    - groundwater_heads_nc: Path to the groundwater heads NetCDF file.
    - groundwater_variable: Variable name for groundwater heads in the NetCDF file.
    """
    start_time = time.time()
    
    # Ensure plots_folder exists
    os.makedirs(plots_folder, exist_ok=True)
    
    # Log file path
    log_file = os.path.join(plots_folder, "plotting_log.txt")
    
    print_and_flush("Plotting script started.", log_file)
    
    # ----------------------------
    # 1. Combine ANOVA Chunk CSVs
    # ----------------------------
    final_anova_excel = os.path.join(data_folder, "final_overall_anova.xlsx")
    
    if os.path.exists(final_anova_excel):
        print_and_flush(f"Loading combined ANOVA results from {final_anova_excel}...", log_file)
        try:
            anova_df = pd.read_excel(final_anova_excel)
            print_and_flush(f"ANOVA data loaded with shape: {anova_df.shape}", log_file)
        except Exception as e:
            print_and_flush(f"Error loading ANOVA results: {e}", log_file)
            return
    else:
        # Combine per-chunk CSVs
        print_and_flush("Combined ANOVA Excel not found. Combining per-chunk CSVs...", log_file)
        anova_csv_files = sorted([
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.startswith("anova_chunk_") and f.endswith(".csv")
        ])
        
        if not anova_csv_files:
            print_and_flush("No ANOVA chunk CSV files found. Exiting plotting script.", log_file)
            return
        
        print_and_flush(f"Found {len(anova_csv_files)} ANOVA chunk files. Combining them...", log_file)
        anova_dfs = []
        for csv_file in tqdm(anova_csv_files, desc="Reading ANOVA CSVs", ncols=100):
            try:
                df = pd.read_csv(csv_file)
                anova_dfs.append(df)
            except Exception as e:
                print_and_flush(f"Error reading {csv_file}: {e}", log_file)
        
        if not anova_dfs:
            print_and_flush("No ANOVA data to plot after reading CSVs. Exiting.", log_file)
            return
        
        anova_df = pd.concat(anova_dfs, ignore_index=True)
        print_and_flush(f"Combined ANOVA DataFrame shape: {anova_df.shape}", log_file)
        
        # Optionally, save the combined DataFrame to Excel
        # print_and_flush(f"Saving combined ANOVA results to {final_anova_excel}...", log_file)
        # anova_df.to_excel(final_anova_excel, index=False)
        # print_and_flush("Combined ANOVA results saved successfully.", log_file)
    
    # ----------------------------
    # 2. Data Cleaning and Aggregation
    # ----------------------------
    required_columns = {'lat', 'lon', 'F_statistic', 'df_between', 'df_within', 'P_value', 'significant'}
    if not required_columns.issubset(anova_df.columns):
        raise ValueError(f"Input data must contain columns: {required_columns}")
    
    # Round lat and lon to 6 decimal places to avoid floating-point duplicates
    anova_df['lat'] = anova_df['lat'].round(6)
    anova_df['lon'] = anova_df['lon'].round(6)
    
    # Check for duplicate (lat, lon) pairs
    print_and_flush("Checking for duplicate (lat, lon) entries...", log_file)
    duplicate_counts = anova_df.duplicated(subset=['lat', 'lon'], keep=False).sum()
    total_duplicates = duplicate_counts // 2  # Each duplicate is counted twice
    if total_duplicates > 0:
        print_and_flush(f"Found {total_duplicates} duplicate (lat, lon) pairs. Aggregating duplicates by mean.", log_file)
        # Aggregate duplicates by taking the mean of F_statistic and P_value
        # For 'significant', use 'any' to indicate if any of the duplicates is significant
        anova_df = anova_df.groupby(['lat', 'lon'], as_index=False).agg({
            'F_statistic': 'mean',
            'df_between': 'mean',  # Assuming df_between is consistent
            'df_within': 'mean',   # Assuming df_within is consistent
            'P_value': 'min',      # Use the smallest p-value in the group
            'significant': 'any'   # True if any p-value in the group is significant
        })
        print_and_flush("Aggregation complete.", log_file)
        
        # Verify that duplicates are resolved
        remaining_duplicates = anova_df.duplicated(subset=['lat', 'lon'], keep=False).sum()
        if remaining_duplicates > 0:
            print_and_flush(f"Warning: {remaining_duplicates // 2} duplicate (lat, lon) pairs remain after aggregation.", log_file)
        else:
            print_and_flush("No duplicate (lat, lon) entries remain after aggregation.", log_file)
    else:
        print_and_flush("No duplicate (lat, lon) entries found.", log_file)
    
    # Convert P_value to numeric, coercing errors to NaN
    anova_df['P_value'] = pd.to_numeric(anova_df['P_value'], errors='coerce')
    
    # Check for NaN values after conversion
    nan_pvalues = anova_df['P_value'].isna().sum()
    if nan_pvalues > 0:
        print_and_flush(f"Found {nan_pvalues} NaN p-values after conversion.", log_file)
        # Remove rows with NaN p-values
        anova_df = anova_df.dropna(subset=['P_value'])
        print_and_flush(f"Removed {nan_pvalues} rows with NaN p-values.", log_file)
    
    # Check for significant p-values
    significant_count = (anova_df['P_value'] < 0.05).sum()
    total_count = anova_df.shape[0]
    print_and_flush(f"Number of significant p-values (p < 0.05): {significant_count} out of {total_count}", log_file)
    
    # Ensure 'significant' is boolean
    anova_df['significant'] = anova_df['significant'].astype(bool)
    
    # ----------------------------
    # 3. Load and Process Groundwater Heads Data
    # ----------------------------
    print_and_flush("Loading and processing Groundwater Heads data from NetCDF...", log_file)
    try:
        ds_gw = xr.open_dataset(groundwater_heads_nc)
        gw_data = ds_gw[groundwater_variable]
        # Compute time average
        print_and_flush("Computing time average of groundwater heads...", log_file)
        gw_mean = gw_data.mean(dim='time').values  # Assuming 'time' is a dimension
        lat_vals = gw_data['lat'].values
        lon_vals = gw_data['lon'].values
        # Create DataFrame
        gw_mean_df = pd.DataFrame({
            'lat': np.repeat(lat_vals, len(lon_vals)),
            'lon': np.tile(lon_vals, len(lat_vals)),
            'head': gw_mean.flatten()
        })
        print_and_flush(f"Groundwater Heads data averaged and loaded with shape: {gw_mean_df.shape}", log_file)
    except Exception as e:
        print_and_flush(f"Error loading Groundwater Heads data: {e}", log_file)
        gw_mean_df = pd.DataFrame()  # Create empty DataFrame
    
    # ----------------------------
    # 4. Merge ANOVA Results with Groundwater Heads
    # ----------------------------
    if not gw_mean_df.empty:
        print_and_flush("Merging ANOVA results with Groundwater Heads data...", log_file)
        try:
            # Round coordinates to 6 decimal places to match ANOVA data
            gw_mean_df['lat'] = gw_mean_df['lat'].round(6)
            gw_mean_df['lon'] = gw_mean_df['lon'].round(6)
        
            # Merge with anova_df
            merged_df = pd.merge(anova_df, gw_mean_df, on=['lat', 'lon'], how='left')
            print_and_flush(f"Merged data shape: {merged_df.shape}", log_file)
        
            # Check for missing head values
            missing_heads = merged_df['head'].isna().sum()
            if missing_heads > 0:
                print_and_flush(f"Found {missing_heads} grids without corresponding groundwater head data. Removing these grids from the histogram.", log_file)
                merged_df = merged_df.dropna(subset=['head'])
                print_and_flush(f"After removal, merged data shape: {merged_df.shape}", log_file)
        except Exception as e:
            print_and_flush(f"Error merging ANOVA and Groundwater Heads data: {e}", log_file)
            merged_df = pd.DataFrame()  # Empty DataFrame
    else:
        print_and_flush("Groundwater Heads data is empty. Skipping Groundwater Heads Histogram.", log_file)
        merged_df = pd.DataFrame()
    
    # ----------------------------
    # 4a. Create Pivot Tables for P-values and Significance
    # ----------------------------
    print_and_flush("Creating pivot tables for p-values and significance...", log_file)
    try:
        # Create P_pivot: Pivot table for p-values
        P_pivot = anova_df.pivot_table(
            index='lat',
            columns='lon',
            values='P_value',
            aggfunc='mean'
        )
        
        # Create S_pivot: Pivot table for significance
        S_pivot = anova_df.pivot_table(
            index='lat',
            columns='lon',
            values='significant',
            aggfunc='first'  # Assumes consistency within groups
        )
        
        # Ensure S_pivot is boolean and handle NaNs
        S_pivot = S_pivot.fillna(False).astype(bool)
        
        # Sort the pivot tables by descending latitude for correct map orientation
        P_pivot = P_pivot.sort_index(ascending=False)
        S_pivot = S_pivot.sort_index(ascending=False)
        
        print_and_flush("Pivot tables created successfully.", log_file)
        
        # Optional: Log data types and NaN counts
        print_and_flush(f"P_pivot data types:\n{P_pivot.dtypes}", log_file)
        print_and_flush(f"S_pivot data types:\n{S_pivot.dtypes}", log_file)
        print_and_flush(f"Number of NaNs in S_pivot:\n{S_pivot.isna().sum()}", log_file)
    except Exception as e:
        print_and_flush(f"Error creating pivot tables: {e}", log_file)
        P_pivot = pd.DataFrame()
        S_pivot = pd.DataFrame()
    
    # ----------------------------
    # 5. Prepare Data for Heatmap Plotting
    # ----------------------------
    print_and_flush("Preparing data for heatmap plotting...", log_file)
    try:
        # Create separate DataFrames for significant and insignificant p-values
        insignificant_pvals = P_pivot.where(~S_pivot)
        significant_pvals = P_pivot.where(S_pivot)
        
        print_and_flush("Data prepared for heatmap plotting.", log_file)
    except Exception as e:
        print_and_flush(f"Error preparing data for heatmap: {e}", log_file)
        insignificant_pvals = pd.DataFrame()
        significant_pvals = pd.DataFrame()
    
    # ----------------------------
    # 5. Generate Plots
    # ----------------------------
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # ------------------------------------------------------
    # Plot 1: F-statistic Map
    # ------------------------------------------------------
    print_and_flush("Generating F-statistic Map...", log_file)
    try:
        plt.figure(figsize=(12, 8))
        plt.title("F-statistic Map")
        # Pivot the F_statistic data and sort latitudes descending
        F_pivot = anova_df.pivot_table(index='lat', columns='lon', values='F_statistic', aggfunc='mean')
        F_pivot = F_pivot.sort_index(ascending=False)  # Sort latitudes descending
        sns.heatmap(F_pivot, cmap="jet_r", cbar_kws={'label': 'F-statistic', 'orientation': 'vertical'})
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])
        plt.yticks([])
        f_stat_map_path = os.path.join(plots_folder, "anova_f_statistic_map.png")
        plt.savefig(f_stat_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"F-statistic map saved at {f_stat_map_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating F-statistic Map: {e}", log_file)
    
    # ------------------------------------------------------
    # Alternative Plot 2: p-value Map with pcolormesh and heatmap
    # ------------------------------------------------------
    print_and_flush("Generating alternative p-value Map with pcolormesh and heatmap...", log_file)
    try:
        # ----------------------------
        # Using pcolormesh
        # ----------------------------
        plt.figure(figsize=(12, 8))
        plt.title("ANOVA p-value Map with pcolormesh")
        
        # Extract sorted unique latitudes and longitudes
        latitudes = P_pivot.index.values
        longitudes = P_pivot.columns.values
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Convert pivot tables to numpy arrays
        p_values = P_pivot.values
        significant = S_pivot.values
        
        # Mask insignificant p-values
        masked_pvals = np.ma.masked_where(~significant, p_values)
        
        # Plot significant p-values with 'viridis'
        plt.pcolormesh(lon_grid, lat_grid, masked_pvals, cmap='viridis', shading='auto', vmin=0, vmax=0.05)
        
        # Overlay insignificant grids in grey
        plt.pcolormesh(lon_grid, lat_grid, ~significant, cmap='Greys', shading='auto')
        
        # Add horizontal color bar
        cbar = plt.colorbar(orientation='horizontal', pad=0.2)
        cbar.set_label('p-value')
        
        # Remove axis labels and ticks
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])
        plt.yticks([])
        
        # Save pcolormesh plot
        pcolormesh_path = os.path.join(plots_folder, "anova_p_value_pcolormesh.png")
        plt.savefig(pcolormesh_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Pcolormesh p-value map saved at {pcolormesh_path}", log_file)
        
        # ----------------------------
        # Using heatmap
        # ----------------------------
        plt.figure(figsize=(12, 8))
        plt.title("ANOVA p-value Heatmap with heatmap")
        
        # Plot insignificant grids in grey
        sns.heatmap(
            insignificant_pvals,
            cmap='Greys',
            cbar=False,
            linewidths=0.05,
            linecolor='white',
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        
        # Overlay significant p-values with 'viridis' colormap
        sns.heatmap(
            significant_pvals,
            cmap='viridis',
            vmin=0,
            vmax=0.05,
            cbar_kws={'label': 'p-value', 'orientation': 'horizontal'},
            linewidths=0.05,
            linecolor='white',
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        
        # Save heatmap plot
        heatmap_path = os.path.join(plots_folder, "anova_p_value_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Heatmap p-value map saved at {heatmap_path}", log_file)
        
    except Exception as e:
        print_and_flush(f"Error generating alternative p-value Map: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 3: Scatter Plot of p-values Across Grids
    # ------------------------------------------------------
    print_and_flush("Generating Scatter Plot of p-values across all grids...", log_file)
    try:
        plt.figure(figsize=(12, 8))
        plt.title("Scatter Plot of ANOVA p-values Across Grids")
    
        # Scatter plot: longitude vs latitude, colored by p-value
        scatter = plt.scatter(
            anova_df['lon'], 
            anova_df['lat'], 
            c=anova_df['P_value'], 
            cmap='viridis',
            marker='s', 
            edgecolor='none',
            alpha=0.7
        )
    
        # Color bar with vertical orientation
        cbar = plt.colorbar(scatter, orientation='vertical')
        cbar.set_label('p-value')
    
        # Significance threshold line (optional)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.axhline(0, color='grey', linewidth=0.5)
    
        # Labels and aesthetics
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
    
        # Save the scatter plot
        p_val_scatter_path = os.path.join(plots_folder, "anova_p_value_scatter.png")
        plt.savefig(p_val_scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Scatter plot of p-values saved at {p_val_scatter_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating Scatter Plot: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 4: Significant Grids Map
    # ------------------------------------------------------
    print_and_flush("Generating Significant Grids Map...", log_file)
    try:
        plt.figure(figsize=(12, 8))
        plt.title("Map of Significant Grids (Blue: Significant, Red: Not Significant)")
    
        # Define colors based on significance
        colors = anova_df['significant'].map({True: 'blue', False: 'red'})
    
        # Scatter plot: longitude vs latitude, colored by significance
        plt.scatter(
            anova_df['lon'], 
            anova_df['lat'], 
            c=colors, 
            marker='s', 
            edgecolor='none',
            alpha=0.7
        )
    
        # Create custom legend
        legend_elements = [
            Patch(facecolor='blue', edgecolor='blue', label='Significant (p < 0.05)'),
            Patch(facecolor='red', edgecolor='red', label='Not Significant (p >= 0.05)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
        # Labels and aesthetics
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
    
        # Save the significant grids map
        signif_map_path = os.path.join(plots_folder, "anova_significant_grids_map.png")
        plt.savefig(signif_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Significant grids map saved at {signif_map_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating Significant Grids Map: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 5: Histogram of p-values
    # ------------------------------------------------------
    print_and_flush("Generating Histogram of p-values...", log_file)
    try:
        plt.figure(figsize=(10, 6))
    
        # Extract all p-values
        all_pvalues = anova_df['P_value'].dropna().values
    
        # Plot histogram with KDE
        sns.histplot(all_pvalues, bins=50, kde=True, color="lightgreen")
    
        # Add a vertical line at p = 0.05
        plt.axvline(0.05, color='red', linestyle='--', label='Significance Threshold (p = 0.05)')
    
        # Titles and labels
        plt.title("Histogram of ANOVA p-values")
        plt.xlabel("p-value")
        plt.ylabel("Frequency")
        plt.legend()
    
        # Save the histogram
        p_val_hist_path = os.path.join(plots_folder, "anova_p_value_histogram.png")
        plt.savefig(p_val_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Histogram of p-values saved at {p_val_hist_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating Histogram of p-values: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 6: Joint Plot of F-statistic vs. p-value
    # ------------------------------------------------------
    print_and_flush("Generating Joint Plot of F-statistic vs. p-value...", log_file)
    try:
        # Ensure 'significant' is a boolean
        anova_df['significant'] = anova_df['significant'].astype(bool)
    
        # Joint plot
        joint_plot = sns.jointplot(
            x='F_statistic', 
            y='P_value', 
            data=anova_df, 
            kind='scatter',
            alpha=0.5,
            color='purple'
        )
    
        joint_plot.fig.suptitle("Joint Plot of F-statistic vs. p-value", y=1.02)
        joint_plot.set_axis_labels("F-statistic", "p-value")
    
        # Save the joint plot
        joint_plot_path = os.path.join(plots_folder, "anova_f_p_joint_plot.png")
        joint_plot.savefig(joint_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Joint plot of F-statistic vs. p-value saved at {joint_plot_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating Joint Plot: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 7: Boxplot of F-statistic by Significance
    # ------------------------------------------------------
    print_and_flush("Generating Boxplot of F-statistic by Significance...", log_file)
    try:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='significant', y='F_statistic', data=anova_df, palette="Set2")
        plt.title("Boxplot of F-statistics by Significance")
        plt.xlabel("Significant (p < 0.05)")
        plt.ylabel("F-statistic")
        boxplot_path = os.path.join(plots_folder, "anova_f_statistic_boxplot.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Boxplot of F-statistics saved at {boxplot_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating Boxplot of F-statistics: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 8: Histogram of F-statistic (1st-99th percentile)
    # ------------------------------------------------------
    print_and_flush("Generating Histogram of F-statistics (1st-99th percentile)...", log_file)
    try:
        plt.figure(figsize=(10, 6))
    
        # Flatten F-statistics and remove NaNs
        f_stats = F_pivot.values.flatten()
        f_stats = f_stats[~np.isnan(f_stats)]
    
        # Remove outliers: define e.g., 1st and 99th percentiles
        lower_bound = np.percentile(f_stats, 1)
        upper_bound = np.percentile(f_stats, 99)
        f_stats_trimmed = f_stats[(f_stats >= lower_bound) & (f_stats <= upper_bound)]
    
        sns.histplot(f_stats_trimmed, bins=50, kde=True, color="skyblue")
        plt.title("Histogram of ANOVA F-statistics (1st-99th percentile)")
        plt.xlabel("F-statistic")
        plt.ylabel("Frequency")
        hist_path = os.path.join(plots_folder, "anova_f_statistic_histogram.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_and_flush(f"Histogram of F-statistics saved at {hist_path}", log_file)
    except Exception as e:
        print_and_flush(f"Error generating Histogram of F-statistics: {e}", log_file)
    
    # ------------------------------------------------------
    # Plot 9: Groundwater Heads Histogram (if merged_df is not empty)
    # ------------------------------------------------------
    if not merged_df.empty:
        print_and_flush("Generating Groundwater Heads Histogram with specified bins...", log_file)
        try:
            # Define bins and labels: <0, 0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60, >=60
            bins = [-np.inf, 0, 5, 10, 20, 30, 40, 50, 60, np.inf]
            bin_labels = ['<0', '0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60', '>=60']
            merged_df['head_bin'] = pd.cut(merged_df['head'], bins=bins, labels=bin_labels, right=False)
    
            # Remove any bins with NaN (optional)
            merged_df = merged_df.dropna(subset=['head_bin'])
    
            # Define color palette using 'jet' with as many colors as bins
            num_bins = len(bin_labels)
            cmap_jet = plt.get_cmap('jet', num_bins)
            colors = cmap_jet(range(num_bins))
    
            # Create a dictionary for bin colors
            bin_color_dict = dict(zip(bin_labels, colors))
    
            # Plot histogram
            plt.figure(figsize=(12, 8))
            plt.title("Histogram of ANOVA p-values by Groundwater Head Bins")
    
            for bin_label in bin_labels:
                subset = merged_df[merged_df['head_bin'] == bin_label]
                if not subset.empty:
                    sns.histplot(
                        subset['P_value'], 
                        bins=50, 
                        kde=False, 
                        color=bin_color_dict[bin_label], 
                        label=bin_label, 
                        alpha=0.6
                    )
    
            # Add a vertical line at p = 0.05
            plt.axvline(0.05, color='black', linestyle='--', label='Significance Threshold (p = 0.05)')
    
            # Titles and labels
            plt.title("Histogram of ANOVA p-values by Groundwater Head Bins")
            plt.xlabel("p-value")
            plt.ylabel("Frequency")
            plt.legend(title="Head Bins", bbox_to_anchor=(1.05, 1), loc='upper left')
    
            # Tight layout to accommodate legend
            plt.tight_layout()
    
            # Save the groundwater heads histogram
            gw_hist_path = os.path.join(plots_folder, "anova_p_value_groundwater_heads_histogram.png")
            plt.savefig(gw_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            print_and_flush(f"Groundwater heads histogram saved at {gw_hist_path}", log_file)
        except Exception as e:
            print_and_flush(f"Error generating Groundwater Heads Histogram: {e}", log_file)
    else:
        print_and_flush("Merged Groundwater Heads data is empty. Skipping Groundwater Heads Histogram.", log_file)
    
    # ------------------------------------------------------
    # Completion Message with Execution Time
    # ------------------------------------------------------
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print_and_flush(f"ANOVA plotting complete. Total execution time: {int(minutes)} minutes {int(seconds)} seconds.", log_file)

if __name__ == "__main__":
    # Configuration
    data_folder = "/scratch-shared/otoo0001/GLOBGDES/Paper2_mapping_wetlands/groundwater_enso_analysis_v2025_01_09/all_v_2025_01_09"
    plots_folder = "/scratch-shared/otoo0001/GLOBGDES/Paper2_mapping_wetlands/groundwater_enso_analysis_v2025_01_09/all_v_2025_01_09/plots_new/"
    groundwater_heads_nc = "/scratch-shared/otoo0001/data/Jarno_dataset/transient_Gw_top_30arcsec_monthly_1958-2015_Australia_avg.nc"
    groundwater_variable = "groundwater_heads_top"
    
    # Run the plotting function
    plot_anova_results(data_folder, plots_folder, groundwater_heads_nc, groundwater_variable)
