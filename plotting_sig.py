
"""
Comprehensive Script for Groundwater and ENSO Analysis
=====================================================

This script performs the following tasks:
1. Dynamically loads and merges significance CSV files.
2. Incorporates Groundwater Head data from a NetCDF file.
3. Prepares and categorizes data for analysis.
4. Generates various plots, including histograms categorized by Groundwater Head.

Author: [Your Name]
Date: [Today's Date]
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import xarray as xr

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------

# Output Directory for Plots and Logs
output_folder = "/gpfs/scratch1/shared/otoo0001/GLOBGDES/Paper2_mapping_wetlands/groundwater_enso_analysis_v2025_01_09/all_v_2025_01_09/plots_new_sig_final_meeting/"
os.makedirs(output_folder, exist_ok=True)

# Path to Merged CSV (if it exists)
merged_csv_path = os.path.join(output_folder, "merged_significance_groundwater.csv")

# Path to Combined CSV (final_significance_combined.csv)
combined_csv_path = "/gpfs/scratch1/shared/otoo0001/GLOBGDES/Paper2_mapping_wetlands/groundwater_enso_analysis_v2025_01_09/all_v_2025_01_09/plots_new_sig_new/final_significance_combined.csv"

# Directory containing Single CSV Files (Assuming they are in 'plots_new_sig_new' directory)
single_csv_directory = "/gpfs/scratch1/shared/otoo0001/GLOBGDES/Paper2_mapping_wetlands/groundwater_enso_analysis_v2025_01_09/all_v_2025_01_09/plots_new_sig_new/"

# Groundwater Heads Data for Plotting (NetCDF File)
groundwater_heads_nc = "/scratch-shared/otoo0001/data/Jarno_dataset/transient_Gw_top_30arcsec_monthly_1958-2015_Australia_avg.nc"
groundwater_variable = "groundwater_heads_top"

# -----------------------------
# Logging Function
# -----------------------------
def log_message(msg, level='info'):
    """
    Logs a message to both the console and a log file.

    Parameters:
        msg (str): The message to log.
        level (str): The logging level ('info', 'warning', 'error').
    """
    log_file = os.path.join(output_folder, "processing_log.txt")
    timestamp = pd.Timestamp.now()
    if level == 'info':
        print(f"[INFO] {timestamp}: {msg}")
    elif level == 'warning':
        print(f"[WARNING] {timestamp}: {msg}")
    elif level == 'error':
        print(f"[ERROR] {timestamp}: {msg}")
    with open(log_file, "a") as f:
        f.write(f"{timestamp}: [{level.upper()}] {msg}\n")

# -----------------------------
# Data Loading Function
# -----------------------------
def load_data(merged_path, combined_path, single_dir, groundwater_nc, groundwater_var):
    """
    Loads the merged CSV if it exists; else, loads the combined CSV if it exists;
    else, loads and merges single CSVs. Incorporates Groundwater Head data from a NetCDF file.

    Parameters:
        merged_path (str): Path to the merged CSV.
        combined_path (str): Path to the combined (final_significance_combined.csv) CSV.
        single_dir (str): Directory containing single CSV files.
        groundwater_nc (str): Path to the Groundwater Heads NetCDF file.
        groundwater_var (str): Variable name for Groundwater Heads in NetCDF.

    Returns:
        pd.DataFrame: The merged DataFrame containing all necessary data.
    """
    # Check if merged CSV exists
    if os.path.exists(merged_path):
        log_message(f"Merged CSV found at {merged_path}. Loading merged data...", level='info')
        merged_df = pd.read_csv(merged_path)
        log_message(f"Merged data loaded with shape: {merged_df.shape}", level='info')
    elif os.path.exists(combined_path):
        log_message(f"Combined CSV found at {combined_path}. Loading combined data...", level='info')
        merged_df = pd.read_csv(combined_path)
        log_message(f"Combined data loaded with shape: {merged_df.shape}", level='info')
    else:
        log_message("Merged CSV not found. Attempting to load and merge single CSV files...", level='info')
        # Find all single CSV files in the specified directory
        csv_files = glob.glob(os.path.join(single_dir, "*.csv"))
        # Exclude the final_combined_csv if present in the directory
        csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(combined_path)]
        
        if not csv_files:
            log_message("No single CSV files found to merge. Exiting.", level='error')
            raise FileNotFoundError("No single CSV files found to merge.")
        
        log_message(f"Found {len(csv_files)} single CSV files to merge.", level='info')
        
        # Load all single CSVs into a list
        df_list = []
        for file in csv_files:
            try:
                temp_df = pd.read_csv(file)
                df_list.append(temp_df)
                log_message(f"Loaded {file} with shape: {temp_df.shape}", level='info')
            except Exception as e:
                log_message(f"Error loading {file}: {e}", level='error')
        
        # Concatenate all DataFrames
        if len(df_list) == 1:
            merged_df = df_list[0]
            log_message("Only one single CSV file found. Using it as merged data.", level='info')
        else:
            merged_df = pd.concat(df_list, ignore_index=True)
            log_message(f"All single CSV files merged with shape: {merged_df.shape}", level='info')
    
    # At this point, merged_df contains the main data (merged, combined, or merged single CSVs)
    # Now, ensure that Groundwater Head data is included by merging with NetCDF data
    # Only if 'Groundwater Head' is not already present

    if 'Groundwater Head' not in merged_df.columns:
        log_message("Groundwater Head data not found in main data. Attempting to merge with NetCDF data...", level='info')
        if os.path.exists(groundwater_nc):
            try:
                log_message(f"Loading Groundwater Head data from NetCDF: {groundwater_nc}", level='info')
                gw_ds = xr.open_dataset(groundwater_nc)
                gw_df = gw_ds[[groundwater_var]].to_dataframe().reset_index()
                gw_df = gw_df.rename(columns={groundwater_var: 'Groundwater Head'})
                log_message(f"Groundwater Head data loaded with shape: {gw_df.shape}", level='info')
                
                # Merge with main data on 'lat' and 'lon'
                log_message("Merging main data with Groundwater Head data on 'lat' and 'lon'...", level='info')
                merged_df = pd.merge(merged_df, gw_df, on=['lat', 'lon'], how='inner')
                log_message(f"Merged DataFrame shape after merging with Groundwater Head: {merged_df.shape}", level='info')
            except Exception as e:
                log_message(f"Error merging with Groundwater Head data: {e}", level='error')
                raise e
        else:
            log_message(f"Groundwater Heads NetCDF file not found at {groundwater_nc}. Exiting.", level='error')
            raise FileNotFoundError(f"Groundwater Heads NetCDF file not found at {groundwater_nc}")
    else:
        log_message("Groundwater Head data already present in main data. Proceeding...", level='info')
    
    return merged_df

# -----------------------------
# Plotting Functions
# -----------------------------
def plot_pcolormesh(pivot_data, title, cmap, label, filename):
    """
    Creates and saves a pcolormesh plot without axis labels and ticks.

    Parameters:
        pivot_data (pd.DataFrame): Pivoted data for plotting.
        title (str): Title of the plot.
        cmap (str or Colormap): Colormap to use.
        label (str): Label for the colorbar.
        filename (str): Path to save the plot.
    """
    if pivot_data.empty:
        log_message(f"Warning: No data available for {title}. Skipping plot.", level='warning')
        return
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)
    mesh = plt.pcolormesh(
        pivot_data.columns.values,
        pivot_data.index.values,
        pivot_data.values,
        cmap=cmap,
        shading='auto'
    )
    cbar = plt.colorbar(mesh, label=label)
    # Turn off axis labels and ticks
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    log_message(f"{title} saved to: {filename}", level='info')

def plot_pcolormesh_with_legend(pivot_data, title, cmap, legend_labels, legend_title, filename):
    """
    Creates and saves a pcolormesh plot with a custom legend instead of a colorbar,
    without axis labels and ticks.

    Parameters:
        pivot_data (pd.DataFrame): Pivoted data for plotting.
        title (str): Title of the plot.
        cmap (str or Colormap): Colormap to use.
        legend_labels (dict): Mapping of colors to labels for the legend.
        legend_title (str): Title for the legend.
        filename (str): Path to save the plot.
    """
    if pivot_data.empty:
        log_message(f"Warning: No data available for {title}. Skipping plot.", level='warning')
        return
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)

    mesh = plt.pcolormesh(
        pivot_data.columns.values,
        pivot_data.index.values,
        pivot_data.values,
        cmap=cmap,
        shading='auto'
    )

    # Create legend patches
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in legend_labels.items()]

    # Add the legend
    plt.legend(
        handles=legend_patches,
        title=legend_title,
        loc="upper right",
        fontsize=12
    )

    # Turn off axis labels and ticks
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()
    log_message(f"{title} saved to: {filename}", level='info')

def plot_boxplot(df, x, y, title, xlabel, ylabel, palette, filename):
    """
    Creates and saves a boxplot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        palette (dict): Color palette mapping.
        filename (str): Path to save the plot.
    """
    if df.empty:
        log_message(f"Warning: No data available for {title}. Skipping plot.", level='warning')
        return
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)
    sns.boxplot(data=df, x=x, y=y, palette=palette)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    log_message(f"{title} saved to: {filename}", level='info')

def plot_category_histogram(df, category_col, variable_col, bins, bin_labels, 
                           cmap_name, title, xlabel, ylabel, 
                           vertical_line=None, vertical_label=None, 
                           filename=None):
    """
    Creates and saves a category histogram for a specified variable.

    Parameters:
        df (pd.DataFrame): The data containing the variables.
        category_col (str): The column name to categorize by.
        variable_col (str): The column name of the variable to plot.
        bins (list): List of bin edges.
        bin_labels (list): List of bin labels.
        cmap_name (str): Name of the matplotlib colormap to use.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        vertical_line (float, optional): x-value to draw a vertical line.
        vertical_label (str, optional): Label for the vertical line.
        filename (str, optional): Path to save the plot.
    """
        # Categorize the data
    df['head_bin'] = pd.cut(df[category_col], bins=bins, labels=bin_labels, right=False)
    
    # Drop any rows with NaN in 'head_bin'
    df = df.dropna(subset=['head_bin'])
    
    # Define color palette
    num_bins = len(bin_labels)
    cmap = plt.get_cmap(cmap_name, num_bins)
    colors = cmap(range(num_bins))
    bin_color_dict = dict(zip(bin_labels, colors))
    
    # Initialize the plot
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)
    
    # Plot histogram for each bin category
    for bin_label in bin_labels:
        subset = df[df['head_bin'] == bin_label]
        if not subset.empty:
            sns.histplot(
                subset[variable_col], 
                bins=50, 
                kde=False, 
                color=bin_color_dict[bin_label], 
                label=bin_label, 
                alpha=0.6,
                edgecolor='black'  # Adds black edges to bins for clarity
            )
    
    # Add vertical line if specified
    if vertical_line is not None:
        plt.axvline(vertical_line, color='red', linestyle='--', label=vertical_label)
    
    # Set labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Create legend outside the plot
    plt.legend(title="Groundwater Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save the plot
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        log_message(f"{title} saved to: {filename}", level='info')
    plt.close()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """
    Main function to orchestrate data loading, preparation, and plotting.
    """
    log_message("Script started.", level='info')
    
    # Step 1: Load Data
    try:
        merged_df = load_data(
            merged_path=merged_csv_path,
            combined_path=combined_csv_path,
            single_dir=single_csv_directory,
            groundwater_nc=groundwater_heads_nc,
            groundwater_var=groundwater_variable
        )
    except Exception as e:
        log_message(f"Data loading failed: {e}", level='error')
        return
    
    # Step 2: Data Preparation
    required_columns = [
        'lat', 'lon', 'R_squared', 'Cross_Correlation', 'Max_Lag',
        'Granger_Significant', 'phase_winner', 'Groundwater Head'
    ]
    
    # Ensure all required columns are present
    missing_columns = set(required_columns) - set(merged_df.columns)
    if missing_columns:
        log_message(f"Missing required columns after loading data: {missing_columns}", level='error')
        return
    
    # Map 'phase_winner' to numerical codes
    phase_map = {"El Niño": 0, "La Niña": 1}
    merged_df['phase_code'] = merged_df['phase_winner'].map(phase_map)
    
    # Convert 'Granger_Significant' to integers (if not already)
    merged_df['Granger_Significant'] = merged_df['Granger_Significant'].astype(int)
    
    # Drop rows with NaNs in critical columns
    merged_df_clean = merged_df.dropna(subset=required_columns + ['phase_code'])
    log_message(f"Cleaned data shape after dropping NaNs: {merged_df_clean.shape}", level='info')
    
    # Renaming Columns for Plotting
    label_mapping = {
        'R_squared': 'R Squared',
        'Cross_Correlation': 'Cross Correlation',
        'Max_Lag': 'Max Lag (Months)',
        'Granger_Significant': 'Granger Significance',
        'phase_winner': 'Phase Winner',
        'lat': 'Latitude',
        'lon': 'Longitude'
    }
    
    # Rename columns
    merged_df_plot = merged_df_clean.rename(columns=label_mapping)
    
    # Define Groundwater Head Categories
    bins = [-np.inf, 0, 5, 10, 20, 30, 40, 50, 60, np.inf]
    bin_labels = ['<0', '0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60', '>=60']
    
    # Categorize Groundwater Head
    merged_df_plot['Groundwater_Category'] = pd.cut(
        merged_df_plot['Groundwater Head'],
        bins=bins,
        labels=bin_labels,
        right=False
    )
    
    # Verify categorization
    sample_categorization = merged_df_plot[['Groundwater Head', 'Groundwater_Category']].head()
    log_message(f"Sample of Groundwater Head Categorization:\n{sample_categorization}", level='info')
    
    # Check for NaN in Groundwater_Category
    nan_categories = merged_df_plot['Groundwater_Category'].isna().sum()
    log_message(f"Number of NaN Groundwater Categories: {nan_categories}", level='info')
    
    # Optionally, drop NaNs in Groundwater_Category
    if nan_categories > 0:
        merged_df_plot = merged_df_plot.dropna(subset=['Groundwater_Category'])
        log_message(f"DataFrame shape after dropping NaN Groundwater Categories: {merged_df_plot.shape}", level='info')
    
    # -----------------------------
    # Plotting
    # -----------------------------
    log_message("Starting plotting process...", level='info')
    
    # 1. Phase Winner Map with Legend
    log_message("Creating Phase Winner Map with Legend...", level='info')
    phase_grid = merged_df_plot.pivot_table(
        index='Latitude', columns='Longitude', values='phase_code'
    ).sort_index(ascending=False)
    
    if phase_grid.empty:
        log_message("No data available for Phase Winner Map. Skipping plot.", level='warning')
    else:
        # Define legend labels
        legend_labels_phase = {
            'red': 'El Niño',
            'blue': 'La Niña'
        }
    
        # Create the pcolormesh plot with legend
        plot_pcolormesh_with_legend(
            pivot_data=phase_grid,
            title="Phase Winners",
            cmap=ListedColormap(['red', 'blue']),
            legend_labels=legend_labels_phase,
            legend_title="Phase Winner",
            filename=os.path.join(output_folder, "Phase_Winners_with_Legend.png")
        )
    
    # 2. R Squared Map
    log_message("Creating R Squared Map...", level='info')
    r2_grid = merged_df_plot.pivot_table(
        index='Latitude', columns='Longitude', values='R Squared'
    ).sort_index(ascending=False)
    plot_pcolormesh(
        pivot_data=r2_grid,
        title="R Squared (Coefficient of Determination)",
        cmap="RdBu_r",
        label="R Squared",
        filename=os.path.join(output_folder, "R_squared.png")
    )
    
    # 3. Cross Correlation Map
    log_message("Creating Cross Correlation Map...", level='info')
    cross_corr_grid = merged_df_plot.pivot_table(
        index='Latitude', columns='Longitude', values='Cross Correlation'
    ).sort_index(ascending=False)
    plot_pcolormesh(
        pivot_data=cross_corr_grid,
        title="Cross Correlation",
        cmap="RdBu_r",
        label="Cross Correlation",
        filename=os.path.join(output_folder, "Cross_Correlation.png")
    )
    
    # 4. Max Lag Map
    log_message("Creating Max Lag (Months) Map...", level='info')
    max_lag_grid = merged_df_plot.pivot_table(
        index='Latitude', columns='Longitude', values='Max Lag (Months)'
    ).sort_index(ascending=False)
    plot_pcolormesh(
        pivot_data=max_lag_grid,
        title="Maximum Lag (Months)",
        cmap="jet",
        label="Max Lag (Months)",
        filename=os.path.join(output_folder, "Max_Lag.png")
    )
    
    # 5. Granger Significance Map with Legend
    log_message("Creating Granger Significance Map with Legend...", level='info')
    granger_grid = merged_df_plot.pivot_table(
        index='Latitude', columns='Longitude', values='Granger Significance'
    ).sort_index(ascending=False)
    
    if granger_grid.empty:
        log_message("No data available for Granger Significance Map. Skipping plot.", level='warning')
    else:
        # Define legend labels for Granger Significance
        legend_labels_granger = {
            'blue': 'False',
            'red': 'True'
        }
    
        # Create the pcolormesh plot with legend
        plot_pcolormesh_with_legend(
            pivot_data=granger_grid,
            title="Granger Causality Significance",
            cmap=ListedColormap(['blue', 'red']),
            legend_labels=legend_labels_granger,
            legend_title="Granger Significance",
            filename=os.path.join(output_folder, "Granger_Significance_with_Legend.png")
        )
    
    # 6. Box Plots
    log_message("Creating Box Plots...", level='info')
    phase_palette = {"El Niño": "red", "La Niña": "blue"}
    
    # R Squared by ENSO Phase
    plot_boxplot(
        df=merged_df_plot, 
        x="Phase Winner", 
        y="R Squared",
        title="R Squared by ENSO Phase", 
        xlabel="ENSO Phase", 
        ylabel="R Squared",
        palette=phase_palette, 
        filename=os.path.join(output_folder, "R_squared_Boxplot.png")
    )
    
    # Cross Correlation by ENSO Phase
    plot_boxplot(
        df=merged_df_plot, 
        x="Phase Winner", 
        y="Cross Correlation",
        title="Cross Correlation by ENSO Phase", 
        xlabel="ENSO Phase", 
        ylabel="Cross Correlation",
        palette=phase_palette, 
        filename=os.path.join(output_folder, "Cross_Correlation_Boxplot.png")
    )
    
    # Max Lag (Months) by ENSO Phase
    plot_boxplot(
        df=merged_df_plot, 
        x="Phase Winner", 
        y="Max Lag (Months)",
        title="Max Lag (Months) by ENSO Phase", 
        xlabel="ENSO Phase", 
        ylabel="Max Lag (Months)",
        palette=phase_palette, 
        filename=os.path.join(output_folder, "Max_Lag_Boxplot.png")
    )
    
    # 7. Pair Plots
    log_message("Creating Pair Plots...", level='info')
    # Define the variables for the pair plot using the new labels
    pairplot_vars_renamed = ['R Squared', 'Cross Correlation', 'Max Lag (Months)']
    
    # Pair Plot with Granger Significance as hue
    log_message("Creating Pair Plot with Granger Significance as Hue...", level='info')
    pairplot_granger = sns.pairplot(
        merged_df_plot,
        vars=pairplot_vars_renamed,
        hue="Granger Significance",
        palette={0: '#17becf', 1: '#d62728'},  # Blue and Red
        diag_kind="kde",
        plot_kws={'alpha': 0.6}  # Transparency to handle overplotting
    )
    
    # Adjust the title and layout
    pairplot_granger.fig.suptitle("Pair Plot: R Squared, Cross Correlation, Max Lag", fontsize=16, y=1.02)
    pairplot_granger.fig.tight_layout()
    
    # Remove the legend from individual subplots
    pairplot_granger._legend.remove()
    
    # Create a single legend outside the plots
    handles_granger = [
        mpatches.Patch(color='#17becf', label='Not Significant'),
        mpatches.Patch(color='#d62728', label='Significant')
    ]
    plt.legend(
        handles=handles_granger,
        title="Granger Significance",
        loc='upper left',
        bbox_to_anchor=(1,1),
        ncol=2,
        frameon=False,
        fontsize=12
    )
    
    # Save the pair plot
    pairplot_granger_path = os.path.join(output_folder, "Pair_Plot_Granger.png")
    plt.savefig(pairplot_granger_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Pair plot with Granger Significance saved to: {pairplot_granger_path}", level='info')
    
    # Pair Plot with Phase Winner as hue
    log_message("Creating Pair Plot with Phase Winner as Hue...", level='info')
    pairplot_phase = sns.pairplot(
        merged_df_plot,
        vars=pairplot_vars_renamed,
        hue="Phase Winner",
        palette={"El Niño": "#d62728", "La Niña": "#17becf"},  # Red and Cyan
        diag_kind="kde",
        plot_kws={'alpha': 0.6}  # Transparency to handle overplotting
    )
    
    # Adjust the title and layout
    pairplot_phase.fig.suptitle("", fontsize=16, y=1.02)
    pairplot_phase.fig.tight_layout()
    
    # Remove the legend from individual subplots
    pairplot_phase._legend.remove()
    
    # Create a single legend outside the plots
    handles_phase = [
        mpatches.Patch(color="#d62728", label='El Niño'),
        mpatches.Patch(color="#17becf", label='La Niña')
    ]
    plt.legend(
        handles=handles_phase,
        title="ENSO Phase",
        loc='upper left',
        bbox_to_anchor=(1,1),
        ncol=2,
        frameon=False,
        fontsize=12
    )
    
    # Save the pair plot
    pairplot_phase_path = os.path.join(output_folder, "Pair_Plot_Phase_Winner.png")
    plt.savefig(pairplot_phase_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Pair plot with Phase Winner saved to: {pairplot_phase_path}", level='info')
    
    # 8. Histograms for Max Lag and Cross Correlation Categorized by Groundwater Head
    log_message("Creating Histograms for Max Lag and Cross Correlation Categorized by Groundwater Head...", level='info')
    
    # Define plot parameters for Max Lag
    max_lag_title = "Histogram of Max Lag (Months) Categorized by Groundwater Head"
    max_lag_xlabel = "Max Lag (Months)"
    max_lag_ylabel = "Frequency"
    max_lag_filename = os.path.join(output_folder, "Max_Lag_Categorized_Histogram.png")
    
    # Plot Histogram for Max Lag
    plot_category_histogram(
        df=merged_df_plot,
        category_col='Groundwater Head',
        variable_col='Max Lag (Months)',
        bins=bins,
        bin_labels=bin_labels,
        cmap_name='viridis',  # Perceptually uniform colormap
        title=max_lag_title,
        xlabel=max_lag_xlabel,
        ylabel=max_lag_ylabel,
        vertical_line=0,  # Reference line at zero lag
        vertical_label="Zero Lag",
        filename=max_lag_filename
    )
    
    # Define plot parameters for Cross Correlation
    cross_corr_title = "Histogram of Cross Correlation Categorized by Groundwater Head"
    cross_corr_xlabel = "Cross Correlation"
    cross_corr_ylabel = "Frequency"
    cross_corr_filename = os.path.join(output_folder, "Cross_Correlation_Categorized_Histogram.png")
    
    # Plot Histogram for Cross Correlation
    plot_category_histogram(
        df=merged_df_plot,
        category_col='Groundwater Head',
        variable_col='Cross Correlation',
        bins=bins,
        bin_labels=bin_labels,
        cmap_name='plasma',  # Alternatively, 'viridis' or 'jet'
        title=cross_corr_title,
        xlabel=cross_corr_xlabel,
        ylabel=cross_corr_ylabel,
        vertical_line=0,  # Reference line at zero correlation
        vertical_label="Zero Correlation",
        filename=cross_corr_filename
    )
    
    log_message("All plots created successfully.", level='info')

# -----------------------------
# Execution Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
