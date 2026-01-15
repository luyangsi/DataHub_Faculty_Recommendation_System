"""
Data utilities for cleaning and visualization
Reusable functions for notebooks and analysis scripts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def clean_text(s):
    """
    Clean text field: lowercase, strip whitespace, handle None.
    
    Args:
        s: Input string or None
    
    Returns:
        Cleaned string
    """
    if s is None or pd.isna(s):
        return ""
    return str(s).strip().lower()


def month_bucket(dt):
    """
    Convert datetime to month string (YYYY-MM).
    
    Args:
        dt: datetime object or parseable string
    
    Returns:
        String in format 'YYYY-MM' or None if invalid
    """
    if pd.isna(dt):
        return None
    
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return dt.strftime('%Y-%m')
    except:
        return None


def parse_date_column(df, col_name, new_col_name=None):
    """
    Parse a date column and optionally create a new column.
    
    Args:
        df: DataFrame
        col_name: Name of column to parse
        new_col_name: Optional name for new column (default: col_name)
    
    Returns:
        DataFrame with parsed date column
    """
    if new_col_name is None:
        new_col_name = col_name
    
    df[new_col_name] = pd.to_datetime(df[col_name], errors='coerce')
    return df


def standardize_categories(df, col_name, top_n=10, other_label='Other'):
    """
    Keep top N categories, group rest as 'Other'.
    
    Args:
        df: DataFrame
        col_name: Column name
        top_n: Number of top categories to keep
        other_label: Label for grouped categories
    
    Returns:
        Series with standardized categories
    """
    top_cats = df[col_name].value_counts().head(top_n).index
    return df[col_name].apply(lambda x: x if x in top_cats else other_label)


def save_fig(fig, filename, output_dir='reports/figs', dpi=300, bbox_inches='tight'):
    """
    Save matplotlib figure to standard output directory.
    
    Args:
        fig: matplotlib figure object
        filename: Output filename (with extension)
        output_dir: Output directory path
        dpi: Resolution
        bbox_inches: Bounding box setting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Saved: {filepath}")


def create_time_series_plot(df, date_col, value_col, title, ylabel, filename=None):
    """
    Create and optionally save a time series plot.
    
    Args:
        df: DataFrame
        date_col: Date column name
        value_col: Value column name
        title: Plot title
        ylabel: Y-axis label
        filename: Optional filename to save
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by date and plot
    time_series = df.groupby(date_col)[value_col].count()
    time_series.plot(ax=ax, linewidth=2, color='#2E86AB')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        save_fig(fig, filename)
    
    return fig


def create_bar_chart(df, col_name, title, top_n=10, filename=None):
    """
    Create and optionally save a bar chart of top categories.
    
    Args:
        df: DataFrame
        col_name: Column name to plot
        title: Plot title
        top_n: Number of top categories
        filename: Optional filename to save
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get top N and plot
    top_values = df[col_name].value_counts().head(top_n)
    top_values.plot(kind='barh', ax=ax, color='#A23B72')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if filename:
        save_fig(fig, filename)
    
    return fig


def get_summary_stats(df, numeric_cols=None):
    """
    Get summary statistics for numeric columns.
    
    Args:
        df: DataFrame
        numeric_cols: Optional list of numeric columns (default: all numeric)
    
    Returns:
        DataFrame with summary statistics
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    return df[numeric_cols].describe()


def handle_missing_values(df, strategy='drop', threshold=0.5):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame
        strategy: 'drop' or 'fill'
        threshold: For 'drop', fraction of non-null values required
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        # Drop columns with too many missing values
        min_count = int(threshold * len(df))
        df = df.dropna(thresh=min_count, axis=1)
        
        # Drop rows with any remaining missing values
        df = df.dropna()
    
    elif strategy == 'fill':
        # Fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df