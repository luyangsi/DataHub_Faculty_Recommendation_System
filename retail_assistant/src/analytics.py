"""
Sales and inventory analytics utilities.
"""
import pandas as pd
import numpy as np


def compute_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    if 'category' not in df.columns:
        return pd.DataFrame()
    cols = {c: (c, 'sum') for c in ['revenue', 'quantity', 'profit'] if c in df.columns}
    cols['orders'] = ('quantity', 'count')
    agg = df.groupby('category').agg(**cols).reset_index()
    agg['revenue_pct'] = agg['revenue'] / agg['revenue'].sum() * 100
    if 'profit' in agg.columns and 'revenue' in agg.columns:
        agg['profit_margin_pct'] = agg['profit'] / agg['revenue'] * 100
    return agg.sort_values('revenue', ascending=False)


def compute_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    cols = {c: (c, 'sum') for c in ['revenue', 'quantity', 'profit'] if c in df.columns}
    return df.groupby('month').agg(**cols).reset_index().sort_values('month')


def compute_weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time).dt.strftime('%Y-%m-%d')
    return df.groupby('week').agg(revenue=('revenue', 'sum'), quantity=('quantity', 'sum')).reset_index().sort_values('week')


def find_top_products(df: pd.DataFrame, n: int = 10, by: str = 'revenue') -> pd.DataFrame:
    if 'product_name' not in df.columns:
        return pd.DataFrame()
    cols = {c: (c, 'sum') for c in ['revenue', 'quantity', 'profit'] if c in df.columns}
    return (
        df.groupby(['product_id', 'product_name', 'category']).agg(**cols)
        .reset_index()
        .sort_values(by, ascending=False)
        .head(n)
    )


def find_slow_movers(df: pd.DataFrame, bottom_pct: float = 0.2) -> pd.DataFrame:
    if 'product_name' not in df.columns or 'revenue' not in df.columns:
        return pd.DataFrame()
    product_rev = df.groupby(['product_id', 'product_name', 'category'])['revenue'].sum().reset_index()
    product_rev['revenue_pct'] = product_rev['revenue'] / product_rev['revenue'].sum() * 100
    cutoff = product_rev['revenue'].quantile(bottom_pct)
    return product_rev[product_rev['revenue'] <= cutoff].sort_values('revenue')


def compute_inventory_status(inventory_df: pd.DataFrame) -> pd.DataFrame:
    df = inventory_df.copy()

    def _status(row):
        if row['current_stock'] == 0:
            return '断货'
        if 'reorder_point' in row and row['current_stock'] <= row['reorder_point']:
            return '低库存'
        if 'max_stock' in row and row['current_stock'] >= row['max_stock']:
            return '库存过高'
        return '正常'

    df['status'] = df.apply(_status, axis=1)

    def _restock(row):
        if row['current_stock'] <= row.get('reorder_point', 0):
            return max(0, int(row.get('max_stock', row.get('reorder_point', 0) * 3) - row['current_stock']))
        return 0

    df['restock_qty'] = df.apply(_restock, axis=1)
    return df


def compute_reorder_suggestions(sales_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df is None or inventory_df is None:
        return pd.DataFrame()

    df = sales_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff = df['date'].max() - pd.Timedelta(days=30)
    last30 = df[df['date'] >= cutoff]

    velocity = (
        last30.groupby(['product_id', 'product_name', 'category'])
        .agg(qty_30d=('quantity', 'sum'))
        .reset_index()
    )
    velocity['avg_daily'] = velocity['qty_30d'] / 30

    inv_status = compute_inventory_status(inventory_df)
    inv_agg = inv_status.groupby(['product_id', 'product_name', 'category']).agg(
        total_stock=('current_stock', 'sum'),
        low_stock_skus=('status', lambda x: (x == '低库存').sum()),
        out_of_stock_skus=('status', lambda x: (x == '断货').sum()),
        min_reorder=('reorder_point', 'min') if 'reorder_point' in inv_status.columns else ('current_stock', 'min'),
        avg_lead_time=('lead_time_days', 'mean') if 'lead_time_days' in inv_status.columns else ('current_stock', 'count'),
        unit_cost=('unit_cost', 'first') if 'unit_cost' in inv_status.columns else ('current_stock', 'mean'),
        supplier=('supplier', lambda x: x.mode().iloc[0] if len(x) > 0 else '') if 'supplier' in inv_status.columns else ('current_stock', 'count'),
    ).reset_index()

    merged = inv_agg.merge(velocity, on=['product_id', 'product_name', 'category'], how='left')
    merged['avg_daily'] = merged['avg_daily'].fillna(0)
    merged['days_of_stock'] = merged.apply(
        lambda r: r['total_stock'] / r['avg_daily'] if r['avg_daily'] > 0 else 999, axis=1
    )

    PRIORITY_ORDER = {'🔴 紧急补货': 0, '🟡 建议补货': 1, '🟢 可考虑补货': 2, '⚪ 暂不需要': 3}

    def _priority(row):
        if row['out_of_stock_skus'] > 0:
            return '🔴 紧急补货'
        if row['low_stock_skus'] > 0 or row['days_of_stock'] < 14:
            return '🟡 建议补货'
        if row['days_of_stock'] < 30:
            return '🟢 可考虑补货'
        return '⚪ 暂不需要'

    merged['priority'] = merged.apply(_priority, axis=1)
    merged['suggested_qty'] = merged.apply(
        lambda r: max(0, int(r['avg_daily'] * 30 - r['total_stock'])) if r['days_of_stock'] < 30 else 0, axis=1
    )
    merged['estimated_cost'] = merged['suggested_qty'] * merged.get('unit_cost', pd.Series(0, index=merged.index))

    return merged.sort_values('priority', key=lambda x: x.map(PRIORITY_ORDER)).reset_index(drop=True)
