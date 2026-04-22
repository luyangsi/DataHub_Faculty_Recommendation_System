"""
Sample data generation and CSV loading utilities.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO

CATEGORIES = ['上衣', '裤子', '裙子', '外套', '鞋类', '配件']
COLORS = ['黑色', '白色', '灰色', '蓝色', '红色']
SUPPLIERS = ['杭州优品服饰', '广州时尚批发', '上海品牌直供', '深圳新款工厂']

# (name, price, cost)
PRODUCTS = {
    '上衣': [('基础款白T恤', 89, 25), ('印花短袖', 129, 40), ('纯棉卫衣', 199, 65),
             ('针织毛衣', 259, 90), ('衬衫', 179, 55)],
    '裤子': [('直筒牛仔裤', 229, 75), ('休闲长裤', 189, 60), ('运动短裤', 99, 30),
             ('工装裤', 259, 85), ('哈伦裤', 199, 65)],
    '裙子': [('百褶短裙', 159, 50), ('连衣裙', 299, 95), ('半身长裙', 199, 65),
             ('碎花裙', 239, 78), ('吊带裙', 179, 55)],
    '外套': [('风衣', 399, 130), ('牛仔外套', 299, 95), ('棉服', 499, 165),
             ('皮衣', 599, 200), ('运动外套', 259, 85)],
    '鞋类': [('小白鞋', 299, 95), ('运动鞋', 399, 130), ('高跟鞋', 459, 150),
             ('踝靴', 519, 170), ('拖鞋', 89, 28)],
    '配件': [('棒球帽', 79, 22), ('围巾', 99, 30), ('皮带', 129, 40),
             ('帆布包', 159, 50), ('太阳镜', 199, 65)],
}

SIZES = {
    '上衣': ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
    '裤子': ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
    '裙子': ['XS', 'S', 'M', 'L', 'XL'],
    '外套': ['S', 'M', 'L', 'XL', 'XXL'],
    '鞋类': ['36', '37', '38', '39', '40', '41', '42'],
    '配件': ['均码'],
}

SIZE_WEIGHTS = {
    '上衣': [0.05, 0.15, 0.30, 0.28, 0.15, 0.07],
    '裤子': [0.05, 0.15, 0.30, 0.28, 0.15, 0.07],
    '裙子': [0.05, 0.20, 0.35, 0.28, 0.12],
    '外套': [0.10, 0.25, 0.35, 0.20, 0.10],
    '鞋类': [0.12, 0.18, 0.22, 0.22, 0.14, 0.08, 0.04],
    '配件': [1.0],
}

# Seasonal multipliers per category per season
SEASON_MULT = {
    'spring': {'上衣': 1.3, '裤子': 1.1, '裙子': 1.8, '外套': 0.7, '鞋类': 1.2, '配件': 1.1},
    'summer': {'上衣': 1.5, '裤子': 0.9, '裙子': 2.0, '外套': 0.3, '鞋类': 1.0, '配件': 1.2},
    'autumn': {'上衣': 1.2, '裤子': 1.4, '裙子': 0.8, '外套': 1.6, '鞋类': 1.3, '配件': 1.0},
    'winter': {'上衣': 0.8, '裤子': 1.1, '裙子': 0.4, '外套': 2.0, '鞋类': 1.3, '配件': 0.9},
}


def _season(month: int) -> str:
    if month in (3, 4, 5):
        return 'spring'
    if month in (6, 7, 8):
        return 'summer'
    if month in (9, 10, 11):
        return 'autumn'
    return 'winter'


def generate_sample_sales(months: int = 6) -> pd.DataFrame:
    """Generate realistic sample sales data for the past N months."""
    np.random.seed(42)
    rows = []

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=30 * months)

    product_list = []
    pid = 1
    for cat, prods in PRODUCTS.items():
        for name, price, cost in prods:
            product_list.append((f'P{pid:03d}', name, cat, price, cost))
            pid += 1

    current = start_date
    while current <= end_date:
        season = _season(current.month)
        day_mult = 1.8 if current.weekday() >= 5 else 1.0

        for pid_key, name, cat, price, cost in product_list:
            smult = SEASON_MULT[season].get(cat, 1.0)
            base = np.random.poisson(max(1, int(2 * smult * day_mult)))
            if base == 0:
                current += timedelta(days=0)
                continue

            sizes = SIZES[cat]
            weights = SIZE_WEIGHTS[cat]
            size_counts = np.random.multinomial(base, weights)

            for i, count in enumerate(size_counts):
                if count > 0:
                    rows.append({
                        'date': current.strftime('%Y-%m-%d'),
                        'product_id': pid_key,
                        'product_name': name,
                        'category': cat,
                        'color': np.random.choice(COLORS),
                        'size': sizes[i],
                        'quantity': int(count),
                        'unit_price': price,
                        'unit_cost': cost,
                    })
        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['quantity'] * df['unit_price']
    df['profit'] = df['quantity'] * (df['unit_price'] - df['unit_cost'])
    df['profit_margin'] = (df['unit_price'] - df['unit_cost']) / df['unit_price']
    return df


def generate_sample_inventory() -> pd.DataFrame:
    """Generate sample inventory data."""
    np.random.seed(42)
    rows = []
    pid = 1

    for cat, prods in PRODUCTS.items():
        for name, price, cost in prods:
            for size in SIZES[cat]:
                for color in COLORS[:3]:
                    stock = int(np.random.randint(0, 30))
                    reorder = int(np.random.randint(5, 12))
                    rows.append({
                        'product_id': f'P{pid:03d}',
                        'product_name': name,
                        'category': cat,
                        'color': color,
                        'size': size,
                        'current_stock': stock,
                        'reorder_point': reorder,
                        'max_stock': reorder * 4,
                        'supplier': np.random.choice(SUPPLIERS),
                        'lead_time_days': int(np.random.choice([3, 5, 7, 10])),
                        'unit_cost': cost,
                    })
            pid += 1

    return pd.DataFrame(rows)


def load_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"CSV解析失败：{e}")


def summarize_for_ai(sales_df: pd.DataFrame = None, inventory_df: pd.DataFrame = None) -> str:
    """Build a concise text summary of loaded data for the AI system prompt."""
    parts = []

    if sales_df is not None and not sales_df.empty:
        df = sales_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}"
        else:
            date_range = '未知'

        total_rev = df['revenue'].sum() if 'revenue' in df.columns else 0
        total_qty = df['quantity'].sum() if 'quantity' in df.columns else 0
        parts.append(f"【销售数据】区间：{date_range}，共 {len(df):,} 条记录")
        parts.append(f"总销售额：¥{total_rev:,.0f}，总售出件数：{total_qty:,} 件")

        if 'category' in df.columns and 'revenue' in df.columns:
            cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
            parts.append('各品类销售额：' + '；'.join(f'{k} ¥{v:,.0f}' for k, v in cat_rev.items()))

        if 'product_name' in df.columns and 'revenue' in df.columns:
            top5 = df.groupby('product_name')['revenue'].sum().sort_values(ascending=False).head(5)
            parts.append('TOP5商品：' + '；'.join(f'{k}(¥{v:,.0f})' for k, v in top5.items()))

        if 'profit' in df.columns and total_rev > 0:
            margin = df['profit'].sum() / total_rev * 100
            parts.append(f"综合利润率：{margin:.1f}%，总利润：¥{df['profit'].sum():,.0f}")

    if inventory_df is not None and not inventory_df.empty:
        has_reorder = 'reorder_point' in inventory_df.columns
        has_stock = 'current_stock' in inventory_df.columns
        low = int((inventory_df['current_stock'] <= inventory_df['reorder_point']).sum()) if (has_stock and has_reorder) else 0
        out = int((inventory_df['current_stock'] == 0).sum()) if has_stock else 0
        parts.append(f"\n【库存数据】共 {len(inventory_df):,} 个SKU，低库存预警 {low} 个，断货 {out} 个")

        if low > 0 and 'product_name' in inventory_df.columns and has_stock and has_reorder:
            names = inventory_df[inventory_df['current_stock'] <= inventory_df['reorder_point']]['product_name'].unique()[:5]
            parts.append(f"低库存商品（前5）：{', '.join(names)}")

    return '\n'.join(parts) if parts else '暂无已加载的数据'
