"""
Page 4 — Inventory management and stock alerts.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from shared_sidebar import render_sidebar
from analytics import compute_inventory_status

st.set_page_config(page_title='库存管理 — 零售数据助手', page_icon='📦', layout='wide')

for key, default in [('api_key', ''), ('sales_data', None), ('inventory_data', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

render_sidebar()

st.title('📦 库存管理')

inv_raw = st.session_state.get('inventory_data')
if inv_raw is None:
    st.warning('请先在「📂 数据导入」页面上传或加载库存数据。')
    st.stop()

df = compute_inventory_status(inv_raw)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric('📦 总 SKU 数', f'{len(df):,}')
k2.metric('🔴 断货', int((df['status'] == '断货').sum()), delta_color='inverse')
k3.metric('🟡 低库存', int((df['status'] == '低库存').sum()), delta_color='inverse')
k4.metric('✅ 库存正常', int((df['status'] == '正常').sum()))

st.markdown('---')

# ── Filters ───────────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns(3)
with fc1:
    status_opts = df['status'].unique().tolist()
    status_filter = st.multiselect('库存状态', options=status_opts, default=['断货', '低库存'])
with fc2:
    cat_opts = sorted(df['category'].unique().tolist()) if 'category' in df.columns else []
    cat_filter = st.multiselect('品类', options=cat_opts)
with fc3:
    search = st.text_input('商品名搜索', placeholder='输入关键词…')

filtered = df.copy()
if status_filter:
    filtered = filtered[filtered['status'].isin(status_filter)]
if cat_filter:
    filtered = filtered[filtered['category'].isin(cat_filter)]
if search:
    filtered = filtered[filtered['product_name'].str.contains(search, na=False)]

# ── Color-coded table ─────────────────────────────────────────────────────────
STATUS_COLORS = {
    '断货': 'background-color: #ffe0e0',
    '低库存': 'background-color: #fff7cc',
    '正常': 'background-color: #e6f4ea',
    '库存过高': 'background-color: #dce8ff',
}

DISPLAY_COLS = [c for c in [
    'product_name', 'category', 'color', 'size',
    'current_stock', 'reorder_point', 'max_stock',
    'status', 'restock_qty', 'supplier', 'lead_time_days',
] if c in filtered.columns]

def _highlight(row):
    color = STATUS_COLORS.get(row.get('status', ''), '')
    return [color] * len(row)

st.dataframe(
    filtered[DISPLAY_COLS].style.apply(_highlight, axis=1),
    use_container_width=True,
    height=420,
)

st.caption(f'显示 {len(filtered):,} 条 / 共 {len(df):,} 条 SKU')

# ── Download filtered ─────────────────────────────────────────────────────────
csv = filtered[DISPLAY_COLS].to_csv(index=False).encode('utf-8-sig')
st.download_button('📥 下载筛选结果 CSV', csv, 'inventory_filtered.csv', 'text/csv')

st.markdown('---')

# ── Stock level bar chart ─────────────────────────────────────────────────────
if 'category' in df.columns and 'current_stock' in df.columns:
    st.subheader('各品类库存总量 vs 补货点')
    cat_stock = df.groupby('category')[
        [c for c in ['current_stock', 'reorder_point'] if c in df.columns]
    ].sum().reset_index()

    fig = px.bar(
        cat_stock, x='category',
        y=[c for c in ['current_stock', 'reorder_point'] if c in cat_stock.columns],
        barmode='group',
        labels={'value': '数量', 'category': '品类', 'variable': ''},
        color_discrete_map={'current_stock': '#636EFA', 'reorder_point': '#EF553B'},
    )
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Supplier concentration ────────────────────────────────────────────────────
if 'supplier' in df.columns:
    st.subheader('供应商集中度')
    sup_count = df.groupby('supplier')['product_name'].nunique().reset_index()
    sup_count.columns = ['supplier', 'product_count']
    fig = px.pie(
        sup_count, names='supplier', values='product_count',
        hole=0.38,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    top_supplier = sup_count.sort_values('product_count', ascending=False).iloc[0]
    top_pct = top_supplier['product_count'] / sup_count['product_count'].sum() * 100
    if top_pct > 50:
        st.warning(
            f'⚠️ 供应商「{top_supplier["supplier"]}」占全部商品的 **{top_pct:.0f}%**，'
            '供应集中度较高，建议分散采购以降低断货风险。'
        )
