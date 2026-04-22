"""
Page 2 — Data import (CSV upload or sample data).
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from shared_sidebar import render_sidebar
from data_utils import generate_sample_sales, generate_sample_inventory, load_csv

st.set_page_config(page_title='数据导入 — 零售数据助手', page_icon='📂', layout='wide')

for key, default in [('api_key', ''), ('sales_data', None), ('inventory_data', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

render_sidebar()

st.title('📂 数据导入')
st.markdown('上传你自己的 CSV 数据，或加载内置示例数据快速体验功能。')

tab1, tab2, tab3 = st.tabs(['📤 上传销售数据', '📦 上传库存数据', '🎲 使用示例数据'])

# ── Tab 1: Upload sales ───────────────────────────────────────────────────────
with tab1:
    st.markdown('#### 销售记录 CSV')
    st.markdown(
        '**必需列：** `date`（YYYY-MM-DD）、`product_name`、`category`、`quantity`、`unit_price`  \n'
        '**可选列：** `product_id`、`color`、`size`、`unit_cost`'
    )

    uploaded = st.file_uploader('选择销售数据文件', type=['csv', 'xlsx'], key='sales_upload')
    if uploaded:
        try:
            df = load_csv(uploaded)
            # Auto-compute derived columns if missing
            if 'revenue' not in df.columns and 'quantity' in df.columns and 'unit_price' in df.columns:
                df['revenue'] = df['quantity'] * df['unit_price']
            if 'profit' not in df.columns and 'unit_cost' in df.columns and 'revenue' in df.columns:
                df['profit'] = df['revenue'] - df['quantity'] * df['unit_cost']
                df['profit_margin'] = (df['unit_price'] - df['unit_cost']) / df['unit_price']
            st.session_state.sales_data = df
            st.success(f'✅ 成功导入 {len(df):,} 条销售记录')
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f'导入失败：{e}')

# ── Tab 2: Upload inventory ───────────────────────────────────────────────────
with tab2:
    st.markdown('#### 库存数据 CSV')
    st.markdown(
        '**必需列：** `product_name`、`category`、`current_stock`、`reorder_point`  \n'
        '**可选列：** `product_id`、`color`、`size`、`max_stock`、`supplier`、`lead_time_days`、`unit_cost`'
    )

    uploaded_inv = st.file_uploader('选择库存数据文件', type=['csv', 'xlsx'], key='inv_upload')
    if uploaded_inv:
        try:
            df = load_csv(uploaded_inv)
            st.session_state.inventory_data = df
            st.success(f'✅ 成功导入 {len(df):,} 个 SKU 的库存数据')
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f'导入失败：{e}')

# ── Tab 3: Sample data ────────────────────────────────────────────────────────
with tab3:
    st.markdown('#### 内置示例数据')
    st.info(
        '示例数据包含一家虚拟服装鞋类店铺过去 6 个月的销售记录和库存快照，'
        '可用来完整体验所有功能。'
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**📊 示例销售数据**')
        st.markdown('- 品类：上衣 / 裤子 / 裙子 / 外套 / 鞋类 / 配件')
        st.markdown('- 含季节性波动、周末效应')
        st.markdown('- 含售价 / 成本 / 利润字段')

        if st.button('加载示例销售数据', type='primary', use_container_width=True):
            with st.spinner('正在生成示例数据（约 10 秒）…'):
                sample = generate_sample_sales(months=6)
            st.session_state.sales_data = sample
            st.success(f'✅ 已加载 {len(sample):,} 条示例销售记录')
            st.dataframe(sample.head(10), use_container_width=True)
            csv_bytes = sample.to_csv(index=False).encode('utf-8-sig')
            st.download_button('📥 下载示例销售 CSV', csv_bytes, 'sample_sales.csv', 'text/csv')

    with col2:
        st.markdown('**📦 示例库存数据**')
        st.markdown('- 覆盖所有品类 / 颜色 / 尺码 SKU')
        st.markdown('- 含供应商 / 补货点 / 最大库存')
        st.markdown('- 模拟真实断货和低库存场景')

        if st.button('加载示例库存数据', type='primary', use_container_width=True):
            with st.spinner('正在生成示例数据…'):
                sample_inv = generate_sample_inventory()
            st.session_state.inventory_data = sample_inv
            st.success(f'✅ 已加载 {len(sample_inv):,} 个 SKU 的示例库存')
            st.dataframe(sample_inv.head(10), use_container_width=True)
            csv_bytes = sample_inv.to_csv(index=False).encode('utf-8-sig')
            st.download_button('📥 下载示例库存 CSV', csv_bytes, 'sample_inventory.csv', 'text/csv')

    st.markdown('---')
    st.markdown('**清除数据**')
    c1, c2 = st.columns(2)
    with c1:
        if st.button('🗑️ 清除销售数据', use_container_width=True):
            st.session_state.sales_data = None
            st.rerun()
    with c2:
        if st.button('🗑️ 清除库存数据', use_container_width=True):
            st.session_state.inventory_data = None
            st.rerun()
