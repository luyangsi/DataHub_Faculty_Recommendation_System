"""
Page 3 — Sales analytics dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from shared_sidebar import render_sidebar
from analytics import compute_category_sales, compute_monthly_trend, find_top_products, find_slow_movers

st.set_page_config(page_title='销售分析 — 零售数据助手', page_icon='📊', layout='wide')

for key, default in [('api_key', ''), ('sales_data', None), ('inventory_data', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

render_sidebar()

st.title('📊 销售分析仪表盘')

df_raw = st.session_state.get('sales_data')
if df_raw is None:
    st.warning('请先在「📂 数据导入」页面上传或加载销售数据。')
    st.stop()

df = df_raw.copy()
df['date'] = pd.to_datetime(df['date'])

# ── Date range filter ─────────────────────────────────────────────────────────
min_date, max_date = df['date'].min().date(), df['date'].max().date()
c1, c2 = st.columns(2)
start = c1.date_input('开始日期', value=min_date, min_value=min_date, max_value=max_date)
end = c2.date_input('结束日期', value=max_date, min_value=min_date, max_value=max_date)
df = df[(df['date'].dt.date >= start) & (df['date'].dt.date <= end)]

if df.empty:
    st.warning('所选日期范围内无数据。')
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('---')
k1, k2, k3, k4 = st.columns(4)
total_rev = df['revenue'].sum() if 'revenue' in df.columns else 0
total_qty = df['quantity'].sum() if 'quantity' in df.columns else 0
total_profit = df['profit'].sum() if 'profit' in df.columns else 0
avg_margin = df['profit_margin'].mean() * 100 if 'profit_margin' in df.columns else 0

k1.metric('💰 总销售额', f'¥{total_rev:,.0f}')
k2.metric('📦 总售出件数', f'{total_qty:,} 件')
k3.metric('💹 总利润', f'¥{total_profit:,.0f}')
k4.metric('📈 平均利润率', f'{avg_margin:.1f}%')

st.markdown('---')

# ── Row 1: Pie + Monthly trend ────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader('各品类销售额占比')
    cat_sales = compute_category_sales(df)
    if not cat_sales.empty:
        fig = px.pie(
            cat_sales, names='category', values='revenue',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.42,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader('月度销售额趋势')
    monthly = compute_monthly_trend(df)
    if not monthly.empty:
        fig = px.line(
            monthly, x='month', y='revenue',
            markers=True,
            labels={'month': '月份', 'revenue': '销售额 (¥)'},
            color_discrete_sequence=['#636EFA'],
        )
        if 'profit' in monthly.columns:
            fig.add_bar(x=monthly['month'], y=monthly['profit'], name='利润', opacity=0.4, marker_color='#EF553B')
        fig.update_layout(xaxis_tickangle=-30, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

# ── Row 2: Top products + Category bar ───────────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader('TOP 10 畅销商品（按销售额）')
    top_prods = find_top_products(df, n=10)
    if not top_prods.empty:
        fig = px.bar(
            top_prods.sort_values('revenue'),
            x='revenue', y='product_name',
            orientation='h',
            color='category',
            labels={'revenue': '销售额 (¥)', 'product_name': ''},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(margin=dict(t=20, b=20), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with col_r2:
    st.subheader('各品类 销量 vs 利润')
    cat_sales2 = compute_category_sales(df)
    if not cat_sales2.empty:
        plot_cols = [c for c in ['quantity', 'profit'] if c in cat_sales2.columns]
        if plot_cols:
            fig = px.bar(
                cat_sales2, x='category', y=plot_cols,
                barmode='group',
                labels={'value': '数量 / 利润 (¥)', 'category': '品类', 'variable': '指标'},
                color_discrete_map={'quantity': '#636EFA', 'profit': '#EF553B'},
            )
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ── Size distribution ─────────────────────────────────────────────────────────
if 'size' in df.columns and 'category' in df.columns:
    st.markdown('---')
    st.subheader('尺码销量分布')
    cats = sorted(df['category'].unique().tolist())
    selected_cat = st.selectbox('选择品类', options=cats)
    size_df = df[df['category'] == selected_cat].groupby('size')['quantity'].sum().reset_index()
    size_df = size_df.sort_values('quantity', ascending=False)
    fig = px.bar(
        size_df, x='size', y='quantity',
        labels={'size': '尺码', 'quantity': '销量'},
        color='quantity',
        color_continuous_scale='Blues',
    )
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Slow movers ───────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('⚠️ 滞销商品（销售额后 20%）')
slow = find_slow_movers(df)
if not slow.empty:
    st.dataframe(
        slow[['product_name', 'category', 'revenue', 'revenue_pct']].rename(
            columns={'revenue': '销售额 (¥)', 'revenue_pct': '占比 (%)', 'product_name': '商品名', 'category': '品类'}
        ).round(2),
        use_container_width=True,
        height=250,
    )
