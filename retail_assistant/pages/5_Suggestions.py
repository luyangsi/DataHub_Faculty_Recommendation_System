"""
Page 5 — Reorder suggestions (rule-based + AI).
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from shared_sidebar import render_sidebar
from analytics import compute_reorder_suggestions
from ai_assistant import stream_chat
from data_utils import summarize_for_ai

st.set_page_config(page_title='进货建议 — 零售数据助手', page_icon='✅', layout='wide')

for key, default in [('api_key', ''), ('sales_data', None), ('inventory_data', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

render_sidebar()

st.title('✅ 进货建议')

sales_df = st.session_state.get('sales_data')
inv_df = st.session_state.get('inventory_data')

if sales_df is None or inv_df is None:
    st.warning('需要同时加载销售数据和库存数据，才能生成进货建议。请先前往「📂 数据导入」。')
    st.stop()

# ── Rule-based suggestions ────────────────────────────────────────────────────
st.subheader('📋 补货优先级清单（规则引擎）')
st.caption('基于近 30 天销售速度 × 当前库存水位自动计算')

with st.spinner('计算补货建议中…'):
    suggestions = compute_reorder_suggestions(sales_df, inv_df)

if suggestions.empty:
    st.info('数据不足以生成建议，请确认销售数据和库存数据中有匹配的 product_id。')
    st.stop()

urgent = suggestions[suggestions['priority'] == '🔴 紧急补货']
recommend = suggestions[suggestions['priority'] == '🟡 建议补货']
consider = suggestions[suggestions['priority'] == '🟢 可考虑补货']

SHOW_COLS = [c for c in [
    'product_name', 'category', 'total_stock', 'out_of_stock_skus',
    'low_stock_skus', 'days_of_stock', 'avg_daily', 'suggested_qty',
    'estimated_cost', 'supplier',
] if c in suggestions.columns]

COL_NAMES = {
    'product_name': '商品名', 'category': '品类',
    'total_stock': '当前总库存', 'out_of_stock_skus': '断货SKU数',
    'low_stock_skus': '低库存SKU数', 'days_of_stock': '可售天数',
    'avg_daily': '日均销量', 'suggested_qty': '建议补货量',
    'estimated_cost': '预计成本(¥)', 'supplier': '供应商',
}

if len(urgent) > 0:
    st.error(f'🔴 **紧急补货：{len(urgent)} 个商品已出现断货，需立即处理！**')
    st.dataframe(
        urgent[SHOW_COLS].rename(columns=COL_NAMES).round(1),
        use_container_width=True,
    )

if len(recommend) > 0:
    st.warning(f'🟡 **建议补货：{len(recommend)} 个商品库存偏低或不足 14 天**')
    st.dataframe(
        recommend[SHOW_COLS].rename(columns=COL_NAMES).round(1),
        use_container_width=True,
    )

if len(consider) > 0:
    with st.expander(f'🟢 可考虑补货（{len(consider)} 个，库存 14-30 天）'):
        st.dataframe(
            consider[SHOW_COLS].rename(columns=COL_NAMES).round(1),
            use_container_width=True,
        )

# ── Cost summary ──────────────────────────────────────────────────────────────
if 'estimated_cost' in suggestions.columns:
    urgent_cost = urgent['estimated_cost'].sum() if len(urgent) > 0 else 0
    rec_cost = recommend['estimated_cost'].sum() if len(recommend) > 0 else 0
    total_cost = urgent_cost + rec_cost

    c1, c2, c3 = st.columns(3)
    c1.metric('🔴 紧急补货预算', f'¥{urgent_cost:,.0f}')
    c2.metric('🟡 建议补货预算', f'¥{rec_cost:,.0f}')
    c3.metric('💰 合计预估成本', f'¥{total_cost:,.0f}')

# ── Download ──────────────────────────────────────────────────────────────────
csv = suggestions[SHOW_COLS + ['priority']].rename(columns=COL_NAMES).round(1).to_csv(index=False).encode('utf-8-sig')
st.download_button('📥 下载完整补货清单 CSV', csv, 'reorder_suggestions.csv', 'text/csv')

# ── AI deep analysis ──────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('🤖 AI 深度进货分析报告')

if not st.session_state.get('api_key'):
    st.info('在左侧侧栏设置 Claude API Key 后，可获取 AI 的综合进货分析和建议。')
else:
    if st.button('生成 AI 进货分析报告', type='primary'):
        data_context = summarize_for_ai(sales_df, inv_df)
        prompt = (
            '请基于当前的销售数据和库存数据，生成一份完整的进货建议报告，内容包括：\n'
            '1. 📊 本期销售关键发现（哪些品类/款式表现突出或下滑，与上月或季节基准对比）\n'
            '2. ⚠️ 库存风险提示（断货/积压详情，以及对销售的潜在影响）\n'
            '3. ✅ 具体进货建议（品类、款式、建议数量区间、采购优先级、最佳时机）\n'
            '4. 💡 供应商和备货策略建议（是否需要分散货源，如何降低缺货风险）\n'
            '5. ❓ 如需进一步分析，还需要补充哪些数据'
        )
        messages = [{'role': 'user', 'content': prompt}]

        with st.chat_message('assistant'):
            try:
                response = st.write_stream(stream_chat(messages, st.session_state.api_key, data_context))
                report_bytes = response.encode('utf-8')
                st.download_button('📥 下载 AI 分析报告', report_bytes, 'ai_purchase_report.txt', 'text/plain')
            except Exception as e:
                st.error(f'AI 调用失败：{e}')
