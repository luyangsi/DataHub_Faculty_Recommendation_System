"""
零售数据助手 — 主入口
Run: streamlit run retail_assistant/app.py
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from shared_sidebar import render_sidebar

st.set_page_config(
    page_title='零售数据助手',
    page_icon='🏪',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Initialize session state defaults
for key, default in [('api_key', ''), ('sales_data', None), ('inventory_data', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

render_sidebar()

# ── Home page ────────────────────────────────────────────────────────────────
st.title('🏪 零售数据助手')
st.markdown('**帮助服装 / 鞋类零售店主 — 分析销售数据、发现规律、制定进货策略**')
st.markdown('---')

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.info('**💬 AI 助手**\n\n与 AI 对话，提问销售分析，获取进货建议')
with col2:
    st.info('**📂 数据导入**\n\n上传 CSV / Excel，或一键加载示例数据')
with col3:
    st.info('**📊 销售分析**\n\n品类趋势、畅销榜单、利润率可视化')
with col4:
    st.info('**📦 库存管理**\n\n库存预警、断货风险、周转率')
with col5:
    st.info('**✅ 进货建议**\n\n规则 + AI 双引擎进货优先级清单')

st.markdown('---')
st.markdown('### 🚀 快速开始')
st.markdown(
    '1. **设置 API Key** → 在左侧侧栏填入你的 Claude API Key  \n'
    '2. **导入数据** → 点击左侧「📂 数据导入」，上传 CSV 或体验示例数据  \n'
    '3. **查看图表** → 「📊 销售分析」和「📦 库存管理」  \n'
    '4. **咨询 AI** → 「💬 AI 助手」提问，例如："哪些款式值得补货？"  \n'
    '5. **获取清单** → 「✅ 进货建议」查看优先级补货计划  \n'
)

st.markdown('---')
st.markdown(
    '> 💡 **提示**：销售 CSV 至少需要字段 `date / product_name / category / quantity / unit_price`。'
    '库存 CSV 至少需要 `product_name / category / current_stock / reorder_point`。'
    '也可以直接在「数据导入」页面生成示例数据体验完整功能。'
)
