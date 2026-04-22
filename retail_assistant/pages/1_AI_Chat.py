"""
Page 1 — AI Chat assistant.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from shared_sidebar import render_sidebar
from ai_assistant import stream_chat
from data_utils import summarize_for_ai

st.set_page_config(page_title='AI 助手 — 零售数据助手', page_icon='💬', layout='wide')

for key, default in [('api_key', ''), ('sales_data', None), ('inventory_data', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

render_sidebar()

st.title('💬 AI 销售助手')

if not st.session_state.get('api_key'):
    st.warning('请先在左侧侧栏填入 Claude API Key，才能使用 AI 功能。')
    st.stop()

sales_df = st.session_state.get('sales_data')
inv_df = st.session_state.get('inventory_data')
data_context = summarize_for_ai(sales_df, inv_df)

if sales_df is not None:
    st.success(
        f'✅ 已加载销售数据（{len(sales_df):,} 条）'
        + (f' + 库存数据（{len(inv_df):,} SKU）' if inv_df is not None else '')
        + '，AI 将结合数据回答问题。'
    )
else:
    st.info('💡 尚未加载数据。AI 可回答通用零售问题；上传数据后分析将更精准。')

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Welcome message on first load
if not st.session_state.chat_history:
    with st.chat_message('assistant'):
        st.markdown(
            '你好！我是你的零售数据助手。你可以直接发给我上月销售数据，'
            '或者问我任何关于进货和销售分析的问题。比如：**"帮我分析一下哪些款式值得补货"**。'
        )

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input('输入你的问题，例如：上个月哪个品类利润最高？'):
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    api_messages = [{'role': m['role'], 'content': m['content']} for m in st.session_state.chat_history]

    with st.chat_message('assistant'):
        try:
            response = st.write_stream(stream_chat(api_messages, st.session_state.api_key, data_context))
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        except Exception as e:
            err = f'❌ API 调用失败：{e}'
            st.error(err)
            st.session_state.chat_history.pop()  # remove the unanswered user message

# ── Quick questions sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('---')
    st.markdown('### 💡 快速提问')
    quick_qs = [
        '分析一下各品类的销售情况',
        '哪些商品有断货风险？',
        '本季度应该重点补哪些货？',
        '哪些款式已经滞销了？',
        '利润最高的前5款是什么？',
        '不同尺码的销量分布如何？',
    ]
    for q in quick_qs:
        if st.button(q, key=f'quick_{q}', use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': q})
            st.rerun()

    st.markdown('---')
    if st.button('🗑️ 清空对话记录', type='secondary', use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
