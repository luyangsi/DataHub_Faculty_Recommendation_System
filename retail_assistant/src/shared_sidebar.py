"""
Shared sidebar rendered on every page.
"""
import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown('## ⚙️ 设置')
        current_key = st.session_state.get('api_key', '')
        api_key = st.text_input(
            'Claude API Key',
            type='password',
            value=current_key,
            placeholder='sk-ant-...',
            help='在 console.anthropic.com 获取你的 API Key',
        )
        if api_key != current_key:
            st.session_state['api_key'] = api_key

        if st.session_state.get('api_key'):
            st.success('✅ API Key 已设置')
        else:
            st.warning('⚠️ 未设置 API Key（AI功能不可用）')

        st.markdown('---')
        st.markdown('### 📊 数据状态')
        sales = st.session_state.get('sales_data')
        inv = st.session_state.get('inventory_data')

        if sales is not None:
            st.success(f'✅ 销售数据：{len(sales):,} 条')
        else:
            st.info('📂 未加载销售数据')

        if inv is not None:
            st.success(f'✅ 库存数据：{len(inv):,} 个SKU')
        else:
            st.info('📂 未加载库存数据')

        st.markdown('---')
        st.caption('零售数据助手 v1.0')
