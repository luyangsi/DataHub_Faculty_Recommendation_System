# 零售数据助手

帮助服装 / 鞋类零售店主分析销售数据、发现规律、制定进货策略的 AI 应用。

## 功能

- **💬 AI 助手** — 与 Claude 对话，提问销售分析，获取进货建议
- **📂 数据导入** — 上传 CSV / Excel，或一键加载示例数据
- **📊 销售分析** — 品类趋势、畅销榜单、利润率可视化
- **📦 库存管理** — 断货预警、低库存标注、供应商集中度
- **✅ 进货建议** — 规则引擎 + AI 双引擎补货优先级清单

## 环境要求

- Python 3.9+
- [Anthropic API Key](https://console.anthropic.com/)

## 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd DataHub_Project

# 2. 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. 安装依赖
pip install -r retail_assistant/requirements.txt
```

## 启动

```bash
streamlit run retail_assistant/app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 使用步骤

1. **设置 API Key** — 在左侧侧栏填入你的 Claude API Key（`sk-ant-...`）
2. **导入数据** — 前往「📂 数据导入」，上传 CSV 或点击「加载示例数据」快速体验
3. **查看图表** — 「📊 销售分析」和「📦 库存管理」会自动渲染
4. **咨询 AI** — 「💬 AI 助手」可自由提问，例如："上个月哪个品类利润最高？"
5. **获取清单** — 「✅ 进货建议」生成补货优先级清单，支持下载 CSV

## 数据格式

**销售数据 CSV（必需列）**

| 列名 | 说明 | 示例 |
|------|------|------|
| `date` | 日期 | `2024-03-15` |
| `product_name` | 商品名称 | `直筒牛仔裤` |
| `category` | 品类 | `裤子` |
| `quantity` | 销售数量 | `3` |
| `unit_price` | 售价（元） | `229` |
| `unit_cost` | 成本价（可选） | `75` |

**库存数据 CSV（必需列）**

| 列名 | 说明 | 示例 |
|------|------|------|
| `product_name` | 商品名称 | `直筒牛仔裤` |
| `category` | 品类 | `裤子` |
| `current_stock` | 当前库存 | `12` |
| `reorder_point` | 补货触发点 | `8` |
| `supplier` | 供应商（可选） | `杭州优品服饰` |

> 不确定格式？在「📂 数据导入 → 使用示例数据」页面可下载标准模板 CSV。
