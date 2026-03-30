import streamlit as st
import xgboost as xgb
import pandas as pd
import sqlite3
import os

# --- 1. 路径配置 ---
# 获取当前脚本所在文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义模型路径 (固定)
model_path = os.path.join(current_dir, 'DataBase', 'Model', 'best_issa_xgboost.json')

# 数据库路径逻辑：优先寻找脱敏后的 sample.db
sample_path = os.path.join(current_dir, 'DataBase', 'sample.db')
original_path = os.path.join(current_dir, 'DataBase', 'water_plant_final.db')

# 自动切换数据库：如果 sample.db 存在，则使用它；否则回退到原始数据库
db_path = sample_path if os.path.exists(sample_path) else original_path

# --- 2. 页面设置 (Streamlit 强制要求：这必须是第一个调用的 Streamlit 命令) ---
st.set_page_config(
    page_title="ISSA-XGBoost 智能投药系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 水厂 - ISSA-XGBoost 智能投药系统")
st.markdown("---")


# --- 3. 提取“昨日记忆” ---
@st.cache_data  # 使用缓存提高加载速度
def load_memory():
    try:
        conn = sqlite3.connect(db_path)
        sql = 'SELECT "矾\n（kg/Km³）" as d, "浑浊度\n（NTU）" as t FROM filled_data ORDER BY "date" DESC LIMIT 1'
        res = pd.read_sql(sql, conn)
        conn.close()
        return float(res['d'].iloc[0]), float(res['t'].iloc[0])
    except:
        return 0.0, 0.0


last_dosage, last_turb = load_memory()

# --- 4. 侧边栏：状态展示 ---
st.sidebar.header("🧠 历史工况记忆")
st.sidebar.info(f"昨日平均投药量: **{last_dosage:.2f}** kg/Km³")
st.sidebar.info(f"昨日原水平均浊度: **{last_turb:.2f}** NTU")

# --- 5. 主界面：实时数据输入 ---
st.subheader("📝 请输入当前实时监测指标")
col1, col2, col3 = st.columns(3)

with col1:
    turb = st.number_input("当前原水浊度 (NTU)", value=2.50, step=0.1)
    ph = st.number_input("当前 pH 值", value=7.20, step=0.01)

with col2:
    flow = st.number_input("当前原水量 (Km³)", value=5000.0, step=100.0)
    temp = st.number_input("当前水温 (℃)", value=18.0, step=0.1)

with col3:
    ammo = st.number_input("当前氨氮 (mg/L)", value=0.10, step=0.01)

# --- 6. 执行预测 ---
if st.button("🚀 开始计算决策建议", use_container_width=True):
    try:
        # 构建 8D 特征 (严格匹配训练顺序)
        features = {
            'turbidity': [turb], 'ph': [ph], 'temp': [temp],
            'flow': [flow], 'ammonia': [ammo],
            'last_turbidity': [last_turb], 'last_dosage': [last_dosage],
            'flow_turbidity_inter': [flow * turb]
        }

        # 加载模型
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        prediction = model.predict(pd.DataFrame(features))[0]

        # 展示结果
        st.markdown("---")
        st.balloons()  # 预测成功的小特效
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="建议投加量", value=f"{prediction:.3f} kg/Km³")
        with c2:
            st.write("💡 **专家建议：**")
            st.write(f"当前工况下，模型预测投药量为 {prediction:.3f}。请结合沉淀池矾花情况进行微调。")

    except Exception as e:
        st.error(f"模型运行失败: {e}")