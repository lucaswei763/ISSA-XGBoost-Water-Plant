# ISSA-XGBoost-Water-Plant 🌊

这是一个基于改进麻雀搜索算法 (ISSA) 优化 XGBoost 的智慧水务投药决策系统。

## 🚀 项目亮点
- **8D 特征工程**：集成原水浊度、pH、流量、氨氮及历史记忆特征。
- **智能调参**：利用 ISSA 算法自动寻找 XGBoost 的最佳超参数，显著提升模型泛化能力。
- **现代化 UI**：基于 Streamlit 构建的 Web 决策界面，支持实时工况输入与药量建议。

## 🛠️ 快速开始
1. **环境克隆与安装**：
    ```bash
    git clone https://github.com/lucas763/ISSA-XGBoost-Water-Plant.git
    cd ISSA-XGBoost-Water-Plant
    pip install -r requirements.txt
    ```

2. **运行演示界面：**

    ```bash
    streamlit run app.py
    ```

3. **🔒 隐私与数据说明**

    本项目已对数据进行脱敏处理。DataBase/sample.db 为合成的演示数据，非真实生产数据。原始生产数据库已通过 .gitignore 屏蔽。

## ⚖️ License
Distributed under the **MIT License**. See `LICENSE` for more information.