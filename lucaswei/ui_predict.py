import tkinter as tk
from tkinter import messagebox, ttk
import os
import sqlite3
import pandas as pd
import xgboost as xgb

# --- 1. 路径与模型配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, 'DataBase', 'water_plant_final.db')
model_path = os.path.join(current_dir, 'DataBase', 'Model', 'best_issa_xgboost.json')


class DosageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("水厂 - ISSA-XGBoost 智能投药助手")
        self.root.geometry("450x600")

        # 加载昨日数据
        self.last_dosage, self.last_turb = self.load_memory()

        # 创建界面
        self.create_widgets()

    def load_memory(self):
        """从数据库提取‘昨日记忆’ """
        try:
            conn = sqlite3.connect(db_path)
            sql = 'SELECT "矾\n（kg/Km³）" as d, "浑浊度\n（NTU）" as t FROM filled_data ORDER BY "date" DESC LIMIT 1'
            res = pd.read_sql(sql, conn)
            conn.close()
            return float(res['d'].iloc[0]), float(res['t'].iloc[0])
        except Exception as e:
            messagebox.showerror("数据库错误", f"无法读取记忆: {e}")
            return 0.0, 0.0

    def create_widgets(self):
        # 标题
        header = tk.Label(self.root, text="智能投药决策系统", font=("PingFang SC", 18, "bold"), pady=20)
        header.pack()

        # 记忆展示区
        mem_frame = tk.LabelFrame(self.root, text=" 🧠 昨日工况记忆 ", padx=10, pady=10)
        mem_frame.pack(fill="x", padx=20)
        tk.Label(mem_frame, text=f"昨日投药量: {self.last_dosage:.2f} kg/Km³").pack(side="left", padx=10)
        tk.Label(mem_frame, text=f"昨日原水浊度: {self.last_turb:.2f} NTU").pack(side="right", padx=10)

        # 输入区
        input_frame = tk.Frame(self.root, pady=20)
        input_frame.pack()

        self.entries = {}
        fields = [
            ("当前浊度 (NTU)", "turb"),
            ("当前 pH 值", "ph"),
            ("当前水温 (℃)", "temp"),
            ("当前流量 (Km³)", "flow"),
            ("当前氨氮 (mg/L)", "ammo")
        ]

        for label_text, key in fields:
            row = tk.Frame(input_frame, pady=5)
            row.pack(fill="x")
            tk.Label(row, text=label_text, width=15, anchor="w").pack(side="left")
            ent = tk.Entry(row)
            ent.pack(side="right", expand=True, fill="x")
            self.entries[key] = ent

        # 预测按钮
        self.btn = tk.Button(self.root, text="开始计算建议药量", command=self.predict,
                             bg="#007AFF", fg="black", font=("PingFang SC", 12, "bold"), pady=10)
        self.btn.pack(fill="x", padx=50, pady=20)

        # 结果展示区
        self.res_label = tk.Label(self.root, text="-- kg/Km³", font=("Helvetica", 24, "bold"), fg="#D32F2F")
        self.res_label.pack()
        tk.Label(self.root, text="建议投加参考值", fg="gray").pack()

    def predict(self):
        try:
            # 获取输入
            inputs = {k: float(v.get()) for k, v in self.entries.items()}

            # 构建 8D 特征向量
            features = {
                'turbidity': [inputs['turb']], 'ph': [inputs['ph']], 'temp': [inputs['temp']],
                'flow': [inputs['flow']], 'ammonia': [inputs['ammo']],
                'last_turbidity': [self.last_turb], 'last_dosage': [self.last_dosage],
                'flow_turbidity_inter': [inputs['flow'] * inputs['turb']]
            }

            # 推理
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            res = model.predict(pd.DataFrame(features))[0]

            # 更新显示
            self.res_label.config(text=f"{res:.3f} kg/Km³")

        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数字！")
        except Exception as e:
            messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DosageApp(root)
    root.mainloop()