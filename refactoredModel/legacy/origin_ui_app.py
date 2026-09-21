#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名称：refactoredModel/origin_ui_app.py
所属类别：重构核心生产代码 (Refactored Core Production) / 历史留存代码 (已弃用)

功能描述：
    重构前的水厂投加量预测桌面 GUI 应用程序历史原型脚本。
    该文件基于选取的 6 维关键指标进行动态输入表单渲染，并执行基础的 `predictor_service` 调用与展示。

运行与使用方法：
    本文件为历史存档，目前已不推荐使用。如果需要运行，需依赖 customtkinter：
    python origin_ui_app.py

调用与依赖关系：
    - 属于历史遗留代码，当前生产系统的主界面已全面升级为新版 [ui_app.py](file:///Users/weiyihang/PycharmProjects/WaterPlant/refactoredModel/ui_app.py)。
    - 调用 `predictor_service.WaterPredictor`。

设计细节与关键备注：
    - ⚠️ **注意**：本文件已经废弃，仅作为历史代码的架构留存与布局参考，不参与当前的预测决策。
"""

import os
import datetime
import threading
import customtkinter as ctk
from tkinter import messagebox
import joblib

# 导入我们刚刚写好的本地预测服务
from predictor_service import WaterPredictor

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class WaterPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("水厂投矾量智能预测系统")
        self.geometry("550x700")  # 稍微拉长一点，适应6个特征
        self.resizable(False, False)

        self.predictor = None
        self.entries = {}

        # 先尝试读取特征列表，用于动态生成界面
        self.required_features = []
        try:
            self.required_features = joblib.load('models/selected_features.pkl')
        except:
            messagebox.showwarning("警告", "无法读取 models/selected_features.pkl，请确认是否已运行模型训练。")

        self._build_ui()

        # 异步加载完整模型
        self.status_label.configure(text="状态: 正在加载预测模型...", text_color="orange")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _build_ui(self):
        # 标题栏
        self.title_label = ctk.CTkLabel(self, text="投矾量单次预测 (核心6维)", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(20, 20))

        # 核心表单区域
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.pack(padx=40, pady=10, fill="both", expand=True)

        # 字典翻译：让英文特征名显示为好看的中文
        # 字典翻译：让 UI 界面的标签看起来更整齐专业
        name_dict = {
            next((f for f in self.required_features if '浑浊度' in f), '浑浊度'): '原水浑浊度 (NTU)',
            next((f for f in self.required_features if '铁' in f), '铁'): '铁含量 (mg/L)',
            next((f for f in self.required_features if '氨氮' in f), '氨氮'): '氨氮含量 (mg/L)',
            next((f for f in self.required_features if '消耗电' in f), '消耗电'): '消耗电量 (kW·h)',
            next((f for f in self.required_features if '亚硝酸盐' in f), '亚硝酸盐'): '亚硝酸盐氮 (mg/L)',
            next((f for f in self.required_features if '温度' in f), '温度'): '原水温度 (℃)'
        }

        # 动态生成输入框！
        for i, feat in enumerate(self.required_features):
            # 尝试翻译特征名，翻译不出来就显示原名
            display_name = name_dict.get(feat, feat)

            lbl = ctk.CTkLabel(self.form_frame, text=display_name, font=ctk.CTkFont(size=14))
            lbl.grid(row=i, column=0, padx=20, pady=12, sticky="e")

            entry = ctk.CTkEntry(self.form_frame, width=200)
            entry.grid(row=i, column=1, padx=20, pady=12, sticky="w")
            self.entries[feat] = entry

        # 预测按钮
        self.predict_btn = ctk.CTkButton(
            self, text="开始预测", font=ctk.CTkFont(size=16, weight="bold"),
            height=45, state="disabled", command=self._on_predict_click
        )
        self.predict_btn.pack(pady=(20, 10))

        # 结果显示框
        self.result_label = ctk.CTkLabel(
            self, text="等待输入数据...", font=ctk.CTkFont(size=20, weight="bold"), text_color="gray"
        )
        self.result_label.pack(pady=(10, 10))

        # 底部状态栏
        self.status_label = ctk.CTkLabel(self, text="状态: 初始化", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="bottom", pady=10)

    def _load_model_thread(self):
        try:
            self.predictor = WaterPredictor()
            self.after(0, self._on_model_loaded_success)
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda err=error_msg: self._on_model_loaded_fail(err))

    def _on_model_loaded_success(self):
        self.status_label.configure(text=f"状态: 模型就绪 ({self.predictor.model_type})", text_color="green")
        self.predict_btn.configure(state="normal")

    def _on_model_loaded_fail(self, error_msg):
        self.status_label.configure(text="状态: 模型加载失败", text_color="red")
        messagebox.showerror("致命错误", f"无法加载预测模型，请检查 models 目录。\n\n详情: {error_msg}")

    def _on_predict_click(self):
        input_data = {}
        try:
            # 遍历模型需要的每一个特征，从输入框提取数字
            for feat in self.required_features:
                val_str = self.entries[feat].get().strip()
                if not val_str:
                    raise ValueError(f"请填满所有参数。")
                input_data[feat] = float(val_str)

        except ValueError as e:
            messagebox.showwarning("输入格式错误", f"请输入有效的数值。\n{str(e)}")
            return

        self.status_label.configure(text="状态: 正在计算...", text_color="blue")
        self.update()

        try:
            # 传给后端预测！
            pred_value, warnings = self.predictor.predict(input_data)

            # 显示结果
            result_text = f"预测投矾量: {float(pred_value):.2f} kg"
            self.result_label.configure(text=result_text, text_color="#1f6aa5")

            if warnings:
                self.status_label.configure(text="状态: 预测完成 (警告: 结果疑似异常下限)", text_color="orange")
            else:
                self.status_label.configure(text="状态: 预测完成", text_color="green")

        except Exception as e:
            messagebox.showerror("预测失败", f"计算过程中发生错误:\n{str(e)}")
            self.status_label.configure(text="状态: 预测报错", text_color="red")


if __name__ == "__main__":
    app = WaterPredictorApp()
    app.mainloop()