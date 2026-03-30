#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量预测系统 - 本地桌面版 UI
"""

import os
import datetime
import threading
import customtkinter as ctk
from tkinter import messagebox

# 导入我们刚刚优化好的本地预测服务
from predictor_service import WaterPredictor

# ==================== UI 全局设置 ====================
ctk.set_appearance_mode("System")  # 跟随系统主题 (深色/浅色)
ctk.set_default_color_theme("blue")  # 主题色


class WaterPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 窗口基础设置 ---
        self.title("水厂投矾量智能预测系统")
        self.geometry("500x600")
        self.resizable(False, False)  # 固定窗口大小，防止布局乱掉

        # 模型实例占位符
        self.predictor = None

        # --- 构建 UI 布局 ---
        self._build_ui()

        # --- 异步加载模型 ---
        # 使用多线程加载模型，防止刚打开软件时界面卡死白屏
        self.status_label.configure(text="状态: 正在加载预测模型...", text_color="orange")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _build_ui(self):
        """构建界面元素"""
        # 标题栏
        self.title_label = ctk.CTkLabel(self, text="投矾量单次预测", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(20, 20))

        # 核心表单区域 (使用 Frame 包装)
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.pack(padx=40, pady=10, fill="both", expand=True)

        # 定义需要输入的字段
        self.entries = {}
        fields = [
            ("日期 (YYYY-MM-DD)", "date"),
            ("原水浊度 (NTU)", "turbidity"),
            ("进水流量 (m³/h)", "flow"),
            ("原水 pH 值", "ph"),
            ("原水温度 (°C)", "temperature")
        ]

        for i, (label_text, key) in enumerate(fields):
            # 标签
            lbl = ctk.CTkLabel(self.form_frame, text=label_text, font=ctk.CTkFont(size=14))
            lbl.grid(row=i, column=0, padx=20, pady=15, sticky="e")

            # 输入框
            entry = ctk.CTkEntry(self.form_frame, width=200)
            entry.grid(row=i, column=1, padx=20, pady=15, sticky="w")
            self.entries[key] = entry

        # 自动填入当前日期
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.entries["date"].insert(0, today_str)

        # 预测按钮
        self.predict_btn = ctk.CTkButton(
            self,
            text="开始预测",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            state="disabled",  # 模型没加载完之前禁用
            command=self._on_predict_click
        )
        self.predict_btn.pack(pady=(20, 10))

        # 结果显示框
        self.result_label = ctk.CTkLabel(
            self,
            text="等待输入数据...",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="gray"
        )
        self.result_label.pack(pady=(10, 10))

        # 底部状态栏
        self.status_label = ctk.CTkLabel(self, text="状态: 初始化", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="bottom", pady=10)

    def _load_model_thread(self):
        """在后台线程加载模型"""
        import traceback
        try:
            print("=> [后台线程] 开始初始化预测引擎...")
            self.predictor = WaterPredictor()
            print("=> [后台线程] 预测引擎初始化完成！")
            # 加载成功后，在主线程更新 UI
            self.after(0, self._on_model_loaded_success)

        except Exception as e:
            print("=> [后台线程] 初始化发生严重错误！详情如下：")
            traceback.print_exc()  # 把完整的报错栈打印到命令行控制台

            # 提前把错误转成字符串，避开 Python 3 的异常变量销毁机制
            error_msg = str(e)
            # 强制绑定变量 err=error_msg
            self.after(0, lambda err=error_msg: self._on_model_loaded_fail(err))

    def _on_model_loaded_success(self):
        self.status_label.configure(text=f"状态: 模型就绪 ({self.predictor.model_type})", text_color="green")
        self.predict_btn.configure(state="normal")

    def _on_model_loaded_fail(self, error_msg):
        self.status_label.configure(text="状态: 模型加载失败", text_color="red")
        messagebox.showerror("致命错误", f"无法加载预测模型，请检查 models 目录。\n\n详情: {error_msg}")

    def _on_predict_click(self):
        """点击预测按钮的响应逻辑"""
        # 1. 收集并校验数据
        input_data = {}
        try:
            date_val = self.entries["date"].get().strip()
            if not date_val:
                raise ValueError("日期不能为空")
            input_data["日期"] = date_val

            # 转换为浮点数
            input_data["turbidity"] = float(self.entries["turbidity"].get().strip())
            input_data["flow"] = float(self.entries["flow"].get().strip())
            input_data["ph"] = float(self.entries["ph"].get().strip())
            input_data["temperature"] = float(self.entries["temperature"].get().strip())

            # 为了兼容模型，补齐可选参数（使用默认值）
            input_data["ammonia"] = 0.0
            input_data["water_level"] = 30.0
            input_data["alum_per_unit"] = 0.0

        except ValueError as e:
            messagebox.showwarning("输入格式错误", f"请确保输入的是有效的数值。\n{str(e)}")
            return

        # 2. 执行预测
        self.status_label.configure(text="状态: 正在计算...", text_color="blue")
        self.update()  # 强制刷新UI

        try:
            # 调用 predictor_service
            pred_value, warnings = self.predictor.predict(input_data)

            # 3. 显示结果
            result_text = f"预测投矾量: {float(pred_value):.2f} kg"
            self.result_label.configure(text=result_text, text_color="#1f6aa5")

            # 如果有越界警告
            if warnings:
                self.status_label.configure(text="状态: 预测完成 (部分输入超出历史范围)", text_color="orange")
            else:
                self.status_label.configure(text="状态: 预测完成", text_color="green")

        except Exception as e:
            messagebox.showerror("预测失败", f"计算过程中发生错误:\n{str(e)}")
            self.status_label.configure(text="状态: 预测报错", text_color="red")


if __name__ == "__main__":
    app = WaterPredictorApp()
    app.mainloop()