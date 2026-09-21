# ui_app.py - 修改后的版本
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""水厂投矾量预测系统桌面界面（含变频泵频率计算）。"""

import datetime
import threading
import matplotlib
matplotlib.use('Agg')  # 无界面后端，防打包后闪退
import customtkinter as ctk
from tkinter import messagebox

from predictor_service import WaterPredictor

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class WaterPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("水厂投矾量智能预测系统 v2.0")
        self.geometry("1200x920")
        self.minsize(1080, 820)
        self.configure(fg_color="#edf1f4")

        self.predictor = None
        self.entries = {}
        self.summary_labels = {}

        self.result_value_var = ctk.StringVar(value="--")
        self.result_detail_var = ctk.StringVar(value="等待模型就绪")
        self.freq_value_var = ctk.StringVar(value="--")
        self.freq_detail_var = ctk.StringVar(value="待预测后计算")
        self.model_var = ctk.StringVar(value="模型加载中...")
        self.status_var = ctk.StringVar(value="系统初始化中")
        self.message_var = ctk.StringVar(value="请先等待模型加载完成，再输入参数进行预测。")
        self.range_var = ctk.StringVar(
            value="建议范围: 浊度 0-100 NTU | 原水量 5-20 Km³/h | pH 6-8 | 温度 -10~50 °C"
        )
        self._build_ui()
        self._fill_today()
        self.bind("<Return>", lambda _event: self._on_predict_click())

        self._set_system_status("正在加载预测模型...", level="warning")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        header.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_frame,
            text="投矾量智能预测 (LSTM+XGBoost)",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1f2937",
        ).pack(anchor="w")

        badge_frame = ctk.CTkFrame(header, fg_color="transparent")
        badge_frame.grid(row=0, column=1, sticky="e")
        badge_frame.grid_columnconfigure((0, 1), weight=1)

        self.model_badge = ctk.CTkLabel(
            badge_frame,
            textvariable=self.model_var,
            width=220,
            height=30,
            corner_radius=10,
            fg_color="#dbe7f5",
            text_color="#1f4e79",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.model_badge.grid(row=0, column=0, padx=(0, 8))

        self.state_badge = ctk.CTkLabel(
            badge_frame,
            textvariable=self.status_var,
            width=190,
            height=30,
            corner_radius=10,
            fg_color="#fff1cf",
            text_color="#8a5a00",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.state_badge.grid(row=0, column=1)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        content.grid_columnconfigure(0, weight=3, uniform="main_panes", minsize=560)
        content.grid_columnconfigure(1, weight=2, uniform="main_panes", minsize=360)
        content.grid_rowconfigure(0, weight=1)

        self._build_input_panel(content)
        self._build_result_panel(content)

    def _build_input_panel(self, parent):
        panel = ctk.CTkFrame(parent, corner_radius=14, fg_color="#ffffff")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text="工艺参数录入",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1f2937",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))

        form = ctk.CTkFrame(panel, fg_color="transparent")
        form.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 6))
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        # 日期
        self._create_input_field(
            parent=form, row=0, column=0, key="date", title="日期",
            hint="格式: YYYY-MM-DD", placeholder="2026-01-15", columnspan=2
        )
        # 浊度
        self._create_input_field(
            parent=form, row=1, column=0, key="turbidity", title="浑浊度 (NTU)",
            hint="例如 21.5", placeholder="21.5"
        )
        # 原水量（Km³/小时）
        self._create_input_field(
            parent=form, row=1, column=1, key="water_supply", title="原水量 (Km³/h)",
            hint="每小时水量，例如 10.0", placeholder="10.0"
        )
        # 温度
        self._create_input_field(
            parent=form, row=2, column=0, key="temperature", title="温度 (℃)",
            hint="单位: °C", placeholder="18.0"
        )
        # pH
        self._create_input_field(
            parent=form, row=2, column=1, key="ph", title="pH值",
            hint="建议输入 0-14", placeholder="7.2"
        )
        # 氨氮
        self._create_input_field(
            parent=form, row=3, column=0, key="ammonia", title="氨氮 (mg/L)",
            hint="例如 0.1", placeholder="0.1"
        )
        # 冲程
        self._create_input_field(
            parent=form, row=3, column=1, key="stroke", title="冲程 (%)",
            hint="量程比默认65%", placeholder="65"
        )

        action_frame = ctk.CTkFrame(panel, fg_color="transparent")
        action_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(2, 12))
        action_frame.grid_columnconfigure(0, weight=3)
        action_frame.grid_columnconfigure(1, weight=1)
        action_frame.grid_columnconfigure(2, weight=1)

        self.predict_btn = ctk.CTkButton(
            action_frame, text="开始预测", height=38,
            font=ctk.CTkFont(size=14, weight="bold"), state="disabled",
            command=self._on_predict_click
        )
        self.predict_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.today_btn = ctk.CTkButton(
            action_frame, text="今天", height=38,
            fg_color="#d9e5f2", hover_color="#c9d9ea", text_color="#1f4e79",
            command=self._fill_today
        )
        self.today_btn.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.clear_btn = ctk.CTkButton(
            action_frame, text="清空", height=38,
            fg_color="#e5e7eb", hover_color="#d7dbe2", text_color="#374151",
            command=self._reset_fields
        )
        self.clear_btn.grid(row=0, column=2, sticky="ew")

    def _build_result_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        panel.grid_columnconfigure(0, weight=1)

        # === 预测结果卡片 ===
        result_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        result_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(result_card, text="预测结果", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color="#1f2937").pack(anchor="w", padx=14, pady=(12, 2))

        ctk.CTkLabel(result_card, textvariable=self.result_value_var,
                     font=ctk.CTkFont(size=28, weight="bold"),
                     text_color="#0f5e9c").pack(anchor="w", padx=14)

        # 投加详情 + 置信度 一行
        info_row = ctk.CTkFrame(result_card, fg_color="transparent")
        info_row.pack(fill="x", padx=14, pady=(4, 2))

        self.result_detail_var = ctk.StringVar(value="小时投加: -- kg/h | 纯矾: -- L/h")
        ctk.CTkLabel(info_row, textvariable=self.result_detail_var,
                     font=ctk.CTkFont(size=11), text_color="#667085").pack(side="left")

        self.confidence_badge = ctk.CTkLabel(info_row, text="",
            width=80, height=22, corner_radius=8, font=ctk.CTkFont(size=10, weight="bold"))
        self.confidence_badge.pack(side="right", padx=(8, 0))

        # 趋势预警标签
        self.trend_label = ctk.CTkLabel(result_card, text="",
            font=ctk.CTkFont(size=10, weight="bold"))
        self.trend_label.pack(anchor="w", padx=14, pady=(0, 4))

        # 记录实际值
        record_row = ctk.CTkFrame(result_card, fg_color="transparent")
        record_row.pack(fill="x", padx=14, pady=(0, 8))
        self.actual_entry = ctk.CTkEntry(record_row, placeholder_text="实际投矾量(kg)", width=120, height=28)
        self.actual_entry.pack(side="left", padx=(0, 6))
        self.record_btn = ctk.CTkButton(record_row, text="记录实际值", width=90, height=28, font=ctk.CTkFont(size=11), command=self._on_record_actual)
        self.record_btn.pack(side="left")

        # === SHAP 解释卡片 ===
        shap_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        shap_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(shap_card, text="为什么是这个数？",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#1f2937").pack(anchor="w", padx=14, pady=(10, 2))

        ctk.CTkLabel(shap_card, text="特征               SHAP贡献      输入值",
                     font=ctk.CTkFont(size=10), text_color="#9ca3af"
                     ).pack(anchor="w", padx=14, pady=(0, 2))

        self.shap_lines = ctk.CTkFrame(shap_card, fg_color="transparent")
        self.shap_lines.pack(fill="x", padx=10, pady=(0, 8))
        self.shap_widgets = []

        # === 泵频率卡片 ===
        freq_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        freq_card.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(freq_card, text="泵频率计算结果", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#1f2937").pack(anchor="w", padx=14, pady=(10, 2))

        ctk.CTkLabel(freq_card, textvariable=self.freq_value_var,
                     font=ctk.CTkFont(size=28, weight="bold"),
                     text_color="#0f5e9c").pack(anchor="w", padx=14)

        ctk.CTkLabel(freq_card, textvariable=self.freq_detail_var,
                     width=400, anchor="w", justify="left",
                     font=ctk.CTkFont(size=11), text_color="#667085"
                     ).pack(anchor="w", padx=14, pady=(2, 10))

        # === 输入摘要卡片 ===
        summary_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        summary_card.grid(row=3, column=0, sticky="nsew")
        summary_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            summary_card, text="本次输入摘要", font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1f2937"
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 8))

        self._create_summary_row(summary_card, 1, "日期", "date")
        self._create_summary_row(summary_card, 2, "浊度(NTU)", "turbidity")
        self._create_summary_row(summary_card, 3, "原水量(Km³/h)", "water_supply")
        self._create_summary_row(summary_card, 4, "温度(℃)", "temperature")
        self._create_summary_row(summary_card, 5, "pH值", "ph")
        self._create_summary_row(summary_card, 6, "氨氮(mg/L)", "ammonia")
        self._create_summary_row(summary_card, 7, "冲程(%)", "stroke")

    # ---------- 辅助方法 ----------
    def _create_input_field(self, parent, row, column, key, title, hint, placeholder, columnspan=1):
        frame = ctk.CTkFrame(parent, corner_radius=10, fg_color="#f6f7f9")
        frame.grid(row=row, column=column, columnspan=columnspan, sticky="nsew",
                   padx=(0, 8) if column == 0 and columnspan == 1 else 0, pady=(0, 6))

        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#1f2937").pack(anchor="w", padx=10, pady=(8, 2))

        entry = ctk.CTkEntry(frame, height=32, border_width=1, fg_color="#ffffff",
                             border_color="#cfd8e3", placeholder_text=placeholder)
        entry.pack(fill="x", padx=10, pady=(0, 2))
        entry.bind("<Return>", lambda _event: self._on_predict_click())

        ctk.CTkLabel(frame, text=hint, font=ctk.CTkFont(size=10),
                     text_color="#667085").pack(anchor="w", padx=10, pady=(0, 8))

        self.entries[key] = entry

    def _create_summary_row(self, parent, row, title, key):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=12),
                     text_color="#5f6b7a").grid(row=row, column=0, sticky="w", padx=14, pady=4)

        value_label = ctk.CTkLabel(parent, text="--", font=ctk.CTkFont(size=12, weight="bold"),
                                   text_color="#1f2937")
        value_label.grid(row=row, column=1, sticky="e", padx=14, pady=4)
        self.summary_labels[key] = value_label

    def _fill_today(self):
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.entries["date"].delete(0, "end")
        self.entries["date"].insert(0, today_str)

    def _reset_fields(self):
        for key, entry in self.entries.items():
            entry.delete(0, "end")
            if key == "date":
                entry.insert(0, datetime.datetime.now().strftime("%Y-%m-%d"))
        self.result_value_var.set("--")
        self.result_detail_var.set("等待新的预测任务")
        self.freq_value_var.set("--")
        self.freq_detail_var.set("待预测后计算")
        self._set_message("已清空数值输入，请重新录入工艺参数。", level="info")
        self._update_summary({})
        self._set_system_status("输入已清空", level="info")

    def _set_system_status(self, message, level="info"):
        palette = {
            "info": ("#dbe7f5", "#1f4e79"),
            "success": ("#dff3e1", "#2f6f3e"),
            "warning": ("#fff1cf", "#8a5a00"),
            "error": ("#f8d7da", "#8b2c35"),
        }
        fg_color, text_color = palette.get(level, palette["info"])
        self.status_var.set(message)
        self.state_badge.configure(fg_color=fg_color, text_color=text_color)

    def _set_message(self, text, level="info"):
        self.message_var.set(text)

    def _load_model_thread(self):
        try:
            self.predictor = WaterPredictor()
            self.after(0, self._on_model_loaded_success)
        except Exception as exc:
            self.after(0, lambda err=str(exc): self._on_model_loaded_fail(err))

    def _on_model_loaded_success(self):
        info = self.predictor.get_model_info()
        model_name = info.get('model_type', '预测模型')
        use_stacked = info.get('use_stacked', False)
        feature_count = len(info.get('features', []))

        model_display = f"{model_name}"
        if use_stacked:
            model_display += " ⭐"
        self.model_var.set(f"{model_display} | {feature_count}特征")
        self.model_badge.configure(fg_color="#dff3e1", text_color="#2f6f3e")
        self.predict_btn.configure(state="normal")
        self._set_system_status("模型已就绪", level="success")
        self._set_message(f"模型加载完成: {model_name}", level="success")

    def _on_model_loaded_fail(self, error_msg):
        self.model_var.set("模型加载失败")
        self.model_badge.configure(fg_color="#f8d7da", text_color="#8b2c35")
        self._set_system_status("模型不可用", level="error")
        self._set_message("无法加载模型，请检查 models 目录和依赖文件是否完整。", level="error")
        messagebox.showerror("致命错误", f"无法加载预测模型。\n\n详情: {error_msg}")

    def _collect_input_data(self):
        date_text = self.entries["date"].get().strip()
        if not date_text:
            raise ValueError("日期不能为空")
        datetime.datetime.strptime(date_text, "%Y-%m-%d")

        # 收集所有输入
        turbidity_val = float(self.entries["turbidity"].get().strip())
        water_supply_val = float(self.entries["water_supply"].get().strip())
        temp_val = float(self.entries["temperature"].get().strip())
        ammonia_val = float(self.entries["ammonia"].get().strip())
        ph_val = float(self.entries["ph"].get().strip())
        stroke_val = float(self.entries["stroke"].get().strip()) if self.entries["stroke"].get().strip() else 65.0

        values = {
            "日期": date_text,
            "浑浊度_0点": turbidity_val,
            "原水量_Km3_hour": water_supply_val,
            "温度_C": temp_val,
            "氨氮_mg_per_L": ammonia_val,
            "pH值": ph_val,
            # 兼容旧模型
            "浑浊度（NTU）_0点": turbidity_val,
            "原水量（Km³/h）": water_supply_val,
            "温度（℃）_9点": temp_val,
            "氨氮（mg/L）_9点": ammonia_val,
            "pH值_9点": ph_val,
            "冲程": stroke_val,
        }
        return values

    def _update_summary(self, input_data):
        display_map = {
            "date": input_data.get("日期", "--"),
            "turbidity": self._format_metric(input_data.get("浑浊度_0点", input_data.get("浑浊度（NTU）_0点")), "NTU"),
            "water_supply": self._format_metric(input_data.get("原水量_Km3_hour", input_data.get("原水量（Km³/h）")), "Km³/h"),
            "temperature": self._format_metric(input_data.get("温度_C", input_data.get("温度（℃）_9点")), "℃"),
            "ph": self._format_metric(input_data.get("pH值", input_data.get("pH值_9点")), ""),
            "ammonia": self._format_metric(input_data.get("氨氮_mg_per_L", input_data.get("氨氮（mg/L）_9点")), "mg/L"),
            "stroke": self._format_metric(input_data.get("冲程"), "%"),
        }
        for key, label in self.summary_labels.items():
            label.configure(text=display_map.get(key, "--"))

    def _on_record_actual(self):
        """记录实际投矾量到运行时数据库"""
        val = self.actual_entry.get().strip()
        if not val:
            messagebox.showwarning("提示", "请输入实际投矾量")
            return
        try:
            actual = float(val)
        except ValueError:
            messagebox.showwarning("格式错误", "请输入有效数字")
            return
        date_str = self.entries.get("date", None)
        if not date_str:
            messagebox.showwarning("提示", "请先填入日期")
            return
        date_str = date_str.get().strip()
        if self.predictor:
            self.predictor.record_actual(date_str, actual)
            self._set_message(f"已记录 {date_str} 实际投矾量: {actual:.0f} kg", level="success")
            self.actual_entry.delete(0, "end")
        else:
            messagebox.showwarning("错误", "模型未加载")

    @staticmethod
    def _format_metric(value, unit):
        if value in (None, "--"):
            return "--"
        if isinstance(value, (int, float)):
            formatted = f"{value:g}"
        else:
            formatted = str(value)
        return f"{formatted} {unit}".strip()

    def _on_predict_click(self):
        if self.predictor is None:
            messagebox.showinfo("提示", "模型仍在加载中，请稍候。")
            return

        try:
            input_data = self._collect_input_data()
        except ValueError as exc:
            messagebox.showwarning("输入格式错误", f"请检查输入内容。\n\n{exc}")
            self._set_system_status("输入校验未通过", level="warning")
            self._set_message("日期必须为 YYYY-MM-DD，数值字段必须填写有效数字。", level="warning")
            return

        self.predict_btn.configure(state="disabled")
        self._set_system_status("正在计算预测结果...", level="info")
        self._set_message("系统正在调用预测模型，请稍候。", level="info")
        self.update_idletasks()

        try:
            # 模型预测 + SHAP解释 + 置信度 + 趋势预警
            result = self.predictor.explain(input_data)
            daily_dosage = result.get('prediction')
            if daily_dosage is None:
                raise ValueError("模型预测返回空值")

            warnings = result.get('warnings', [])

            stroke = input_data.get("冲程", 65.0) or 65.0
            raw_water_hour = input_data.get("原水量_Km3_hour", input_data.get("原水量（Km³/h）", 1.0)) or 1.0

            # 计算单位投加量 (kg/km3)：日投矾量 / 日原水量
            raw_water_daily = raw_water_hour * 24.0  # 转换为日值
            if raw_water_daily > 0:
                unit_dosage = daily_dosage / raw_water_daily
            else:
                unit_dosage = 0.0

            # 泵频率计算（按小时投加量）
            hourly_dosage = daily_dosage / 24.0
            pure_volume = hourly_dosage / 1.25
            diluted_volume = pure_volume * 4.0
            rated_flow = 1000.0
            rated_freq = 50.0
            if stroke > 0:
                actual_freq = rated_freq * (diluted_volume / rated_flow) * (100.0 / stroke)
            else:
                actual_freq = 0.0
            actual_freq = max(0.0, actual_freq)

            self.result_value_var.set(f"{daily_dosage:.2f} kg/d ({unit_dosage:.3f} kg/km3)")
            self.result_detail_var.set(f"小时投加: {hourly_dosage:.0f} kg/h | 纯矾: {pure_volume:.0f} L/h | 稀释: {diluted_volume:.0f} L/h")
            # 预测区间
            lb = result.get('lower_bound')
            ub = result.get('upper_bound')
            if lb and ub:
                self.result_detail_var.set(
                    f"90%置信区间: {lb:.0f} ~ {ub:.0f} kg  |  "
                    f"小时: {hourly_dosage:.0f} kg/h | 纯矾: {pure_volume:.0f} L/h"
                )
            self.freq_value_var.set(f"{actual_freq:.2f} Hz")
            self.freq_detail_var.set(
                f"冲程 {stroke:.1f}% | 目标流量 {diluted_volume:.2f} L/h | 额定 1000L/h @50Hz"
            )

            # === 置信度徽章 ===
            conf = result.get('confidence', {})
            conf_level = conf.get('level', 'high')
            conf_pct = conf.get('confidence', 1.0) * 100
            conf_colors = {'high': ('#dff3e1', '#2f6f3e'), 'medium': ('#e8f0fe', '#1a56db'), 'low': ('#f8d7da', '#8b2c35')}
            cf_bg, cf_fg = conf_colors.get(conf_level, conf_colors['medium'])
            badge_icon = '●' if conf_level != 'low' else '⚠'
            self.confidence_badge.configure(
                text=f"{badge_icon} {conf_pct:.0f}%",
                fg_color=cf_bg, text_color=cf_fg
            )

            # === 趋势预警 ===
            trend = result.get('trend_alert')
            if trend:
                self.trend_label.configure(
                    text=f"⚠ {trend['message']}",
                    text_color="#8b2c35" if trend['level'] == 'critical' else "#8a5a00"
                )
            else:
                self.trend_label.configure(text="")

            # === SHAP 解释条 ===
            for w in self.shap_widgets:
                w.destroy()
            self.shap_widgets.clear()
            for c in result.get('contributions', [])[:6]:
                row = ctk.CTkFrame(self.shap_lines, fg_color="transparent")
                row.pack(fill="x", pady=1)
                # 特征名
                name_map = {'浑浊度_0点':'浑浊度', '原水量_Km3':'原水量', '温度_C':'温度',
                    '氨氮_mg_per_L':'氨氮', 'pH值':'pH值', 'lag1_unit':'昨日单位投矾',
                    'lag2_unit':'前日单位投矾', 'lag1':'昨日投矾',
                    'lag2':'前日投矾'}
                ctk.CTkLabel(row, text=name_map.get(c['feature'],c['feature'])[:8],
                    width=70, anchor="w", font=ctk.CTkFont(size=11),
                    text_color="#374151").pack(side="left")
                # SHAP条
                shap_val = c['shap']
                max_shap = max(abs(x['shap']) for x in result['contributions'][:6]) or 1
                bar_w = int(abs(shap_val) / max_shap * 120)
                bar_color = "#2563eb" if shap_val >= 0 else "#dc2626"
                bar = ctk.CTkFrame(row, width=bar_w, height=14, fg_color=bar_color, corner_radius=4)
                bar.pack(side="left", padx=(4, 4))
                bar.pack_propagate(False)
                # 数值
                ctk.CTkLabel(row, text=f"{shap_val:+.0f} kg", width=60, anchor="e",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#2563eb" if shap_val >= 0 else "#dc2626").pack(side="left", padx=(0,8))
                # 输入值
                ctk.CTkLabel(row, text=f"{c['value']:.1f}",
                    font=ctk.CTkFont(size=10), text_color="#9ca3af").pack(side="left")
                self.shap_widgets.extend([row, bar])

            # === 异常特征标记 ===
            anomaly_flags = conf.get('anomaly_flags', [])
            if anomaly_flags:
                flags_text = "异常参数: " + ", ".join(f['feature'] for f in anomaly_flags[:3])
                flag_row = ctk.CTkFrame(self.shap_lines, fg_color="transparent")
                flag_row.pack(fill="x", pady=(4,0))
                ctk.CTkLabel(flag_row, text=flags_text, font=ctk.CTkFont(size=10, weight="bold"),
                    text_color="#dc2626").pack(anchor="w")
                self.shap_widgets.append(flag_row)

            # === 历史基准线 ===
            lag_info = []
            for c in result['contributions']:
                if c['feature'] == 'lag1_unit' and c['value'] > 0:
                    lag_info.append(f"昨日单位投矾: {c['value']:.1f} kg/Km³")
                if c['feature'] == 'lag2_unit' and c['value'] > 0:
                    lag_info.append(f"前日单位投矾: {c['value']:.1f} kg/Km³")
                if c['feature'] == 'lag1' and c['value'] > 0:
                    lag_info.append(f"昨日实际投矾: {c['value']:.0f} kg")
                if c['feature'] == 'lag2' and c['value'] > 0:
                    lag_info.append(f"前日实际投矾: {c['value']:.0f} kg")
            if lag_info:
                lag_row = ctk.CTkFrame(self.shap_lines, fg_color="transparent")
                lag_row.pack(fill="x", pady=(4, 0))
                ctk.CTkLabel(lag_row, text=" | ".join(lag_info),
                    font=ctk.CTkFont(size=10), text_color="#6b7280").pack(anchor="w")
                self.shap_widgets.append(lag_row)
            self.freq_detail_var.set(
                f"冲程 {stroke:.1f}% | 目标流量 {diluted_volume:.2f} L/h | 额定 1000L/h @50Hz"
            )

            self._update_summary(input_data)

            # 警告和趋势
            status_msgs = []
            if warnings:
                status_msgs.append("输入越界")
            trend = result.get('trend_alert')
            if trend:
                status_msgs.append(f"趋势{trend['level']}: {trend['delta_pct']:.0f}%")
                self._set_message(f"⚠ 趋势预警: {trend['message']}", level="warning")
            elif warnings:
                self._set_message(f"预测已完成，但部分输入超出历史范围: {warnings}", level="warning")
            else:
                self._set_message("预测及泵频率计算完成，可现场调试。", level="success")

            if status_msgs:
                self._set_system_status("预测完成 | " + " | ".join(status_msgs), level="warning")
            else:
                self._set_system_status("预测完成", level="success")

        except Exception as exc:
            self.result_value_var.set("--")
            self.result_detail_var.set("计算失败")
            self.freq_value_var.set("--")
            self.freq_detail_var.set("预测出错")
            self._set_system_status("预测报错", level="error")
            self._set_message(f"计算过程中出现错误: {exc}", level="error")
            messagebox.showerror("预测失败", f"计算过程中发生错误。\n\n{exc}")
        finally:
            if self.predictor is not None:
                self.predict_btn.configure(state="normal")


if __name__ == "__main__":
    app = WaterPredictorApp()
    app.mainloop()