#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""水厂投矾量预测系统桌面界面（含变频泵频率计算，原水量输入单位为 km³/h）。"""

import datetime
import threading
import customtkinter as ctk
from tkinter import messagebox

from predictor_service import WaterPredictor

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class WaterPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("水厂投矾量智能预测系统")
        self.geometry("1080x800")
        self.minsize(1000, 720)
        self.configure(fg_color="#edf1f4")

        self.predictor = None
        self.entries = {}
        self.summary_labels = {}

        self.result_value_var = ctk.StringVar(value="--")
        self.result_detail_var = ctk.StringVar(value="等待模型就绪")
        self.freq_value_var = ctk.StringVar(value="--")
        self.freq_detail_var = ctk.StringVar(value="待预测后计算")
        self.model_var = ctk.StringVar(value="模型未加载")
        self.status_var = ctk.StringVar(value="系统初始化中")
        self.message_var = ctk.StringVar(value="请先等待模型加载完成，再输入参数进行预测。")
        self.range_var = ctk.StringVar(
            value="建议范围: 浊度 0-100 NTU | 原水量 0-200000 | pH 0-14 | 温度 -10~50 °C"
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
            text="投矾量单次预测",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1f2937",
        ).pack(anchor="w")

        badge_frame = ctk.CTkFrame(header, fg_color="transparent")
        badge_frame.grid(row=0, column=1, sticky="e")
        badge_frame.grid_columnconfigure((0, 1), weight=1)

        self.model_badge = ctk.CTkLabel(
            badge_frame,
            textvariable=self.model_var,
            width=190,
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
            hint="格式: YYYY-MM-DD", placeholder="2026-03-31", columnspan=2
        )
        # 浊度
        self._create_input_field(
            parent=form, row=1, column=0, key="turbidity", title="浑浊度 (NTU)",
            hint="例如 21.5", placeholder="21.5"
        )
        # 原水量（km³/h）
        self._create_input_field(
            parent=form, row=1, column=1, key="flow", title="原水量 (km³/h)",
            hint="例如 0.3 (对应日处理 7.2 万立方米)", placeholder="0.3"
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
            hint="例如 0.1", placeholder="0.1", columnspan=2
        )
        # 冲程
        self._create_input_field(
            parent=form, row=4, column=0, key="stroke", title="冲程 (%)",
            hint="计量泵冲程百分比，默认65", placeholder="65", columnspan=2
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
        panel.grid_rowconfigure(2, weight=1)

        # 投矾量预测卡片
        result_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        result_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(
            result_card, text="预测结果", font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1f2937"
        ).pack(anchor="w", padx=14, pady=(12, 4))

        ctk.CTkLabel(
            result_card, textvariable=self.result_value_var,
            font=ctk.CTkFont(size=28, weight="bold"), text_color="#0f5e9c"
        ).pack(anchor="w", padx=14)

        # 结果详情标签：自动换行
        ctk.CTkLabel(
            result_card, textvariable=self.result_detail_var,
            width=400, anchor="w", justify="left", wraplength=400,
            font=ctk.CTkFont(size=12), text_color="#667085"
        ).pack(anchor="w", padx=14, pady=(2, 12))

        # 泵频率计算结果卡片
        freq_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        freq_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(
            freq_card, text="泵频率计算结果", font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1f2937"
        ).pack(anchor="w", padx=14, pady=(12, 4))

        ctk.CTkLabel(
            freq_card, textvariable=self.freq_value_var,
            font=ctk.CTkFont(size=28, weight="bold"), text_color="#0f5e9c"
        ).pack(anchor="w", padx=14)

        # 频率详情标签：自动换行
        ctk.CTkLabel(
            freq_card, textvariable=self.freq_detail_var,
            width=400, anchor="w", justify="left", wraplength=400,
            font=ctk.CTkFont(size=12), text_color="#667085"
        ).pack(anchor="w", padx=14, pady=(2, 12))

        # 输入摘要卡片
        summary_card = ctk.CTkFrame(panel, corner_radius=14, fg_color="#ffffff")
        summary_card.grid(row=2, column=0, sticky="nsew")
        summary_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            summary_card, text="本次输入摘要", font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1f2937"
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 8))

        self._create_summary_row(summary_card, 1, "日期", "date")
        self._create_summary_row(summary_card, 2, "浊度(NTU)", "turbidity")
        self._create_summary_row(summary_card, 3, "原水量(km³/h)", "flow")
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
        model_name = self.predictor.model_type or "预测模型"
        feature_count = len(self.predictor.features or [])
        self.model_var.set(f"{model_name} | 特征 {feature_count}")
        self.model_badge.configure(fg_color="#dff3e1", text_color="#2f6f3e")
        self.predict_btn.configure(state="normal")
        self._set_system_status("模型已就绪", level="success")
        self._set_message("模型加载完成，可以直接按 Enter 或点击“开始预测”。", level="success")

        ranges = getattr(self.predictor, "feature_ranges", {}) or {}
        range_parts = []
        mapping = [("浊度", "浊度"), ("原水量", "原水量"), ("pH", "pH"), ("温度", "温度")]
        for label, key in mapping:
            value_range = ranges.get(key)
            if value_range:
                range_parts.append(f"{label} {value_range[0]}-{value_range[1]}")
        if range_parts:
            self.range_var.set("建议范围: " + " | ".join(range_parts))

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

        hourly_water_km3 = float(self.entries["flow"].get().strip())
        daily_water_km3 = hourly_water_km3 * 24.0   # 转换为日原水量 (km³/d)

        values = {
            "日期": date_text,
            "浑浊度（NTU）_0点": float(self.entries["turbidity"].get().strip()),
            "原水量（Km³）": daily_water_km3,
            "温度（℃）_9点": float(self.entries["temperature"].get().strip()),
            "氨氮（mg/L）_9点": float(self.entries["ammonia"].get().strip()),
            "pH值_9点": float(self.entries["ph"].get().strip()),
            "冲程": float(self.entries["stroke"].get().strip()) if self.entries["stroke"].get().strip() else 65.0,
            "小时原水量_km3": hourly_water_km3,
        }
        return values

    def _update_summary(self, input_data):
        display_map = {
            "date": input_data.get("日期", "--"),
            "turbidity": self._format_metric(input_data.get("浑浊度（NTU）_0点"), ""),
            "flow": self._format_metric(input_data.get("小时原水量_km3", 0), "km³/h"),
            "temperature": self._format_metric(input_data.get("温度（℃）_9点"), ""),
            "ph": self._format_metric(input_data.get("pH值_9点"), ""),
            "ammonia": self._format_metric(input_data.get("氨氮（mg/L）_9点"), ""),
            "stroke": self._format_metric(input_data.get("冲程", ""), "%"),
        }
        for key, label in self.summary_labels.items():
            label.configure(text=display_map.get(key, "--"))

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
        self._set_message("系统正在调用本地预测模型，请稍候。", level="info")
        self.update_idletasks()

        try:
            # 模型预测（日投矾量 kg/d）
            daily_dosage, warnings = self.predictor.predict(input_data)

            stroke = input_data.get("冲程", 65.0)
            raw_water_km3 = input_data.get("原水量（Km³）", 0.0)  # 日原水量（千立方米）

            # 计算单位投加量 (kg/Km³)
            if raw_water_km3 > 0:
                unit_dosage = daily_dosage / raw_water_km3
            else:
                unit_dosage = 0.0

            # ----- 泵频率计算 -----
            hourly_dosage = daily_dosage / 24.0                     # kg/h
            pure_volume = hourly_dosage / 1.25                     # L/h (密度1.25 kg/L)
            diluted_volume = pure_volume * 4.0                     # L/h (1份原液+4份水)
            rated_flow = 1000.0                                    # L/h @50Hz, 100%冲程
            rated_freq = 50.0                                      # Hz
            if stroke > 0:
                actual_freq = rated_freq * (diluted_volume / rated_flow) * (100.0 / stroke)
            else:
                actual_freq = 0.0
            actual_freq = max(0.0, actual_freq)                    # 非负

            # 更新投矾量显示
            self.result_value_var.set(f"{daily_dosage:.2f} kg/d ({unit_dosage:.2f} kg/Km³)")
            self.result_detail_var.set(
                f"小时投加: {hourly_dosage:.2f} kg/h | 纯矾流量: {pure_volume:.2f} L/h | 稀释后流量: {diluted_volume:.2f} L/h"
            )
            # 更新频率显示
            self.freq_value_var.set(f"{actual_freq:.2f} Hz")
            self.freq_detail_var.set(
                f"冲程 {stroke:.1f}% | 目标流量 {diluted_volume:.2f} L/h | 额定 1000L/h @50Hz"
            )

            self._update_summary(input_data)

            if warnings:
                self._set_system_status("预测完成，输入存在越界提示", level="warning")
                self._set_message(f"预测已完成，但部分输入超出历史范围: {warnings}", level="warning")
            else:
                self._set_system_status("预测完成", level="success")
                self._set_message("预测及泵频率计算完成，可现场调试。", level="success")

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