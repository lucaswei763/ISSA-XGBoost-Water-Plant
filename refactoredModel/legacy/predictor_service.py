"""
文件名称：refactoredModel/predictor_service.py
所属类别：重构核心生产代码 (Refactored Core Production) / Linux CLI 部署代码镜像

功能描述：
    本项目模型加载与投药决策推理逻辑的底层封装服务类 `WaterPredictor`。
    支持：
    1. 在初始化时动态定位并加载模型目录下的四大金刚权重文件 (`best_model.pkl`、`scaler.pkl`、`selected_features.pkl`、`imputer.pkl`)；
    2. 支持预测输入时的高维特征自动补齐与缺省值中位数（`SimpleImputer` 训练统计值）自动填充；
    3. 后台调用 `utils.py` 进行 7 维衍生特征的实时特征工程计算，完成 19 维模型输入对齐。

运行与使用方法：
    在 Python 代码中导入并实例化：
    from predictor_service import WaterPredictor
    predictor = WaterPredictor(model_dir='models/resnet')
    
    # 执行预测，只需传入基础测得指标
    input_data = {
        "日期": "2026-06-06",
        "浑浊度（NTU）": 21.5,
        "原水量（Km³）": 7.2
    }
    daily_dosage, has_warnings = predictor.predict(input_data)

调用与依赖关系：
    - 被 `ui_app.py`、`cli_app.py`、`app.py` 导入和调用，是所有预测入口的核心依赖。
    - 导入并依赖于 `utils.add_engineered_features` 进行 7 维衍生特征计算。
    - 依赖 `joblib` 加载序列化模型。

设计细节与关键备注：
    - 带有对 PyInstaller 打包环境的自适应检查（通过 `getattr(sys, 'frozen', False)` ），支持读取解压路径 `_MEIPASS` 下的权重。
    - 具有智能路径解析功能，如果指定默认 `models` 目录，则在有 `resnet` 或 `xgboost` 子目录存在时，自动切换到对应的最佳模型目录。
    - 在 `predict` 中，如果模型输出值小于 0，则自动裁剪并置为 0.0 并抛出范围警告。
"""
import os
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class WaterPredictor:
    def __init__(self, model_dir='models'):
        import sys
        import os
        
        # Check PyInstaller environment
        if getattr(sys, 'frozen', False):
            # 获取 PyInstaller 运行时的临时/解压目录
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            resolved_model_dir = os.path.join(base_dir, model_dir)
        else:
            resolved_model_dir = model_dir

        # 智能路径解析：如果指定的是默认 models 目录但存在子目录，则自动切换
        if model_dir == 'models':
            if os.path.exists(os.path.join(resolved_model_dir, 'resnet', 'best_model.pkl')):
                resolved_model_dir = os.path.join(resolved_model_dir, 'resnet')
            elif os.path.exists(os.path.join(resolved_model_dir, 'xgboost', 'best_model.pkl')):
                resolved_model_dir = os.path.join(resolved_model_dir, 'xgboost')
                
        self.model_dir = resolved_model_dir

        # 加载三大件
        try:
            # 确保当前目录在 sys.path 中，以便 joblib.load 能够反序列化 ResNetRegressor
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)

            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_dir, 'imputer.pkl'))
            
            # 动态检测模型类别
            if hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'ResNetRegressor':
                self.model_type = "ResNet (残差神经网络)"
            else:
                self.model_type = "ISSA-XGBoost"
        except Exception as e:
            raise RuntimeError(
                f"加载模型文件失败，请确保 {self.model_dir} 文件夹下有 best_model.pkl, scaler.pkl, selected_features.pkl, imputer.pkl。详情: {e}")

    def get_required_features(self):
        """返回模型需要的特征列表，告诉 UI 需要生成哪些输入框"""
        return self.features

    def predict(self, input_dict):
        """
        执行预测，对于未在 input_dict 中提供的高维输入特征，自动采用训练集的中位数（Imputer statistics）填充，
        并在后台自动计算新增加的 7 个特征工程列，支持 19 维特征输入。
        :param input_dict: 字典格式，键为特征名，值为浮点数
        :return: (预测值, 是否有警告)
        """
        # 1. 自动填充 12 个原始基础特征
        base_values = {}
        num_base = len(self.imputer.statistics_)
        base_features = self.features[:num_base]
        
        for idx, f in enumerate(base_features):
            if f not in input_dict:
                # 自动从 imputer 的中位数中加载该维度的缺省值
                fallback_val = self.imputer.statistics_[idx]
                base_values[f] = fallback_val
            else:
                base_values[f] = input_dict[f]

        # 2. 动态计算 7 个衍生特征工程列
        from utils import add_engineered_features
        full_values = add_engineered_features(base_values)

        # 3. 按最终特征列表顺序重组输入向量 (19维)
        feature_vector = [full_values[f] for f in self.features]

        # 4. 转换为二维数组并进行归一化
        X_arr = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)

        # 预测
        pred_value = self.model.predict(X_scaled)[0]

        # 简单的范围警告检查 (比如预测结果小于0肯定不合理)
        has_warnings = False
        if pred_value < 0:
            pred_value = 0.0
            has_warnings = True

        return pred_value, has_warnings