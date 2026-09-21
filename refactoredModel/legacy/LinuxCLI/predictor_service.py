"""
文件名称：refactoredModel/LinuxCLI/predictor_service.py
所属类别：Linux CLI 部署代码 (Linux CLI Release)

功能描述：
    专为 Linux 服务器环境打包的模型决策推理封装服务类 `WaterPredictor`。
    支持在无图形界面的服务器端自动加载模型对应的四个关键权重文件 (`best_model.pkl`、`scaler.pkl`、`selected_features.pkl`、`imputer.pkl`)，
    并能够自动补全输入中缺失的高维特征，动态进行特征工程计算并对齐。

运行与使用方法：
    在 Python 中导入并实例化使用：
    from predictor_service import WaterPredictor
    predictor = WaterPredictor(model_dir='models')
    
    # 传入指标字典进行单次推理
    pred, warnings = predictor.predict(input_dict)

调用与依赖关系：
    - 被同级目录下的部署主程序 `cli_app.py` 调用。
    - 导入并调用同级目录下的 `utils.py` 完成特征工程的衍生变量添加工作。
    - 该目录是完全自包含的，可独立复制到 Linux 机器部署，无 customtkinter 图形依赖。

设计细节与关键备注：
    - 本文件与根目录的 predictor_service.py 逻辑完全一致，为保障 LinuxCLI 目录能独立完整地被部署并开箱即用，在此放置镜像。
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