# predictor_service.py
import os
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class WaterPredictor:
    def __init__(self, model_dir='models'):
        import sys
        if getattr(sys, 'frozen', False):
            # 获取 PyInstaller 运行时的临时/解压目录
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            self.model_dir = os.path.join(base_dir, model_dir)
        else:
            self.model_dir = model_dir

        # 加载三大件
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))
            self.model_type = "ISSA-XGBoost"
        except Exception as e:
            raise RuntimeError(
                f"加载模型文件失败，请确保 {self.model_dir} 文件夹下有 best_model.pkl, scaler.pkl, selected_features.pkl。详情: {e}")

    def get_required_features(self):
        """返回模型需要的特征列表，告诉 UI 需要生成哪些输入框"""
        return self.features

    def predict(self, input_dict):
        """
        执行预测
        :param input_dict: 字典格式，键为特征名，值为浮点数
        :return: (预测值, 是否有警告)
        """
        feature_vector = []
        for f in self.features:
            if f not in input_dict:
                raise ValueError(f"缺少模型必须的特征参数: {f}")
            feature_vector.append(input_dict[f])

        # 转换为二维数组并进行归一化
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