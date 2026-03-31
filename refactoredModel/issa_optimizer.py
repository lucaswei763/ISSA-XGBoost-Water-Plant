
# issa_optimizer.py
import numpy as np
import math
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


class ISSA_XGBoost:
    def __init__(self, pop_size=20, max_iter=30, bounds=None, cv_splits=5):
        """
        :param pop_size: 种群数量 (麻雀数)
        :param max_iter: 最大迭代次数
        :param bounds: 超参数边界 [(min, max), ...]
        """
        self.pop_size = pop_size
        self.max_iter = max_iter

        # 👑 核心修改 1：收紧树深限制，增加 reg_lambda (L2正则化) 压制过拟合
        self.bounds = bounds or [
            (50, 350),  # pos[0]: n_estimators (稍微增加树的数量上限)
            (3, 8),  # pos[1]: max_depth (上限从 6 放宽到 8)
            (0.01, 0.15),  # pos[2]: learning_rate
            (0.6, 0.9),  # pos[3]: subsample (稍微提高采样率，减少随机性带来的平滑)
            (0.6, 0.9),  # pos[4]: colsample_bytree
            (0.1, 5.0)  # pos[5]: reg_lambda (下限从 1.0 降到 0.1，上限降到 5.0，释放拟合能力)
        ]
        self.dim = len(self.bounds)
        self.cv_splits = cv_splits

        # 发现者和警戒者比例
        self.pd_rate = 0.2
        self.sd_rate = 0.2
        self.p_num = int(self.pop_size * self.pd_rate)
        self.s_num = int(self.pop_size * self.sd_rate)
        self.safe_threshold = 0.8

    def _sine_map_initialization(self):
        """使用 Sine 混沌映射初始化种群，代替纯随机，覆盖更均匀"""
        pop = np.zeros((self.pop_size, self.dim))
        x = np.random.rand(self.dim)
        for i in range(self.pop_size):
            x = 4 * x * (1 - x)  # Logistic/Sine 变体
            for j in range(self.dim):
                low, high = self.bounds[j]
                pop[i, j] = low + x[j] * (high - low)
        return pop

    def _fitness(self, pos, X, y):
        """适应度函数：使用时间序列交叉验证的 MAE (脱敏极端值)"""
        # 👑 核心修改 2：解析新增的 reg_lambda 参数
        n_estimators = int(round(pos[0]))
        max_depth = int(round(pos[1]))
        learning_rate = pos[2]
        subsample = pos[3]
        colsample_bytree = pos[4]
        reg_lambda = pos[5]

        # 边界控制 (确保算法在搜索时不会越界)
        n_estimators = np.clip(n_estimators, self.bounds[0][0], self.bounds[0][1])
        max_depth = np.clip(max_depth, self.bounds[1][0], self.bounds[1][1])
        learning_rate = np.clip(learning_rate, self.bounds[2][0], self.bounds[2][1])
        subsample = np.clip(subsample, self.bounds[3][0], self.bounds[3][1])
        colsample_bytree = np.clip(colsample_bytree, self.bounds[4][0], self.bounds[4][1])
        reg_lambda = np.clip(reg_lambda, self.bounds[5][0], self.bounds[5][1])

        model = XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,  # 传入正则化参数
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        # 👑 核心修改 3：改用 neg_mean_absolute_error，对过拟合尖峰进行降维打击
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        return abs(scores.mean())

    def optimize(self, X_train, y_train):
        print(f"🚀 开始 ISSA-XGBoost 超参数寻优 (种群:{self.pop_size}, 迭代:{self.max_iter})...")

        # 1. 初始化
        pop = self._sine_map_initialization()
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self._fitness(pop[i], X_train, y_train)

        # 记录全局最优
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        # 迭代寻优
        for t in range(self.max_iter):
            # 动态自适应权重，前期搜索范围大，后期精细搜索
            w = 0.9 - 0.5 * (t / self.max_iter)

            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]

            worst_pos = pop[-1].copy()
            best_pos_current = pop[0].copy()

            # 发现者位置更新 (Producers)
            R2 = np.random.rand()
            for i in range(self.p_num):
                if R2 < self.safe_threshold:
                    pop[i] = pop[i] * np.exp(-i / (np.random.rand() * self.max_iter + 1e-8))
                else:
                    pop[i] = pop[i] + np.random.randn(self.dim)

            # 加入者位置更新 (Scroungers)
            for i in range(self.p_num, self.pop_size):
                if i > self.pop_size / 2:
                    pop[i] = np.random.randn() * np.exp((worst_pos - pop[i]) / (i ** 2 + 1e-8))
                else:
                    # 保留上一轮修复的 A 矩阵维度问题
                    A = np.random.choice([-1, 1], size=(1, self.dim))
                    A_plus = A.T @ np.linalg.pinv(A @ A.T)
                    pop[i] = best_pos_current * w + np.abs(pop[i] - best_pos_current) * A_plus.flatten()

            # 警戒者位置更新 (Scouts)
            scout_idx = np.random.choice(range(self.pop_size), size=self.s_num, replace=False)
            for i in scout_idx:
                if fitness[i] > best_fit:
                    pop[i] = best_pos + np.random.randn(self.dim) * np.abs(pop[i] - best_pos)
                else:
                    pop[i] = pop[i] + (np.random.choice([-1, 1]) * np.abs(pop[i] - worst_pos)) / (
                                fitness[i] - fitness[-1] + 1e-8)

            # 边界处理与适应度评估
            for i in range(self.pop_size):
                for j in range(self.dim):
                    pop[i, j] = np.clip(pop[i, j], self.bounds[j][0], self.bounds[j][1])

                fit_i = self._fitness(pop[i], X_train, y_train)
                if fit_i < fitness[i]:
                    fitness[i] = fit_i
                if fit_i < best_fit:
                    best_fit = fit_i
                    best_pos = pop[i].copy()

            print(f"  [迭代 {t + 1}/{self.max_iter}] 当前最优 MAE: {best_fit:.4f}")

        print("\n✅ ISSA 寻优完成！")

        # 👑 核心修改 4：将第 6 个维度打包返回给主程序
        best_params = {
            'n_estimators': int(round(best_pos[0])),
            'max_depth': int(round(best_pos[1])),
            'learning_rate': best_pos[2],
            'subsample': best_pos[3],
            'colsample_bytree': best_pos[4],
            'reg_lambda': best_pos[5]
        }
        return best_params