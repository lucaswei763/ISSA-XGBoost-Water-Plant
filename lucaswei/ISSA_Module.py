import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import qmc  # 用于 Sobol 序列

class ISSA_XGBoost_Optimizer:
    # 这里的 test_x, test_y 实际上在外部调用时会传入验证集 (Validation Set)
    def __init__(self, train_x, train_y, val_x, val_y, pop_size=32, max_iter=200, patience=50):
        self.train_x, self.train_y = train_x, train_y
        self.val_x, self.val_y = val_x, val_y

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.patience = patience
        self.no_improve_count = 0

        # --- 核心修改：搜索维度升级为 4 ---
        self.dim = 6
        # 搜索边界：[n_estimators, learning_rate, max_depth, gamma, min_child_weight, reg_lambda]
        self.lb = np.array([10, 0.001, 2, 0.01, 1, 0.1])  # 注意 max_depth 下界放宽到 2
        self.ub = np.array([1000, 0.3, 10, 0.5, 10, 5.0])

        # 策略参数
        self.P_percent = 0.45
        self.V_percent = 0.2
        self.ST = 0.4         # 降低安全值，增加全局搜索概率

        self.fitness_history = []
        self.X = self.init_sobol()
        self.fitness = np.zeros(pop_size)
        self.best_x = None
        self.best_fit = float('inf')

    def init_sobol(self):
        """策略1: Sobol序列种群初始化"""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n=self.pop_size)
        return qmc.scale(sample, self.lb, self.ub)

    def get_fitness(self, params):
        """适应度函数：将寻优参数注入 XGBoost"""
        model = xgb.XGBRegressor(
            n_estimators=int(params[0]),
            learning_rate=params[1],
            max_depth=int(params[2]),
            gamma=params[3],
            min_child_weight=int(params[4]),  # 新增武器 1
            reg_lambda=params[5],  # 新增武器 2
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(self.train_x, self.train_y)
        # 用传入的验证集来计算 RMSE 指导进化
        preds = model.predict(self.val_x)
        return np.sqrt(mean_squared_error(self.val_y, preds))

    def optimize(self):
        # 初始化适应度
        for i in range(self.pop_size):
            self.fitness[i] = self.get_fitness(self.X[i])
            if self.fitness[i] < self.best_fit:
                self.best_fit = self.fitness[i]
                self.best_x = self.X[i].copy()

        for t in range(self.max_iter):
            old_best_fit = self.best_fit

            # 排序
            idx = np.argsort(self.fitness)
            self.X = self.X[idx]
            self.fitness = self.fitness[idx]

            best_p = self.X[0]
            worst_p = self.X[-1]

            # 发现者位置更新
            num_p = int(self.pop_size * self.P_percent)
            for i in range(num_p):
                r2 = np.random.rand()
                if r2 < self.ST:
                    self.X[i] = self.X[i] * np.exp(-i / (np.random.rand() * self.max_iter))
                else:
                    self.X[i] = self.X[i] + np.random.randn() * np.ones(self.dim)

            # 追随者位置更新
            for i in range(num_p, self.pop_size):
                if i > self.pop_size / 2:
                    self.X[i] = np.random.randn() * np.exp((worst_p - self.X[i]) / i ** 2)
                else:
                    k = np.random.randint(0, num_p)
                    x_k = self.X[k]
                    A = np.random.choice([1, -1], size=self.dim)
                    self.X[i] = self.best_x + np.random.rand() * np.abs(self.X[i] - self.best_x) * A + \
                                np.random.rand() * (x_k - self.best_x)

            # 警戒者位置更新
            num_v = int(self.pop_size * self.V_percent)
            v_indices = np.random.choice(range(self.pop_size), num_v, replace=False)
            for i in v_indices:
                if self.fitness[i] > self.best_fit:
                    self.X[i] = self.best_x + np.random.randn() * np.abs(self.X[i] - self.best_x)
                else:
                    self.X[i] = self.X[i] + np.random.uniform(-1, 1) * \
                                (np.abs(self.X[i] - worst_p) / (self.fitness[i] - self.fitness[-1] + 1e-8))

            # 策略3: 自适应柯西-高斯变异
            lambda1 = 1 - (t ** 2 / self.max_iter ** 2)
            lambda2 = t ** 2 / self.max_iter ** 2
            mutation_x = self.best_x * (1 + lambda1 * np.random.standard_cauchy() + \
                                        lambda2 * np.random.standard_normal())

            self.X = np.clip(self.X, self.lb, self.ub)
            mutation_x = np.clip(mutation_x, self.lb, self.ub)

            fit_mut = self.get_fitness(mutation_x)
            if fit_mut < self.best_fit:
                self.best_fit = fit_mut
                self.best_x = mutation_x

            for i in range(self.pop_size):
                self.fitness[i] = self.get_fitness(self.X[i])

            self.fitness_history.append(self.best_fit)

            if self.best_fit < old_best_fit:
                self.no_improve_count = 0
                print(f"==========发现新的更优解============, {self.best_fit:.4f}")
            else:
                self.no_improve_count += 1

            print(f"Iteration {t + 1}/{self.max_iter}, Best RMSE: {self.best_fit:.4f}, Patience: {self.no_improve_count}/{self.patience}")

            if self.no_improve_count >= self.patience:
                print(f"📢 检测到模型已收敛，触发早停机制。")
                break

        return self.best_x, self.best_fit