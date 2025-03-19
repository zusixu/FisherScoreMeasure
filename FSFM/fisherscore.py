import numpy as np


def fisher_score(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    overall_mean = np.mean(X, axis=0)
    fisher_scores = np.zeros(n_features)

    for i in range(n_features):
        between_class = 0
        within_class = 0

        for cls in classes:
            X_cls = X[y == cls, i]
            n_cls = len(X_cls)  # 获取类别样本数
            mean_cls = np.mean(X_cls)
            var_cls = np.var(X_cls, ddof=1)

            # 加权类间方差
            between_class += n_cls * (mean_cls - overall_mean[i]) ** 2
            # 累计类内方差
            within_class += var_cls * (n_cls - 1)  # 无偏估计修正

        # 添加分母保护机制
        denominator = within_class / (len(X) - len(classes))  # 标准化处理
        fisher_scores[i] = between_class / denominator if denominator != 0 else 0

    # 使用 Sigmoid 函数进行归一化
    fisher_scores = 1 / (1 + np.exp(-fisher_scores))

    return fisher_scores


def calculate_fisher_score(X, y):
    """
    计算各个特征的 Fisher Score
    :param X: 特征矩阵, shape (n_samples, n_features)
    :param y: 标签向量, shape (n_samples,)
    :return: Fisher Score 向量, shape (n_features,)
    """
    # 使用 sklearn 的 fisher_score 函数计算 Fisher Score
    scores = fisher_score(X, y)
    return scores

def get_data_and_calculate_fisher_score(data, labels):
    """
    获取数据并计算各个特征的 Fisher Score
    :param data: 特征矩阵, shape (n_samples, n_features)
    :param labels: 标签向量, shape (n_samples,)
    :return: Fisher Score 向量, shape (n_features,)
    """
    # 确保输入数据为 numpy 数组
    X = np.array(data)
    y = np.array(labels)
    
    # 计算 Fisher Score
    fisher_scores = calculate_fisher_score(X, y)
    
    return fisher_scores

def calculate_combined_fisher_score(fs_A, fs_B, lambda_value):
    """
    计算特征组合的 Fisher Score
    :param fs_A: 特征 A 的 Fisher Score
    :param fs_B: 特征 B 的 Fisher Score
    :param lambda_value: 参数 lambda
    :return: 特征组合的 Fisher Score
    """
    return fs_A + fs_B + lambda_value * fs_A * fs_B


def solve_lambda(fs_values, lambda_init=-0.1, max_iter=100, tol=1e-6):
    """
    使用牛顿迭代法求解方程 1 + λ = Π(1 + λ*fs_i)

    参数:
    fs_values -- Fisher Score值列表/数组
    lambda_init -- 初始猜测值 (默认0.0)
    max_iter -- 最大迭代次数 (默认100)
    tol -- 收敛容差 (默认1e-6)

    返回:
    float -- 求解得到的λ值
    """
    lambda_est = lambda_init
    for _ in range(max_iter):
        # 计算乘积项和导数项
        product = 1.0
        derivative_sum = 0.0

        for fs in fs_values:
            term = 1 + lambda_est * fs
            product *= term
            derivative_sum += fs / term if term != 0 else 0  # 防止除以零

        # 计算函数值和导数值
        f = product - (1 + lambda_est)
        df = product * derivative_sum - 1

        # 检查收敛
        if abs(f) < tol:
            break

        # 牛顿迭代更新
        if abs(df) < 1e-12:  # 防止导数值过小导致数值不稳定
            break
        lambda_est -= f / df

    return lambda_est
def test_combined_fisher_score():
    """
    测试特征组合的 Fisher Score 计算
    """
    # 模拟数据
    X = np.array([
        [1, 2, 3],
        [4, 1, 6],
        [1, 4, 2],
        [2, 1, 4],
        [1, 6, 1]
    ])
    y = np.array([0, 1, 0, 0, 1])

    # 计算各个特征的 Fisher Score
    fisher_scores = calculate_fisher_score(X, y)
    
    # 求解参数 lambda
    lambda_value = solve_lambda(fisher_scores)
    
    # 计算特征组合的 Fisher Score
    fs_A = fisher_scores[0]
    fs_B = fisher_scores[1]
    combined_fs = calculate_combined_fisher_score(fs_A, fs_B, lambda_value)
    
    # 输出结果
    print("Fisher Scores for individual features:", fisher_scores)
    print("Lambda value:", lambda_value)
    print("Combined Fisher Score for feature 0 and 1:", combined_fs)

# 运行测试用例
test_combined_fisher_score()