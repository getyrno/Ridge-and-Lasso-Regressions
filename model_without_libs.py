# model_without_lib.py

import pandas as pd
import csv
import math
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def load_data():
    print("Загрузка данных из OpenML...")
    ames = fetch_openml(name="house_prices", as_frame=True)
    df = ames.frame
    print("Данные загружены.")
    return df

def preprocess_data(df):
    print("Начало предобработки данных...")
    
    y = df['SalePrice'].tolist()
    
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.remove('SalePrice')  # Удаляем целевую переменную
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    print("Заполнение пропущенных значений...")
    for feature in numeric_features:
        mean_val = df[feature].mean()
        df[feature] = df[feature].fillna(mean_val)
    
    for feature in categorical_features:
        mode_val = df[feature].mode()
        if not mode_val.empty:
            df[feature] = df[feature].fillna(mode_val[0])
        else:
            df[feature] = df[feature].fillna('Unknown')
    
    
    print("Кодирование категориальных признаков...")
    encoded_dfs = []
    encoded_features = []
    
    for feature in categorical_features:
        unique_values = sorted(df[feature].unique())
        if unique_values:
            unique_values = unique_values[1:]
        for val in unique_values:
            encoded_feature = df[feature].apply(lambda x: 1.0 if x == val else 0.0)
            encoded_dfs.append(encoded_feature.rename(f"{feature}_{val}"))
            encoded_features.append(f"{feature}_{val}")
    
    if encoded_dfs:
        encoded_df = pd.concat(encoded_dfs, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
    
    df.drop(columns=categorical_features, inplace=True)
    
    print("Объединение признаков...")
    feature_names = numeric_features + encoded_features
    X = df[feature_names].values.tolist()
    
    print("Предобработка данных завершена.")
    return X, y, feature_names

def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    print("Разделение данных на обучающую и тестовую выборки...")
    combined = list(zip(X, y))
    random.seed(random_state)
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def standard_scaler(X_train, X_test):
    print("Масштабирование признаков...")
    n_features = len(X_train[0])
    means = []
    stds = []
    for i in range(n_features):
        feature = [x[i] for x in X_train]
        mean = sum(feature) / len(feature)
        variance = sum((x - mean) ** 2 for x in feature) / len(feature)
        std = math.sqrt(variance) if variance > 0 else 1.0
        means.append(mean)
        stds.append(std)
    
    X_train_scaled = []
    for row in X_train:
        scaled_row = [(row[i] - means[i]) / stds[i] for i in range(n_features)]
        X_train_scaled.append(scaled_row)
    
    X_test_scaled = []
    for row in X_test:
        scaled_row = [(row[i] - means[i]) / stds[i] for i in range(n_features)]
        X_test_scaled.append(scaled_row)
    
    print("Масштабирование завершено.")
    return X_train_scaled, X_test_scaled, means, stds

def mat_vec_mult(matrix, vector):
    result = []
    for row in matrix:
        res = sum(m * v for m, v in zip(row, vector))
        result.append(res)
    return result

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def mat_mult(a, b):
    b_t = transpose(b)
    return [[sum(x*y for x, y in zip(row, col)) for col in b_t] for row in a]

def inverse_matrix(matrix):
    n = len(matrix)
    AM = [row[:] for row in matrix]
    IM = [[float(i == j) for i in range(n)] for j in range(n)]
    
    for fd in range(n):
        if AM[fd][fd] == 0:
            for i in range(fd+1, n):
                if AM[i][fd] != 0:
                    AM[fd], AM[i] = AM[i], AM[fd]
                    IM[fd], IM[i] = IM[i], IM[fd]
                    break
            else:
                raise ValueError("Матрица необратима")
        
        fd_scalar = AM[fd][fd]
        AM[fd] = [x / fd_scalar for x in AM[fd]]
        IM[fd] = [x / fd_scalar for x in IM[fd]]
        
        for i in range(n):
            if i != fd:
                scalar = AM[i][fd]
                AM[i] = [a - scalar * b for a, b in zip(AM[i], AM[fd])]
                IM[i] = [a - scalar * b for a, b in zip(IM[i], IM[fd])]
    
    return IM

def ridge_regression(X, y, alpha=1.0):
    print("Обучение Ridge Регрессии...")
    X_t = transpose(X)
    X_t_X = mat_mult(X_t, X)
    n = len(X_t_X)
    for i in range(n):
        X_t_X[i][i] += alpha
    X_t_y = mat_vec_mult(X_t, y)
    try:
        X_t_X_inv = inverse_matrix(X_t_X)
    except ValueError:
        print("Матрица X^T X необратима.")
        return None
    weights = mat_vec_mult(X_t_X_inv, X_t_y)
    print("Ridge Регрессия обучена.")
    return weights

def lasso_regression(X, y, alpha=0.1, learning_rate=0.001, epochs=100600):
    print("Обучение Lasso Регрессии...")
    n_samples = len(X)
    n_features = len(X[0])
    weights = [0.0 for _ in range(n_features)]
    
    for epoch in range(epochs):
        predictions = mat_vec_mult(X, weights)
        errors = [pred - actual for pred, actual in zip(predictions, y)]
        
        for i in range(n_features):
            gradient = sum(errors[j] * X[j][i] for j in range(n_samples)) / n_samples
            if weights[i] > 0:
                gradient += alpha
            elif weights[i] < 0:
                gradient -= alpha
            weights[i] -= learning_rate * gradient
        
        if epoch % 100 == 0:
            loss = sum(e**2 for e in errors) / n_samples + alpha * sum(abs(w) for w in weights)
            print(f"epoch - {epoch}, loss - {loss}")
    print("Lasso Регрессия обучена.")
    return weights

def mean_squared_error_custom(y_true, y_pred):
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    return mse

def mean_absolute_error_custom(y_true, y_pred):
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
    return mae

def evaluate_models(ridge_weights, lasso_weights, X_test, y_test, epochs):
    print("Оценка моделей...")
    
    def predict(weights, X):
        return [sum(w * x for w, x in zip(weights, features)) for features in X]
    
    y_ridge_pred = predict(ridge_weights, X_test)
    y_lasso_pred = predict(lasso_weights, X_test)
    
    rmse_ridge = math.sqrt(mean_squared_error_custom(y_test, y_ridge_pred))
    mae_ridge = mean_absolute_error_custom(y_test, y_ridge_pred)
    
    rmse_lasso = math.sqrt(mean_squared_error_custom(y_test, y_lasso_pred))
    mae_lasso = mean_absolute_error_custom(y_test, y_lasso_pred)
    
    print(f"Ridge Регрессия - RMSE: {rmse_ridge:.2f}, MAE: {mae_ridge:.2f}")
    print(f"Lasso Регрессия - RMSE: {rmse_lasso:.2f}, MAE: {mae_lasso:.2f}")
    print(f"Количество эпох - {epochs}")
    
    return {
        'Ridge': {'RMSE': rmse_ridge, 'MAE': mae_ridge},
        'Lasso': {'RMSE': rmse_lasso, 'MAE': mae_lasso},
        'Predictions': {
            'Ridge': y_ridge_pred,
            'Lasso': y_lasso_pred
        }
    }

def visualize_results(ridge_weights, lasso_weights, feature_names):
    print("Визуализация результатов...")
    
    plt.figure(figsize=(12, 6))
    indices = range(len(ridge_weights))
    plt.scatter(indices, ridge_weights, label='Ridge Коэффициенты', alpha=0.7)
    plt.scatter(indices, lasso_weights, label='Lasso Коэффициенты', alpha=0.7)
    plt.xlabel('Индекс Признака')
    plt.ylabel('Коэффициент')
    plt.title('Сравнение Коэффициентов Ridge и Lasso Регрессий')
    plt.legend()
    plt.show()
    
def main():
    df = load_data()
    
    X, y, feature_names = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled, X_test_scaled, means, stds = standard_scaler(X_train, X_test)
    
    for row in X_train_scaled:
        row.insert(0, 1.0)
    for row in X_test_scaled:
        row.insert(0, 1.0)
    feature_names = ['Intercept'] + feature_names
    
    ridge_weights = ridge_regression(X_train_scaled, y_train, alpha=1.0)
    lasso_weights = lasso_regression(X_train_scaled, y_train, alpha=0.1, learning_rate=0.001, epochs=10000)
    
    if ridge_weights is None or lasso_weights is None:
        print("Ошибка при обучении моделей.")
        return
    
    results = evaluate_models(ridge_weights, lasso_weights, X_test_scaled, y_test, epochs=10000)
    
    # visualize_results(ridge_weights, lasso_weights, feature_names)
    
if __name__ == "__main__":
    main()
