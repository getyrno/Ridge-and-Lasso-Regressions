import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def load_data():
    print("Загрузка данных из OpenML...")
    ames = fetch_openml(name="house_prices", as_frame=True)
    df = ames.frame
    print("Данные загружены.")
    return df

def preprocess_data(df):
    print("Начало предобработки данных...")
    y = df['SalePrice'].values

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
    categorical_features = df.select_dtypes(include=['object']).columns

    print("Заполнение пропущенных значений...")
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

    print("Кодирование категориальных признаков...")
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    df_encoded = pd.DataFrame(encoded_categorical, columns=encoded_feature_names, index=df.index)

    print("Объединение признаков...")
    X = pd.concat([df[numeric_features], df_encoded], axis=1).values

    print("Предобработка данных завершена.")
    return X, y

def scale_features(X_train, X_test):
    print("Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Масштабирование завершено.")
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train, y_train):
    print("Обучение моделей Ridge и Lasso...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    print("Ridge Регрессия обучена.")

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    print("Lasso Регрессия обучена.")

    return ridge, lasso

def evaluate_models(models, X_test, y_test):
    ridge, lasso = models
    print("Оценка моделей...")

    y_ridge_pred = ridge.predict(X_test)
    y_lasso_pred = lasso.predict(X_test)

    rmse_ridge = mean_squared_error(y_test, y_ridge_pred, squared=False)
    mae_ridge = mean_absolute_error(y_test, y_ridge_pred)

    rmse_lasso = mean_squared_error(y_test, y_lasso_pred, squared=False)
    mae_lasso = mean_absolute_error(y_test, y_lasso_pred)

    print(f"Ridge Регрессия - RMSE: {rmse_ridge:.2f}, MAE: {mae_ridge:.2f}")
    print(f"Lasso Регрессия - RMSE: {rmse_lasso:.2f}, MAE: {mae_lasso:.2f}")

    return {
        'Ridge': {'RMSE': rmse_ridge, 'MAE': mae_ridge},
        'Lasso': {'RMSE': rmse_lasso, 'MAE': mae_lasso},
        'Predictions': {
            'Ridge': y_ridge_pred,
            'Lasso': y_lasso_pred
        }
    }

def visualize_results(models, scaler, poly=None):
    ridge, lasso = models
    print("Визуализация результатов...")

    feature_importances = np.abs(ridge.coef_)
    top_feature_index = np.argmax(feature_importances)
    top_feature_name = f"Feature {top_feature_index}"

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(ridge.coef_)), ridge.coef_, label='Ridge Коэффициенты')
    plt.scatter(range(len(lasso.coef_)), lasso.coef_, label='Lasso Коэффициенты')
    plt.xlabel('Индекс Признака')
    plt.ylabel('Коэффициент')
    plt.title('Сравнение Коэффициентов Ridge и Lasso Регрессий')
    plt.legend()
    plt.show()


def main():
    df = load_data()

    X, y = preprocess_data(df)

    print("Разделение данных на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    ridge, lasso = train_models(X_train_scaled, y_train)

    results = evaluate_models((ridge, lasso), X_test_scaled, y_test)

    visualize_results((ridge, lasso), scaler)

if __name__ == "__main__":
    main()
