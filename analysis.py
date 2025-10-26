import numpy as np
import pandas as pd
from data_drift import get_feature_pvalues, get_feature_shift_status
from backtest import backtest
from graphs import (
    plot_portfolio_train,
    plot_portfolio_test,
    plot_portfolio_validation,
    plot_portfolio_combined
)

def run_data_drift_analysis(train_scaled, test_scaled, val_scaled, feature_cols):
    """
    Ejecuta el análisis de estabilidad (Data Drift) comparando
    Test vs Train y Validation vs Train.
    """
    # --- ANÁLISIS DE ESTABILIDAD (DATA DRIFT) ---
    print("\n--- Iniciando Análisis de Estabilidad de Features ---")

    # Usar las features escaladas (sin etiquetas) para la comparación
    baseline_data_features = train_scaled[feature_cols]
    test_data_features = test_scaled[feature_cols]
    val_data_features = val_scaled[feature_cols]

    # 1. Comparar Test vs Train
    print("\nComparando Test vs. Train...")
    p_values_test = get_feature_pvalues(baseline_data_features, test_data_features)
    shift_status_test = get_feature_shift_status(baseline_data_features, test_data_features, threshold=0.05)

    drift_df_test = pd.DataFrame(list(p_values_test.items()), columns=['Feature', 'p_value'])
    drift_df_test['Drift_Detected'] = drift_df_test['Feature'].map(shift_status_test).fillna(False)
    drift_df_test = drift_df_test.sort_values(by='p_value', ascending=True)

    print(f"Features con shift detectado (Test): {drift_df_test['Drift_Detected'].sum()}")
    if drift_df_test['Drift_Detected'].sum() > 0:
        print(drift_df_test[drift_df_test['Drift_Detected'] == True].to_string())

    # 2. Comparar Validation vs Train
    print("\nComparando Validation vs. Train...")
    p_values_val = get_feature_pvalues(baseline_data_features, val_data_features)
    shift_status_val = get_feature_shift_status(baseline_data_features, val_data_features, threshold=0.05)

    drift_df_val = pd.DataFrame(list(p_values_val.items()), columns=['Feature', 'p_value'])
    drift_df_val['Drift_Detected'] = drift_df_val['Feature'].map(shift_status_val).fillna(False)
    drift_df_val = drift_df_val.sort_values(by='p_value', ascending=True)

    print(f"Features con shift detectado (Validation): {drift_df_val['Drift_Detected'].sum()}")
    if drift_df_val['Drift_Detected'].sum() > 0:
        print(drift_df_val[drift_df_val['Drift_Detected'] == True].to_string())
    print("--- Análisis de Estabilidad Completado ---")


def run_backtest_and_plots(
    best_model, model_name,
    X_train_final, X_test_final, X_val_final,
    train_df, test_df, validation_df, # DataFrames *originales*
    backtest_params: dict
):
    """
    Ejecuta el backtest final usando el modelo ganador y genera las gráficas.
    """
    # --- EJECUCIÓN DE BACKTEST Y GRÁFICAS ---
    print(f"\n--- Iniciando Backtest (Modelo Ganador: {model_name}) ---")

    # Parámetros fijos del backtest
    STOP_LOSS = backtest_params['stop_loss']
    TAKE_PROFIT = backtest_params['take_profit']
    N_SHARES = backtest_params['n_shares']

    # Generar predicciones para los 3 sets
    y_pred_train = np.argmax(best_model.predict(X_train_final), axis=1)
    y_pred_test = np.argmax(best_model.predict(X_test_final), axis=1)
    y_pred_val = np.argmax(best_model.predict(X_val_final), axis=1)

    # Asignar predicciones (usar copias para evitar warnings)
    # Usamos los DataFrames originales (sin escalar) para el backtest
    train_df_bt = train_df.copy()
    test_df_bt = test_df.copy()
    val_df_bt = validation_df.copy()

    train_df_bt["target"] = y_pred_train
    test_df_bt["target"] = y_pred_test
    val_df_bt["target"] = y_pred_val

    # Ejecutar backtests y capturar la SERIE de portafolio (port_series_...)
    print("\nEjecutando Backtest (Train)...")
    cash_train, port_series_train, buy_train, sell_train, hold_train, total_ops_train = backtest(
        train_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nEjecutando Backtest (Test)...")
    cash_test, port_series_test, buy_test, sell_test, hold_test, total_ops_test = backtest(
        test_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nEjecutando Backtest (Validation)...")
    cash_val, port_series_val, buy_val, sell_val, hold_val, total_ops_val = backtest(
        val_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    # --- GENERACIÓN DE GRÁFICAS ---
    print("\n--- Generando Gráficas del Portafolio ---")

    # 1. Gráfica de Entrenamiento
    plot_portfolio_train(port_series_train, model_name)

    # 2. Gráfica de Prueba (Test)
    plot_portfolio_test(port_series_test, model_name) 

    # 3. Gráfica de Validación
    plot_portfolio_validation(port_series_val, model_name)

    # 4. Gráfica Combinada
    plot_portfolio_combined(port_series_train, port_series_test, port_series_val, model_name)

    print("--- Ejecución Finalizada ---")