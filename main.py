from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
from model_training import train_and_select_best_model
from analysis import run_data_drift_analysis, run_backtest_and_plots

# --- Constantes de configuración del Flujo ---

# Configuración de Carga y Features
DATA_CSV_PATH = 'data/wynn_daily_15y.csv'
FWD_RETURN_HORIZON = 5
lower = -0.002
upper = 0.002
SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}

# Configuración del Backtest
BACKTEST_PARAMS = {
    'stop_loss': 0.2,
    'take_profit': 0.2,
    'n_shares': 30
}

def main():
    """
    Ejecuta el flujo completo del proyecto de trading:
    1. Carga y preparación de datos.
    2. Escalado de features.
    3. Preparación de X/y.
    4. Entrenamiento y selección del mejor modelo (CNN vs MLP).
    5. Análisis de Data Drift.
    6. Ejecución de Backtest y generación de gráficas.
    """
    
    # --- 1. Carga y Preparación de Datos ---
    # (Guarda los DataFrames originales para el backtest final)
    train_df, test_df, validation_df = load_and_prepare_data(
        csv_path=DATA_CSV_PATH,
        horizon=FWD_RETURN_HORIZON,
        lower=lower,
        upper=upper,
        split_ratios=SPLIT_RATIOS
    )

    # --- 2. Escalado de Features ---
    # (Guarda los DataFrames escalados para el análisis de drift)
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )

    # --- 3. Preparación de X/y ---
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    # --- 4. Entrenamiento y Selección del Modelo ---
    # Esta función encapsula los bucles de prueba de CNN y MLP,
    # el re-shape de CNN, y la lógica de selección final.
    best_model, model_name, X_train_final, X_test_final, X_val_final = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # --- 5. Análisis de Data Drift ---
    run_data_drift_analysis(
        train_scaled, test_scaled, val_scaled, feature_cols
    )

    # --- 6. Backtest y Gráficas ---
    run_backtest_and_plots(
        best_model, model_name,
        X_train_final, X_test_final, X_val_final,
        train_df, test_df, validation_df, # Usa los DFs *originales*
        BACKTEST_PARAMS
    )

if __name__ == "__main__":
    main()