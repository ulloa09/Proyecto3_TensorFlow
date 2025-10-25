import pandas as pd
import numpy as np
import mlflow
from backtest import backtest 
from metrics import * 
from graphs import ( 
    plot_portfolio_train,
    plot_portfolio_test,
    plot_portfolio_validation,
    plot_portfolio_combined
)
from functions import prepare_xy 
from drift_analisis import find_rolling_drift_breakpoint

def main():
    """
    Función principal para cargar un modelo entrenado,
    generar predicciones y ejecutar el backtest.
    """
    # --- 1. Configuración del Backtest ---
    MODEL_NAME = "CNN_conv4_filters16_dense50_relu_Weighted"
    # MODEL_NAME = "MLP_layers2_units200_relu_Weighted"
    MODEL_STAGE = "latest"

    print(f"Iniciando backtest para el modelo: {MODEL_NAME} (stage: {MODEL_STAGE})")

    # Parámetros del backtest
    STOP_LOSS = 0.05
    TAKE_PROFIT = 0.15
    N_SHARES = 100
    INITIAL_CASH = 1_000_000 

    # --- 2. Cargar Datos Escalados ---
    print("Cargando datos escalados...")
    try:
        train_data = pd.read_csv("data/train_scaled.csv")
        test_data = pd.read_csv("data/test_scaled.csv")
        val_data = pd.read_csv("data/val_scaled.csv")
        
        # Asegurar que 'Datetime' (usado por el backtest y drift) exista
        for df in [train_data, test_data, val_data]:
             if 'Date' in df.columns:
                  # Convertir a Datetime y poner como índice para el análisis
                  df['Datetime'] = pd.to_datetime(df['Date'])
                  df = df.set_index('Datetime', drop=False)
             else:
                 print("Advertencia: No se encontró 'Date', el drift rodante puede fallar.")
    except FileNotFoundError:
         print("Error: Archivos escalados no encontrados. Ejecuta train_models.py primero.")
         return
    
    if train_data.empty or test_data.empty or val_data.empty:
         print("Error: Uno o más archivos CSV cargados están vacíos.")
         return

    # --- 3. Cargar Modelo desde MLflow ---
    print(f"Cargando modelo '{MODEL_NAME}' desde MLflow...")
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.tensorflow.load_model(model_uri)
    except Exception as e:
        print(f"Error al cargar el modelo de MLflow: {e}")
        return
    model.summary()


    # --- 4. Preparar Datos para Predicción ---
    try:
        # Usamos prepare_xy para obtener la lista de features consistentemente
        X_train_np, X_val_np, X_test_np, _, _, _, feature_cols = prepare_xy(
            train_data, val_data, test_data
        )
    except ValueError as e:
         print(f"Error preparando X/y en run_backtest.py: {e}")
         return

    print(f"Preparando {len(feature_cols)} features para predicción...")
    
    x_train_pred = X_train_np
    x_test_pred = X_test_np
    x_val_pred = X_val_np

    is_cnn = "CNN" in MODEL_NAME.upper()
    if is_cnn:
        print("Modelo CNN detectado, ajustando forma de entrada a 3D.")
        x_train_pred = np.expand_dims(x_train_pred, axis=-1)
        x_test_pred = np.expand_dims(x_test_pred, axis=-1)
        x_val_pred = np.expand_dims(x_val_pred, axis=-1)

    # --- 5. Generar Predicciones ---
    print("Generando predicciones...")
    pred_train = model.predict(x_train_pred)
    pred_test = model.predict(x_test_pred)
    pred_val = model.predict(x_val_pred)

    # Obtener clases (0, 1, 2)
    class_train = np.argmax(pred_train, axis=1)
    class_test = np.argmax(pred_test, axis=1)
    class_val = np.argmax(pred_val, axis=1)

    # --- ASIGNACIÓN DE SEÑAL ---
    # El backtest lee la columna "target" (0=Sell, 1=Hold, 2=Buy)
    train_data["target"] = class_train
    test_data["target"] = class_test
    val_data["target"] = class_val

    print("\n--- Distribución de Señales (Predicción) ---")
    for name, df in [("TRAIN", train_data), ("TEST", test_data), ("VALIDATION", val_data)]:
        print(f"\n{name} Set:")
        signal_counts = df["target"].value_counts(normalize=True).sort_index()
        print(f"  Holds (1): \t{signal_counts.get(1, 0):.2%}")
        print(f"  Longs (2): \t{signal_counts.get(2, 0):.2%}")
        print(f"  Shorts (0):\t{signal_counts.get(0, 0):.2%}")


    # --- 6. EJECUTAR ANÁLISIS DE DRIFT RODANTE (NUEVO) ---
    print("\n--- Iniciando Análisis de Punto de Quiebre (Drift Rodante) ---")
    
    # Asegurar que los DFs tengan índice Datetime para el análisis
    train_data_drift = train_data.set_index('Datetime')
    test_data_drift = test_data.set_index('Datetime')
    val_data_drift = val_data.set_index('Datetime')

    test_break_date = find_rolling_drift_breakpoint(
        train_data_drift, test_data_drift, feature_cols,
        window_size=90, drift_threshold=0.20
    )
    
    val_break_date = find_rolling_drift_breakpoint(
        train_data_drift, val_data_drift, feature_cols,
        window_size=90, drift_threshold=0.20
    )

    # --- 7. Ejecutar Backtest ---
    print("\n--- Iniciando Backtest (Train) ---")
    cash_train, port_train, wr_train, buys_train, sells_train, holds_train, trades_train, metrics_train = backtest(
        train_data.copy(), STOP_LOSS, TAKE_PROFIT, N_SHARES
    )
    print("\n--- Iniciando Backtest (Test) ---")
    cash_test, port_test, wr_test, buys_test, sells_test, holds_test, trades_test, metrics_test = backtest(
        test_data.copy(), STOP_LOSS, TAKE_PROFIT, N_SHARES
    )
    print("\n--- Iniciando Backtest (Validation) ---")
    cash_val, port_val, wr_val, buys_val, sells_val, holds_val, trades_val, metrics_val = backtest(
        val_data.copy(), STOP_LOSS, TAKE_PROFIT, N_SHARES
    )

    # --- 8. Mostrar Resultados ---
    print(f"\n--- RESULTADOS DEL BACKTEST (Modelo: {MODEL_NAME}) ---")
    results_data = {
        "TRAIN": (cash_train, wr_train, metrics_train, trades_train, buys_train, sells_train, holds_train),
        "TEST": (cash_test, wr_test, metrics_test, trades_test, buys_test, sells_test, holds_test),
        "VALIDATION": (cash_val, wr_val, metrics_val, trades_val, buys_val, sells_val, holds_val)
    }
    
    for name, (cash, win_rate, metrics_df, n_trades, n_buys, n_sells, n_holds) in results_data.items():
        print(f"\nResultados: {name}")
        print(f"  Capital Inicial: ${INITIAL_CASH:,.2f}")
        print(f"  Capital Final:   ${cash:,.2f}")
        print(f"  Retorno Total:   {((cash / INITIAL_CASH) - 1):.2%}")
        print(f"  Total Trades:    {n_trades} (Buys: {n_buys}, Sells: {n_sells})")
        print(f"  Hold Signals:    {n_holds}")
        print(f"  Win Rate (calc): {win_rate:.2%}")
        
        if metrics_df is not None:
            print("  Métricas de Rendimiento (del backtest):")
            print(metrics_df.to_string(float_format="%.4f"))
        else:
            print("  Métricas de Rendimiento: No calculadas (sin trades o error).")


    # --- 9. Generar Gráficas ---
    print("\nGenerando gráficas del portafolio...")
    plot_portfolio_train(port_train, MODEL_NAME)
    
    # Pasar las fechas de quiebre a las gráficas
    plot_portfolio_test(port_test, MODEL_NAME, breakpoint_date=test_break_date)
    plot_portfolio_validation(port_val, MODEL_NAME, breakpoint_date=val_break_date)
    
    plot_portfolio_combined(port_train, port_test, port_val, MODEL_NAME)


if __name__ == "__main__":
    main()