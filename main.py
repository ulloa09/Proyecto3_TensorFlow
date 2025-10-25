import pandas as pd
import numpy as np
import mlflow
from P2_backtest import backtest
from P2_metrics import all_metrics
from graphs import (
    plot_portfolio_train,
    plot_portfolio_test,
    plot_portfolio_validation,
    plot_portfolio_combined
)
from functions import prepare_xy

def main():
    """
    Función principal para cargar un modelo entrenado,
    generar predicciones y ejecutar el backtest.
    """
    # --- 1. Configuración del Backtest ---
    # MODEL_NAME = "MLP_layers3_units150_relu_Weighted"
    MODEL_NAME = "CNN_conv4_filters16_dense50_tanh_Weighted"
    MODEL_STAGE = "latest"

    print(f"Iniciando backtest para el modelo: {MODEL_NAME} (stage: {MODEL_STAGE})")

    INITIAL_CASH = 1_000_000
    STOP_LOSS = 0.05
    TAKE_PROFIT = 0.10
    N_SHARES = 100

    # --- 2. Cargar Datos Escalados ---
    print("Cargando datos escalados...")
    try:
        train_data = pd.read_csv("data/train_scaled.csv")
        test_data = pd.read_csv("data/test_scaled.csv")
        val_data = pd.read_csv("data/val_scaled.csv")
        for df in [train_data, test_data, val_data]:
             if 'Date' in df.columns:
                  df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
         print("Error: Archivos escalados no encontrados. Ejecuta load_model.py primero.")
         return
    # Verificar si están vacíos después de cargar
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
    print(model.summary())


    # --- 4. Preparar Datos para Predicción ---
    # Usar prepare_xy para obtener la lista de features consistentemente
    # Pasar los DFs cargados
    try:
        X_train_np, X_val_np, X_test_np, _, _, _, feature_cols = prepare_xy(
            train_data, val_data, test_data
        )
    except ValueError as e:
        print(f"Error preparando X/y en main.py: {e}")
        return

    print(f"Preparando {len(feature_cols)} features para predicción...")
    
    # Los arrays numpy ya están listos desde prepare_xy
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

    class_train = np.argmax(pred_train, axis=1)
    class_test = np.argmax(pred_test, axis=1)
    class_val = np.argmax(pred_val, axis=1)

    signal_map = {0: -1, 1: 0, 2: 1}
    train_data["signal"] = np.vectorize(signal_map.get)(class_train)
    test_data["signal"] = np.vectorize(signal_map.get)(class_test)
    val_data["signal"] = np.vectorize(signal_map.get)(class_val)

    print("\n--- Distribución de Señales (Predicción) ---")
    for name, df in [("TRAIN", train_data), ("TEST", test_data), ("VALIDATION", val_data)]:
        print(f"\n{name} Set:")
        signal_counts = df["signal"].value_counts(normalize=True).sort_index()
        print(f"  Holds (0): \t{signal_counts.get(0, 0):.2%}")
        print(f"  Longs (1): \t{signal_counts.get(1, 0):.2%}")
        print(f"  Shorts (-1):\t{signal_counts.get(-1, 0):.2%}")


    # --- 6. Ejecutar Backtest ---
    print("\n--- Iniciando Backtest (Train) ---")
    port_train, cash_train, wr_train, buys_train, sells_train, holds_train, trades_train = backtest(
        train_data.copy(), INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, N_SHARES
    )
    print("\n--- Iniciando Backtest (Test) ---")
    port_test, cash_test, wr_test, buys_test, sells_test, holds_test, trades_test = backtest(
        test_data.copy(), INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, N_SHARES
    )
    print("\n--- Iniciando Backtest (Validation) ---")
    port_val, cash_val, wr_val, buys_val, sells_val, holds_val, trades_val = backtest(
        val_data.copy(), INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, N_SHARES
    )


    # --- 7. Mostrar Resultados ---
    print(f"\n--- RESULTADOS DEL BACKTEST (Modelo: {MODEL_NAME}) ---")
    results_data = {
        "TRAIN": (cash_train, wr_train, port_train, trades_train, buys_train, sells_train, holds_train),
        "TEST": (cash_test, wr_test, port_test, trades_test, buys_test, sells_test, holds_test),
        "VALIDATION": (cash_val, wr_val, port_val, trades_val, buys_val, sells_val, holds_val)
    }
    for name, (cash, win_rate, portfolio, n_trades, n_buys, n_sells, n_holds) in results_data.items():
        print(f"\nResultados: {name}")
        print(f"  Capital Inicial: ${INITIAL_CASH:,.2f}")
        print(f"  Capital Final:   ${cash:,.2f}")
        print(f"  Retorno Total:   {((cash / INITIAL_CASH) - 1):.2%}")
        print(f"  Total Trades:    {n_trades} (Buys: {n_buys}, Sells: {n_sells})")
        print(f"  Hold Signals:    {n_holds}")
        print(f"  Win Rate:        {win_rate:.2%}")
        if not portfolio.empty and len(portfolio) > 1 and portfolio.iloc[0] != portfolio.iloc[-1]:
            metrics_df = all_metrics(portfolio)
            print("  Métricas de Rendimiento:")
            print(metrics_df.to_string(float_format="%.4f"))
        else:
            print("  Métricas de Rendimiento: No calculadas (sin trades o sin cambio en valor)")


    # --- 8. Generar Gráficas ---
    print("\nGenerando gráficas del portafolio...")
    plot_portfolio_train(port_train, MODEL_NAME)
    plot_portfolio_test(port_test, MODEL_NAME)
    plot_portfolio_validation(port_val, MODEL_NAME)
    plot_portfolio_combined(port_train, port_test, port_val, MODEL_NAME)


if __name__ == "__main__":
    main()