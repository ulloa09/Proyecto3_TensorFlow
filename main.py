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


def main():
    """
    Función principal para cargar un modelo entrenado,
    generar predicciones y ejecutar el backtest.
    """
    # --- 1. Configuración del Backtest ---
    # Cambia este nombre por el modelo
    MODEL_NAME = "MLP_capas2_unidades200_relu"
    MODEL_STAGE = "latest"

    print(f"Iniciando backtest para el modelo: {MODEL_NAME} (stage: {MODEL_STAGE})")

    INITIAL_CASH = 1_000_000
    STOP_LOSS = 0.05
    TAKE_PROFIT = 0.15
    N_SHARES = 100

    # --- 2. Cargar Datos Escalados ---
    print("Cargando datos escalados...")
    train_data = pd.read_csv("data/train_scaled.csv")
    test_data = pd.read_csv("data/test_scaled.csv")
    val_data = pd.read_csv("data/val_scaled.csv")

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
    columnas_excluir = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'fwd_ret', 'target']
    sample_cols = train_data.columns
    columnas_features = [col for col in sample_cols if col not in columnas_excluir]

    print(f"Preparando {len(columnas_features)} features para predicción...")
    x_train = train_data[columnas_features]
    x_test = test_data[columnas_features]
    x_val = val_data[columnas_features]

    is_cnn = "CNN" in MODEL_NAME.upper()
    if is_cnn:
        print("Modelo CNN detectado, ajustando forma de entrada a 3D.")
        x_train = np.expand_dims(x_train.values, axis=-1)
        x_test = np.expand_dims(x_test.values, axis=-1)
        x_val = np.expand_dims(x_val.values, axis=-1)

    # --- 5. Generar Predicciones ---
    print("Generando predicciones...")
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    pred_val = model.predict(x_val)

    class_train = np.argmax(pred_train, axis=1)
    class_test = np.argmax(pred_test, axis=1)
    class_val = np.argmax(pred_val, axis=1)

    # Mapear clases (0, 1, 2) de vuelta a señales (-1, 0, 1)
    train_data["signal"] = class_train - 1
    test_data["signal"] = class_test - 1
    val_data["signal"] = class_val - 1

    print("\n--- Distribución de Señales (Predicción) ---")

    for name, df in [("TRAIN", train_data), ("TEST", test_data), ("VALIDATION", val_data)]:
        print(f"\n{name} Set:")
        signal_counts = df["signal"].value_counts()
        total_signals = len(df)

        # Obtener conteos
        longs = signal_counts.get(1, 0)
        shorts = signal_counts.get(-1, 0)
        holds = signal_counts.get(0, 0)

        print(f"  Holds (0): \t{holds:d} \t({(holds / total_signals):.2%})")
        print(f"  Longs (1): \t{longs:d} \t({(longs / total_signals):.2%})")
        print(f"  Shorts (-1):\t{shorts:d} \t({(shorts / total_signals):.2%})")

    # --- 6. Ejecutar Backtest ---
    print("\n--- Iniciando Backtest (Train) ---")
    port_train, cash_train, wr_train, trades_train = backtest(
        train_data.copy(), INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, N_SHARES
    )

    print("\n--- Iniciando Backtest (Test) ---")
    port_test, cash_test, wr_test, trades_test = backtest(
        test_data.copy(), INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, N_SHARES
    )

    print("\n--- Iniciando Backtest (Validation) ---")
    port_val, cash_val, wr_val, trades_val = backtest(
        val_data.copy(), INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, N_SHARES
    )

    # --- 7. Mostrar Resultados ---
    print(f"\n--- RESULTADOS DEL BACKTEST (Modelo: {MODEL_NAME}) ---")

    results_data = {
        "TRAIN": (cash_train, wr_train, port_train, trades_train),
        "TEST": (cash_test, wr_test, port_test, trades_test),
        "VALIDATION": (cash_val, wr_val, port_val, trades_val)
    }

    for name, (cash, win_rate, portfolio, n_trades) in results_data.items():
        print(f"\nResultados: {name}")
        print(f"  Capital Inicial: ${INITIAL_CASH:,.2f}")
        print(f"  Capital Final:   ${cash:,.2f}")
        print(f"  Retorno Total:   {((cash / INITIAL_CASH) - 1):.2%}")
        print(f"  Total Trades:    {n_trades}")
        print(f"  Win Rate:        {win_rate:.2%}")

        metrics_df = all_metrics(portfolio)
        print("  Métricas de Rendimiento:")
        print(metrics_df.to_string(float_format="%.4f"))

    # --- 8. Generar Gráficas ---
    print("\nGenerando gráficas del portafolio...")

    plot_portfolio_train(port_train, MODEL_NAME)
    plot_portfolio_test(port_test, MODEL_NAME)
    plot_portfolio_validation(port_val, MODEL_NAME)
    plot_portfolio_combined(port_train, port_test, port_val, MODEL_NAME)


if __name__ == "__main__":
    main()