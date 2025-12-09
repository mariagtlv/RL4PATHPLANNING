import pandas as pd

def main():
    parquet_path = "REP-main/rep_results/rep_cnn_darts_results.parquet"   
    csv_path = "REP-main/rep_results/rep_cnn_darts_results.csv"          

    try:
        print(f"Cargando archivo Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        print(f"Guardando CSV en: {csv_path}")
        df.to_csv(csv_path, index=False)

        print("Conversión completada correctamente.")

    except Exception as e:
        print(f"Error durante la conversión: {e}")

if __name__ == "__main__":
    main()
