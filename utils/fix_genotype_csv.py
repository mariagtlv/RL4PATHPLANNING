import pandas as pd

input_csv = "REP-main/rep_results/rep_cnn_nb101_training_results.csv"  
output_csv = "REP-main/rep_results/rep_cnn_nb101_training_results2.csv" 

df = pd.read_csv(input_csv)

df["genotype_full"] = (
    df["genotype_full"]
    .astype(str)
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df.to_csv(output_csv, index=False)

print(f"CSV normalizado guardado en: {output_csv}")
