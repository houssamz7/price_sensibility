from src.pipeline_clean import build_clean_dataset, save_clean

raw_path = "data/raw/Projet_Sensibilite_Prix1.csv"
out_path = "data/clean/Projet_Sensibilite_Prix_clean.csv"

df_clean = build_clean_dataset(raw_path)
save_clean(df_clean, out_path)

print("Clean saved:", out_path)