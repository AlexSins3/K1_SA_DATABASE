import pandas as pd

# ---- Charger le fichier ----
df = pd.read_csv("data/Database_K1_SA(old).csv", sep=";",encoding="utf-8")

print("Colonnes trouvées dans le CSV :")
for col in df.columns:
    print(f"- '{col}'")

# ---- Dictionnaires de correction ----
to_shitoryu = ["Anan", "Anan_Dai", "Seisan", "Shisochin", "Chatanyara_Kushanku", "Chibana_No_Kushanku", "Ohan", "Ohan_Dai", "Papuren"]
to_shotokan = ["Enpi", "Gankaku", "Gojushiho_Sho", "Sochin", "Unsu", "Gojushiho_Dai"]

# ---- Correction ShitoRyu ----
mask_shitoryu = df["Kata"].isin(to_shitoryu) & (df["Style"] == "Shotokan")
df.loc[mask_shitoryu, "Style"] = "ShitoRyu"

# ---- Correction Shotokan ----
mask_shotokan = df["Kata"].isin(to_shotokan) & (df["Style"] == "ShitoRyu")
df.loc[mask_shotokan, "Style"] = "Shotokan"

# ---- Sauvegarde ----
df.to_csv("data/Database_K1_SA.csv", sep=";", index=False, encoding="utf-8")

print("✔️ Correction terminée !")
print(f"{mask_shitoryu.sum()} lignes corrigées en ShitoRyu")
print(f"{mask_shotokan.sum()} lignes corrigées en Shotokan")
print("👉 Fichier exporté : Database_K1_SA.csv")
