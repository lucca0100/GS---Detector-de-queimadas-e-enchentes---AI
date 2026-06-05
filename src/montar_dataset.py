import pandas as pd
import numpy as np


def converter_confidence(valor):
    """
    Converte a coluna confidence da FIRMS para valor numérico.

    Alguns sensores retornam:
    l = low
    n = nominal
    h = high

    Outros podem retornar valores numéricos.
    """

    if pd.isna(valor):
        return 0

    valor_str = str(valor).strip().lower()

    mapa = {
        "l": 30,
        "low": 30,
        "n": 60,
        "nominal": 60,
        "h": 90,
        "high": 90
    }

    if valor_str in mapa:
        return mapa[valor_str]

    try:
        return float(valor)
    except ValueError:
        return 0


def preparar_queimadas(caminho):
    df = pd.read_csv(caminho)

    df = df.rename(columns={
        "acq_date": "data"
    })

    df["classe"] = "queimada"
    df["fonte"] = "NASA_FIRMS"

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["bright_ti4"] = pd.to_numeric(df.get("bright_ti4", 0), errors="coerce").fillna(0)
    df["bright_ti5"] = pd.to_numeric(df.get("bright_ti5", 0), errors="coerce").fillna(0)
    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df["scan"] = pd.to_numeric(df.get("scan", 0), errors="coerce").fillna(0)
    df["track"] = pd.to_numeric(df.get("track", 0), errors="coerce").fillna(0)

    df["confidence_num"] = df.get("confidence", 0).apply(converter_confidence)

    df["daynight_num"] = df.get("daynight", "D").map({
        "D": 1,
        "N": 0
    }).fillna(0)

    colunas = [
        "latitude",
        "longitude",
        "data",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "scan",
        "track",
        "confidence_num",
        "daynight_num",
        "fonte",
        "classe"
    ]

    return df[colunas]


def preparar_enchentes(caminho):
    df = pd.read_csv(caminho)

    df["classe"] = "enchente"
    df["fonte"] = "NASA_EONET"

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["bright_ti4"] = 0
    df["bright_ti5"] = 0
    df["frp"] = 0
    df["scan"] = 0
    df["track"] = 0
    df["confidence_num"] = 0
    df["daynight_num"] = 0

    colunas = [
        "latitude",
        "longitude",
        "data",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "scan",
        "track",
        "confidence_num",
        "daynight_num",
        "fonte",
        "classe"
    ]

    return df[colunas]


def preparar_normais(caminho):
    df = pd.read_csv(caminho)

    df["classe"] = "normal"
    df["fonte"] = "GERADO"

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["bright_ti4"] = 0
    df["bright_ti5"] = 0
    df["frp"] = 0
    df["scan"] = 0
    df["track"] = 0
    df["confidence_num"] = 0
    df["daynight_num"] = 0

    colunas = [
        "latitude",
        "longitude",
        "data",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "scan",
        "track",
        "confidence_num",
        "daynight_num",
        "fonte",
        "classe"
    ]

    return df[colunas]


def criar_features_temporais(df):
    df["data_original"] = df["data"]

    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
        utc=True,
        format="mixed"
    )

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia"] = df["data"].dt.day
    df["dia_do_ano"] = df["data"].dt.dayofyear

    df["data"] = df["data"].dt.strftime("%Y-%m-%d")

    return df


def criar_features_derivadas(df):
    df["risco_fogo"] = df["bright_ti4"] * df["confidence_num"]
    df["diferenca_brilho"] = df["bright_ti4"] - df["bright_ti5"]

    # Ainda vamos enriquecer depois com dados de clima.
    df["temperatura"] = np.nan
    df["umidade"] = np.nan
    df["vento"] = np.nan
    df["chuva"] = np.nan

    return df


def main():
    caminho_queimadas = "data/raw/queimadas_firms.csv"
    caminho_enchentes = "data/raw/enchentes_eonet.csv"
    caminho_normais = "data/raw/amostras_normais.csv"

    print("Lendo dados brutos...")

    queimadas = preparar_queimadas(caminho_queimadas)
    enchentes = preparar_enchentes(caminho_enchentes)
    normais = preparar_normais(caminho_normais)

    print("\nQuantidade original:")
    print("Queimadas:", queimadas.shape)
    print("Enchentes:", enchentes.shape)
    print("Normais:", normais.shape)

    print("\nClasses antes do balanceamento:")
    print("Queimadas:", queimadas["classe"].value_counts().to_dict())
    print("Enchentes:", enchentes["classe"].value_counts().to_dict())
    print("Normais:", normais["classe"].value_counts().to_dict())

    n_amostras = min(len(queimadas), len(enchentes), len(normais), 500)

    print("\nQuantidade escolhida por classe:", n_amostras)

    queimadas = queimadas.sample(n=n_amostras, random_state=42)
    enchentes = enchentes.sample(n=n_amostras, random_state=42)
    normais = normais.sample(n=n_amostras, random_state=42)

    df = pd.concat([queimadas, enchentes, normais], ignore_index=True)

    print("\nDistribuição logo após juntar:")
    print(df["classe"].value_counts())

    df = criar_features_temporais(df)

    print("\nQuantidade de datas inválidas por classe:")
    print(df[df["data"].isna()]["classe"].value_counts())

    df = criar_features_derivadas(df)

    antes_drop = df.shape[0]

    df = df.dropna(subset=[
        "latitude",
        "longitude",
        "data",
        "ano",
        "mes",
        "dia",
        "dia_do_ano"
    ])

    depois_drop = df.shape[0]

    print("\nLinhas removidas por dados essenciais inválidos:", antes_drop - depois_drop)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    caminho_saida = "data/processed/dataset_eventos_ambientais.csv"
    df.to_csv(caminho_saida, index=False)

    print("\nDataset processado criado com sucesso.")
    print("Arquivo:", caminho_saida)

    print("\nDimensão final:")
    print(df.shape)

    print("\nDistribuição das classes:")
    print(df["classe"].value_counts())

    print("\nColunas finais:")
    print(list(df.columns))

    print("\nPrimeiras linhas:")
    print(df.head())


if __name__ == "__main__":
    main()