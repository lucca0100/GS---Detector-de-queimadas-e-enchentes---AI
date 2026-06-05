import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalizar_shap_values(shap_values, n_classes, n_features):
    """
    Normaliza o retorno do SHAP para uma lista:
    [
        shap_values_classe_0,
        shap_values_classe_1,
        shap_values_classe_2
    ]

    Isso evita erro entre versões diferentes do SHAP.
    """

    if isinstance(shap_values, list):
        return shap_values

    if isinstance(shap_values, np.ndarray):
        # Caso binário ou regressão
        if shap_values.ndim == 2:
            return [shap_values]

        # Caso multiclasse
        if shap_values.ndim == 3:
            # Formato: linhas, features, classes
            if shap_values.shape[1] == n_features and shap_values.shape[2] == n_classes:
                return [shap_values[:, :, i] for i in range(n_classes)]

            # Formato: classes, linhas, features
            if shap_values.shape[0] == n_classes and shap_values.shape[2] == n_features:
                return [shap_values[i, :, :] for i in range(n_classes)]

    raise ValueError(f"Formato inesperado de shap_values: {type(shap_values)}")


def main():
    os.makedirs("reports", exist_ok=True)

    print("Carregando modelo e dados...")

    modelo = joblib.load("models/modelo_final.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    features = joblib.load("models/colunas_modelo.pkl")

    df = pd.read_csv("data/processed/dataset_eventos_ambientais.csv")

    X = df[features].copy()

    for coluna in features:
        X[coluna] = pd.to_numeric(X[coluna], errors="coerce")

    X = X.fillna(0)

    # Usamos uma amostra para deixar o SHAP mais rápido
    X_sample = X.sample(n=min(300, len(X)), random_state=42)

    classes = list(label_encoder.classes_)
    n_classes = len(classes)
    n_features = len(features)

    print("Modelo carregado:", type(modelo))
    print("Classes:", classes)
    print("Quantidade de features:", n_features)
    print("Amostra usada no SHAP:", X_sample.shape)

    print("\nCalculando valores SHAP...")

    explainer = shap.TreeExplainer(modelo)
    shap_values_raw = explainer.shap_values(X_sample)

    shap_values_por_classe = normalizar_shap_values(
        shap_values_raw,
        n_classes=n_classes,
        n_features=n_features
    )

    print("SHAP calculado com sucesso.")

    # Importância global média considerando todas as classes
    importancias_por_classe = []

    for i, nome_classe in enumerate(classes):
        valores_classe = shap_values_por_classe[i]
        importancia_media = np.abs(valores_classe).mean(axis=0)

        df_importancia_classe = pd.DataFrame({
            "classe": nome_classe,
            "feature": features,
            "importancia_media_abs_shap": importancia_media
        })

        importancias_por_classe.append(df_importancia_classe)

    df_importancias = pd.concat(importancias_por_classe, ignore_index=True)

    df_importancia_global = (
        df_importancias
        .groupby("feature", as_index=False)["importancia_media_abs_shap"]
        .mean()
        .sort_values(by="importancia_media_abs_shap", ascending=False)
    )

    df_importancias.to_csv("reports/shap_importancia_por_classe.csv", index=False)
    df_importancia_global.to_csv("reports/shap_importancia_global.csv", index=False)

    print("\nImportância global das variáveis:")
    print(df_importancia_global)

    # Gráfico global de barras usando os valores SHAP médios
    top_features = df_importancia_global.head(15).sort_values(
        by="importancia_media_abs_shap",
        ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["importancia_media_abs_shap"]
    )
    plt.xlabel("Importância média absoluta SHAP")
    plt.title("Importância Global das Variáveis - SHAP")
    plt.tight_layout()
    plt.savefig("reports/shap_importancia_global.png", dpi=300)
    plt.close()

    print("\nGráfico global salvo em:")
    print("reports/shap_importancia_global.png")

    # Summary plot por classe
    for i, nome_classe in enumerate(classes):
        print(f"Gerando SHAP summary plot para a classe: {nome_classe}")

        plt.figure()
        shap.summary_plot(
            shap_values_por_classe[i],
            X_sample,
            feature_names=features,
            show=False
        )
        plt.title(f"SHAP Summary Plot - Classe {nome_classe}")
        plt.tight_layout()

        nome_arquivo = nome_classe.lower().replace(" ", "_")
        caminho_saida = f"reports/shap_summary_{nome_arquivo}.png"

        plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close()

        print("Salvo em:", caminho_saida)

    print("\nArquivos SHAP gerados com sucesso:")
    print("reports/shap_importancia_global.csv")
    print("reports/shap_importancia_por_classe.csv")
    print("reports/shap_importancia_global.png")

    for classe in classes:
        nome_arquivo = classe.lower().replace(" ", "_")
        print(f"reports/shap_summary_{nome_arquivo}.png")


if __name__ == "__main__":
    main()