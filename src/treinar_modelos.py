import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


def carregar_dataset(caminho):
    df = pd.read_csv(caminho)

    print("Dataset carregado:")
    print(df.shape)

    print("\nDistribuição das classes:")
    print(df["classe"].value_counts())

    return df


def preparar_dados(df):
    """
    Prepara X e y para treinamento.

    Nesta primeira versão, ainda não usamos temperatura, umidade, vento e chuva,
    porque essas colunas ainda estão vazias. Depois vamos enriquecer com OpenWeather.
    """

    features = [
        "latitude",
        "longitude",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "scan",
        "track",
        "confidence_num",
        "daynight_num",
        "ano",
        "mes",
        "dia",
        "dia_do_ano",
        "risco_fogo",
        "diferenca_brilho"
    ]

    X = df[features].copy()

    # Garantir que tudo é numérico
    for coluna in features:
        X[coluna] = pd.to_numeric(X[coluna], errors="coerce")

    X = X.fillna(0)

    y = df["classe"].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nMapeamento das classes:")
    for classe, codigo in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"{classe} -> {codigo}")

    return X, y_encoded, label_encoder, features


def avaliar_modelo(nome, modelo, X_test, y_test, label_encoder):
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print(f"Modelo: {nome}")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro: {recall_macro:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    matriz = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=label_encoder.classes_
    )

    disp.plot(values_format="d")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    nome_arquivo = nome.lower().replace(" ", "_")
    caminho_figura = f"reports/matriz_confusao_{nome_arquivo}.png"
    plt.savefig(caminho_figura)
    plt.close()

    print(f"Matriz de confusão salva em: {caminho_figura}")

    return {
        "modelo": nome,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    caminho_dataset = "data/processed/dataset_eventos_ambientais.csv"

    df = carregar_dataset(caminho_dataset)

    X, y, label_encoder, features = preparar_dados(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTamanho dos conjuntos:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    modelos = {}

    modelos["Logistic Regression"] = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    modelos["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    modelos["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    resultados = []

    melhor_modelo = None
    melhor_nome = None
    melhor_f1 = -1

    for nome, modelo in modelos.items():
        print("\nTreinando:", nome)
        modelo.fit(X_train, y_train)

        resultado = avaliar_modelo(
            nome=nome,
            modelo=modelo,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder
        )

        resultados.append(resultado)

        if resultado["f1_macro"] > melhor_f1:
            melhor_f1 = resultado["f1_macro"]
            melhor_modelo = modelo
            melhor_nome = nome

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by="f1_macro", ascending=False)

    print("\n" + "=" * 70)
    print("Comparação final dos modelos")
    print("=" * 70)
    print(df_resultados)

    df_resultados.to_csv("reports/resultados_modelos.csv", index=False)

    joblib.dump(melhor_modelo, "models/modelo_final.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    joblib.dump(features, "models/colunas_modelo.pkl")

    print("\nMelhor modelo:", melhor_nome)
    print(f"Melhor F1 Macro: {melhor_f1:.4f}")
    print("\nArquivos salvos:")
    print("models/modelo_final.pkl")
    print("models/label_encoder.pkl")
    print("models/colunas_modelo.pkl")
    print("reports/resultados_modelos.csv")


if __name__ == "__main__":
    main()