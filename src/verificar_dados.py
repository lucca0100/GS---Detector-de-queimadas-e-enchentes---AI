import pandas as pd


def verificar_csv(caminho, nome):
    df = pd.read_csv(caminho)

    print("=" * 60)
    print(nome)
    print("=" * 60)
    print("Linhas e colunas:", df.shape)
    print("Colunas:")
    print(list(df.columns))
    print("\nPrimeiras linhas:")
    print(df.head())
    print()


def main():
    verificar_csv("data/raw/queimadas_firms.csv", "QUEIMADAS - NASA FIRMS")
    verificar_csv("data/raw/enchentes_eonet.csv", "ENCHENTES - NASA EONET")
    verificar_csv("data/raw/amostras_normais.csv", "AMOSTRAS NORMAIS")


if __name__ == "__main__":
    main()