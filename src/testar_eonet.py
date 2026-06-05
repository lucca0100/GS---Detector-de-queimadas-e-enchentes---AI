import requests
import pandas as pd


def testar_eonet_enchentes():
    url = "https://eonet.gsfc.nasa.gov/api/v3/events"

    params = {
        "category": "floods",
        "status": "all",
        "limit": 500
    }

    print("Consultando API NASA EONET para enchentes...")

    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        print("Erro na requisição.")
        print("Status code:", response.status_code)
        print("Resposta:")
        print(response.text[:1000])
        response.raise_for_status()

    data = response.json()
    eventos = data.get("events", [])

    linhas = []

    for evento in eventos:
        titulo = evento.get("title")
        categorias = evento.get("categories", [])
        geometrias = evento.get("geometry", [])

        categoria = categorias[0]["title"] if categorias else None

        for geo in geometrias:
            coords = geo.get("coordinates")
            data_evento = geo.get("date")

            if isinstance(coords, list) and len(coords) >= 2:
                longitude = coords[0]
                latitude = coords[1]

                linhas.append({
                    "titulo_evento": titulo,
                    "categoria": categoria,
                    "data": data_evento,
                    "longitude": longitude,
                    "latitude": latitude,
                    "classe": "enchente"
                })

    df = pd.DataFrame(linhas)

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nQuantidade de linhas e colunas:")
    print(df.shape)

    if df.empty:
        print("Nenhum evento de enchente foi retornado.")
        return

    caminho_saida = "data/raw/enchentes_eonet.csv"
    df.to_csv(caminho_saida, index=False)

    print(f"\nArquivo salvo em: {caminho_saida}")
    print("\nColunas retornadas:")
    print(list(df.columns))


if __name__ == "__main__":
    testar_eonet_enchentes()