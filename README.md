# Detector de Queimadas e Enchentes com Dados de Satélite e Inteligência Artificial

# Integrantes:
Guilherme Daher - 98611
Gabriel Freitas - 550187
Heitor Nobre - 551539
Vinicius Yamashita - 550908
Lucca Alexandre - 99700

## 1. Contexto do problema

Queimadas e enchentes são eventos ambientais extremos que causam impactos sociais, econômicos e ambientais. O monitoramento desses eventos é essencial para apoiar ações de prevenção, resposta emergencial, planejamento urbano, agricultura, seguros e defesa civil.

Este projeto utiliza dados relacionados à Economia Espacial para desenvolver um pipeline completo de Inteligência Artificial capaz de classificar uma região como normal, afetada por queimada ou afetada por enchente.

## 2. Relação com Economia Espacial

A Economia Espacial envolve o uso de tecnologias, dados e serviços derivados do setor espacial para gerar valor econômico e social. Neste projeto, são utilizados dados provenientes de satélites e APIs da NASA para monitoramento ambiental.

As principais fontes utilizadas foram:

- NASA FIRMS: dados de focos de calor e anomalias térmicas detectadas por satélites.
- NASA EONET: eventos naturais registrados globalmente, incluindo enchentes.
- Amostras normais geradas dentro de uma área geográfica definida.

## 3. Objetivo

Desenvolver um pipeline completo de Machine Learning para classificar eventos ambientais em três classes:

- Normal
- Queimada
- Enchente

## 4. Fontes dos dados

### NASA FIRMS

A NASA FIRMS foi utilizada para coletar dados de focos de calor associados a queimadas. A base contém informações como latitude, longitude, brilho térmico, FRP, sensor, data e confiança da detecção.

### NASA EONET

A NASA EONET foi utilizada para coletar eventos naturais da categoria floods, representando eventos de enchentes.

### Amostras normais

Foram geradas amostras aleatórias dentro de uma área geográfica definida, representando locais sem associação direta com os eventos coletados nas APIs utilizadas.

## 5. Estrutura do dataset

O dataset final foi balanceado com:

- 500 amostras da classe normal
- 500 amostras da classe queimada
- 500 amostras da classe enchente

Total:

- 1.500 linhas
- Mais de 10 colunas

Principais variáveis utilizadas:

- latitude
- longitude
- bright_ti4
- bright_ti5
- frp
- scan
- track
- confidence_num
- daynight_num
- ano
- mes
- dia
- dia_do_ano
- risco_fogo
- diferenca_brilho

## 6. Pré-processamento

As etapas de pré-processamento incluíram:

- Leitura dos dados brutos das APIs
- Padronização dos nomes das colunas
- Conversão de datas
- Conversão de variáveis categóricas para numéricas
- Tratamento de valores ausentes
- Balanceamento das classes
- Criação do dataset processado

## 7. Engenharia de atributos

Foram criadas variáveis derivadas para melhorar a capacidade preditiva dos modelos:

- risco_fogo = bright_ti4 * confidence_num
- diferenca_brilho = bright_ti4 - bright_ti5
- ano, mês, dia e dia do ano extraídos da data

Essas variáveis ajudam o modelo a capturar padrões térmicos e temporais relacionados aos eventos ambientais.

## 8. Modelos utilizados

Foram treinados e comparados três modelos de Machine Learning:

- Logistic Regression
- Random Forest
- XGBoost

O objetivo foi comparar um modelo linear simples com modelos baseados em árvores e ensembles.

## 9. Resultados

Os resultados obtidos foram:

| Modelo | Accuracy | F1 Macro | Precision Macro | Recall Macro |
|---|---:|---:|---:|---:|
| Random Forest | 0.9767 | 0.9767 | 0.9774 | 0.9767 |
| XGBoost | 0.9700 | 0.9700 | 0.9700 | 0.9700 |
| Logistic Regression | 0.9367 | 0.9365 | 0.9400 | 0.9367 |

O melhor modelo foi o Random Forest, escolhido com base no F1 Macro.

## 10. Interpretabilidade com SHAP

Foi utilizado SHAP para analisar a importância das variáveis no modelo final.

As variáveis mais influentes foram:

- longitude
- latitude
- bright_ti4
- track
- frp
- bright_ti5
- scan
- risco_fogo
- confidence_num
- diferenca_brilho

A análise mostrou que o modelo utiliza tanto variáveis espaciais quanto variáveis térmicas para realizar as classificações.

## 11. Aplicação Streamlit

Foi desenvolvida uma aplicação interativa com Streamlit, permitindo:

- Inserir dados manualmente
- Classificar uma região como normal, queimada ou enchente
- Visualizar probabilidades por classe
- Enviar um CSV para classificação em lote
- Baixar o resultado das previsões

## 12. Como executar o projeto localmente

Clone o repositórioe e entre na pasta correta:

```bash
git clone https://github.com/lucca0100/GS---Detector-de-queimadas-e-enchentes---AI.git

cd GS---Detector-de-queimadas-e-enchentes---AI
```

Crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente virtual:

No Windows PowerShell/CMD:

```bash
venv\Scripts\activate
```

No Git Bash:

```bash
source venv/Scripts/activate
```

No Linux/Mac:

```bash
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app/streamlit_app.py
```

Acesse no navegador:

```text
http://localhost:8501
```
## 13. Links

- Link do repositório: https://github.com/lucca0100/GS---Detector-de-queimadas-e-enchentes---AI.git
- Link da aplicação em funcionamento: https://gs---detector-de-queimadas-e-enchentes---ai-bn36naxft7a8lvy8sf.streamlit.app/

## 14. Entendimento dos dados e uso da aplicação

## Explicação das colunas do dataset

O dataset final utilizado no treinamento foi construído a partir de três fontes principais: NASA FIRMS, NASA EONET e amostras normais geradas. Após o pré-processamento, as colunas foram padronizadas para permitir o treinamento dos modelos de Machine Learning.

| Coluna             | Descrição                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `latitude`         | Coordenada geográfica de latitude do evento ou amostra.                                                                            |
| `longitude`        | Coordenada geográfica de longitude do evento ou amostra.                                                                           |
| `data`             | Data da observação ou do evento registrado.                                                                                        |
| `bright_ti4`       | Temperatura de brilho detectada pelo sensor VIIRS na banda I4, associada à intensidade térmica do foco de calor.                   |
| `bright_ti5`       | Temperatura de brilho detectada pelo sensor VIIRS na banda I5, usada como apoio na análise térmica.                                |
| `frp`              | Fire Radiative Power, medida da potência radiativa do fogo. Valores maiores indicam maior intensidade do foco de calor.            |
| `scan`             | Medida relacionada à largura do pixel observado pelo satélite.                                                                     |
| `track`            | Medida relacionada ao comprimento do pixel observado pelo satélite.                                                                |
| `confidence_num`   | Valor numérico criado a partir da confiança da detecção da FIRMS. Valores como low, nominal e high foram convertidos para números. |
| `daynight_num`     | Indica se a observação ocorreu durante o dia ou à noite. Foi codificada como 1 para dia e 0 para noite.                            |
| `ano`              | Ano extraído da data da observação.                                                                                                |
| `mes`              | Mês extraído da data da observação.                                                                                                |
| `dia`              | Dia do mês extraído da data da observação.                                                                                         |
| `dia_do_ano`       | Dia correspondente dentro do ano, variando de 1 a 365 ou 366.                                                                      |
| `risco_fogo`       | Variável criada pela multiplicação entre `bright_ti4` e `confidence_num`, representando um indicador derivado de risco térmico.    |
| `diferenca_brilho` | Diferença entre `bright_ti4` e `bright_ti5`, usada para capturar variações térmicas entre bandas.                                  |
| `classe`           | Variável alvo do modelo. Pode assumir os valores `normal`, `queimada` ou `enchente`.                                               |

## Tratamento e pré-processamento dos dados

O tratamento dos dados foi realizado em várias etapas para transformar dados brutos de diferentes fontes em um dataset único e adequado para Machine Learning.

As principais etapas foram:

1. **Coleta dos dados brutos**

   Os dados de queimadas foram coletados por meio da API NASA FIRMS. Os dados de enchentes foram coletados pela API NASA EONET. Também foram geradas amostras normais dentro de uma área geográfica definida.

2. **Padronização das colunas**

   Como cada fonte de dados possui estrutura própria, as colunas foram padronizadas para que todas as classes tivessem o mesmo formato final.

3. **Criação da variável alvo**

   Cada amostra recebeu uma classe:

   * `queimada`: registros provenientes da NASA FIRMS.
   * `enchente`: registros provenientes da NASA EONET.
   * `normal`: amostras geradas sem associação direta com eventos das APIs utilizadas.

4. **Conversão de dados categóricos**

   A coluna de confiança da FIRMS podia conter valores categóricos, como `l`, `n` e `h`. Esses valores foram convertidos para números:

   * `l` ou low: 30
   * `n` ou nominal: 60
   * `h` ou high: 90

5. **Tratamento de valores ausentes**

   Algumas colunas existem apenas para dados de queimadas, como `bright_ti4`, `bright_ti5`, `frp`, `scan`, `track` e `confidence_num`. Para eventos de enchente e amostras normais, esses campos foram preenchidos com zero.

6. **Conversão de datas**

   A coluna de data foi convertida para o formato de data e utilizada para criar novas variáveis temporais, como ano, mês, dia e dia do ano.

7. **Engenharia de atributos**

   Foram criadas variáveis derivadas para melhorar a capacidade de aprendizado dos modelos:

   * `risco_fogo = bright_ti4 * confidence_num`
   * `diferenca_brilho = bright_ti4 - bright_ti5`

8. **Balanceamento das classes**

   Para evitar que o modelo aprendesse mais uma classe do que outra, o dataset final foi balanceado com 500 amostras por classe:

   * 500 amostras normais
   * 500 amostras de queimadas
   * 500 amostras de enchentes

   O dataset final possui 1.500 linhas.

## Como utilizar a aplicação

A aplicação foi desenvolvida com Streamlit e permite realizar previsões de duas formas: manualmente ou por upload de arquivo CSV.

### Previsão manual

Na aba **Previsão manual**, o usuário pode preencher os valores das variáveis utilizadas pelo modelo, como latitude, longitude, brilho térmico, FRP, confiança, data e período da observação.

Após preencher os campos, basta clicar em **Classificar região**. O app retorna:

* Classe prevista: normal, queimada ou enchente.
* Probabilidade de cada classe.
* Gráfico de probabilidades.

### Previsão por CSV

Na aba **Previsão por CSV**, o usuário pode enviar um arquivo contendo várias amostras para classificação em lote.

O CSV deve conter as seguintes colunas:

```text
latitude, longitude, bright_ti4, bright_ti5, frp, scan, track, confidence_num, daynight_num, ano, mes, dia, dia_do_ano, risco_fogo, diferenca_brilho
```

Após o envio do arquivo, o app gera uma tabela com:

* Dados originais enviados.
* Classe prevista para cada linha.
* Probabilidade da classe normal.
* Probabilidade da classe queimada.
* Probabilidade da classe enchente.

Também é possível baixar o resultado em um novo arquivo CSV.

### Como interpretar o resultado

A classe prevista representa a categoria que o modelo considera mais provável para aquela amostra.

As probabilidades mostram o grau de confiança do modelo para cada classe. Por exemplo:

| Classe   | Probabilidade |
| -------- | ------------: |
| normal   |          0.02 |
| queimada |          0.97 |
| enchente |          0.01 |

Nesse exemplo, o modelo classificaria a amostra como `queimada`, pois essa classe possui a maior probabilidade.

## Informações importantes para o usuário

Este projeto possui finalidade acadêmica e demonstrativa. A aplicação não substitui sistemas oficiais de monitoramento ambiental, defesa civil ou análise especializada.

O modelo foi treinado com dados de diferentes fontes, o que pode influenciar seu comportamento. A análise com SHAP mostrou que variáveis como latitude e longitude tiveram grande influência nas previsões, indicando que parte da classificação pode estar associada à distribuição geográfica dos dados utilizados.

Portanto, os resultados devem ser interpretados como uma demonstração de pipeline de Inteligência Artificial aplicado à Economia Espacial, e não como uma ferramenta operacional oficial de detecção de desastres.
