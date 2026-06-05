# Detector de Queimadas e Enchentes com Dados de Satélite e Inteligência Artificial

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

Clone o repositório:

```bash
git clone LINK_DO_REPOSITORIO
cd NOME_DO_REPOSITORIO