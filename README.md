# Predicao do peso de animais por regressao

## Objetivo

Treinar um modelo de regressao para estimar o `PESO` do animal a partir das demais caracteristicas disponiveis na base.

## Estrutura

```text
.
├── data/
│   ├── raw/                  # base original
│   └── processed/            # base apos limpeza
├── models/                   # modelo treinado
├── notebooks/                # notebook do trabalho
├── reports/                  # metricas e saidas da avaliacao
│   └── figures/
├── src/                      # codigo fonte reproduzivel
├── README.md
└── requirements.txt
```

## Fluxo do projeto

1. Carregar a base `data/raw/dados_ultrassom_animais.csv`.
2. Definir `PESO` como rotulo (`y`).
3. Remover colunas sem nome e colunas com mais de 50% de dados faltantes.
4. Manter apenas as medidas consideradas mais importantes: `AC`, `AG`, `CC`, `AP`, `P.C`, `CT`, `CO`, `CCAB`, `LIL`, `LIS`, `Cga`, `Cper`, `PerPe`, `Ccau` e `DC`.
5. Converter valores numericos com virgula decimal para formato numerico.
6. Separar os dados em 70% treino e 30% teste.
7. Tratar valores faltantes com imputacao.
8. Treinar o modelo de regressao linear e avaliar com MAE, RMSE e R2.

## Como executar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute o treinamento:

```bash
python src/train_model.py
```

As saidas geradas sao:

- `data/processed/dados_limpos.csv`
- `reports/metricas_modelo.csv`
- `reports/relatorio_limpeza.csv`
- `models/modelo_peso.joblib`
