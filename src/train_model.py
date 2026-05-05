from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS DOS ARQUIVOS
# ==========================================
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "dados_ultrassom_animais.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "dados_limpos.csv"
MODEL_PATH = ROOT_DIR / "models" / "modelo_peso.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metricas_modelo.csv"
CLEANING_REPORT_PATH = ROOT_DIR / "reports" / "relatorio_limpeza.csv"

# Variável alvo (o que queremos prever)
TARGET = "PESO"
# Limite máximo de valores nulos permitidos em uma coluna (50%)
MISSING_THRESHOLD = 0.50
# Proporção da base que será separada para teste (30%)
TEST_SIZE = 0.30
# Semente aleatória para garantir reprodutibilidade
RANDOM_STATE = 50

# Funcionalidades (features) que serão utilizadas no treinamento
SELECTED_FEATURES = [
    "AC",
    "AG",
    "CC",
    "AP",
    "P.C",
    "CT",
    "CO",
    "CCAB",
    "LIL",
    "LIS",
    "Cga",
    "Cper",
    "PerPe",
    "Ccau",
    "DC",
]

# ==========================================
# 2. FUNÇÕES AUXILIARES E PRÉ-PROCESSAMENTO
# ==========================================

def read_data(path: Path) -> pd.DataFrame:
    """Carrega os dados brutos a partir de um arquivo CSV."""
    return pd.read_csv(path)


def convert_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas que estão como texto devido a vírgulas decimais para números."""
    df = df.copy()

    for column in df.columns:
        # Se a coluna for do tipo texto/objeto
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            values = df[column].astype("string").str.strip()
            # Remove pontos (milhares) e troca vírgula por ponto (decimais)
            numeric_values = pd.to_numeric(
                values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )

            original_not_null = values.notna().sum()
            converted_not_null = numeric_values.notna().sum()
            # Se a conversão for bem-sucedida para pelo menos 80% dos dados, aplica a mudança
            if original_not_null > 0 and converted_not_null / original_not_null >= 0.80:
                df[column] = numeric_values

    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza a limpeza inicial dos dados e gera um relatório do que foi alterado."""
    df = df.copy()
    rows_original = len(df)
    
    # Remove colunas vazias geradas na importação do Excel/CSV
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed")]
    df = df.drop(columns=unnamed_columns, errors="ignore")
    
    # Corrige formatação de números
    df = convert_decimal_columns(df)

    # Remove linhas que não possuem a variável alvo (não servem para treino supervisionado)
    df = df.dropna(subset=[TARGET])
    rows_after_target = len(df)

    # Remove colunas que possuem muitos valores nulos (acima da MISSING_THRESHOLD)
    missing_ratio = df.isna().mean()
    columns_with_many_missing = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=columns_with_many_missing, errors="ignore")

    # Verifica se as colunas selecionadas para o modelo existem na base
    selected_columns = [*SELECTED_FEATURES, TARGET]
    missing_selected_columns = [col for col in selected_columns if col not in df.columns]
    if missing_selected_columns:
        missing_columns = ", ".join(missing_selected_columns)
        raise ValueError(f"Colunas obrigatorias ausentes na base: {missing_columns}")

    # Filtra o DataFrame apenas com as colunas que importam para o modelo
    columns_before_selection = df.columns.tolist()
    df = df[selected_columns].copy()
    unselected_columns = [col for col in columns_before_selection if col not in selected_columns]

    # Gera um relatório documentando tudo que foi limpo/removido
    cleaning_report = pd.DataFrame(
        [
            {
                "etapa": "colunas_sem_nome",
                "criterio": "nome inicia com Unnamed",
                "linhas_removidas": 0,
                "colunas_removidas": len(unnamed_columns),
                "detalhes": "colunas auxiliares vazias da planilha",
            },
            {
                "etapa": "peso_vazio",
                "criterio": f"{TARGET} ausente",
                "linhas_removidas": rows_original - rows_after_target,
                "colunas_removidas": 0,
                "detalhes": "linhas sem variavel alvo nao servem para treino supervisionado",
            },
            {
                "etapa": "muitos_faltantes",
                "criterio": f"mais de {MISSING_THRESHOLD:.0%} ausente",
                "linhas_removidas": 0,
                "colunas_removidas": len(columns_with_many_missing),
                "detalhes": ", ".join(columns_with_many_missing),
            },
            {
                "etapa": "selecao_variaveis",
                "criterio": "manter apenas medidas importantes informadas",
                "linhas_removidas": 0,
                "colunas_removidas": len(unselected_columns),
                "detalhes": ", ".join(unselected_columns),
            },
        ]
    )

    return df, cleaning_report


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Constrói o pipeline de transformação dos dados (Apenas Numérico Ativo)."""
    
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    
    # 1. Variáveis categóricas comentadas, pois o banco é 100% numérico
    # categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    # ==========================================
    # PIPELINE DE NÚMEROS (Ativo)
    # ==========================================
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()), # Normalização Min-Max ativa
            # ("scaler", StandardScaler()), # Variante Z-Score (Padronização) comentada
        ]
    )
    
    # ==========================================
    # PIPELINE DE CATEGORIAS/TEXTOS (Comentado)
    # ==========================================
    # categorical_pipeline = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #         ("encoder", OneHotEncoder(handle_unknown="ignore")),
    #     ]
    # )

    # Junta os pipelines. O categórico foi removido da lista final.
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            # ("categorical", categorical_pipeline, categorical_features), # Comentado!
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    """Realiza as previsões na base de teste e calcula as métricas de erro."""
    predictions = model.predict(x_test)

    return {
        "modelo": name,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": root_mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

# ==========================================
# 3. FUNÇÃO PRINCIPAL: ORQUESTRAÇÃO DO TREINO
# ==========================================



def train() -> None:
    # Etapa 3.1: Carregamento e Limpeza dos Dados
    df = read_data(RAW_DATA_PATH)
    clean_df, cleaning_report = clean_data(df)

    # Etapa 3.2: Separação entre Variáveis Preditoras (X) e Alvo (y)
    x = clean_df.drop(columns=[TARGET])
    y = clean_df[TARGET]

    # Etapa 3.3: Divisão da base em Treino (para aprender) e Teste (para avaliar)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Etapa 3.4: Construção do Pré-processador (Agora apenas numérico)
    preprocessor = build_preprocessor(x_train)
    
    # Etapa 3.5: Definição dos Modelos Candidatos
    candidate_models = {
        "Regressao Linear": LinearRegression(),
    }

    # Etapa 3.6: Treinamento e Avaliação
    results = []
    trained_models = {}
    
    for name, estimator in candidate_models.items():
        # Cria um pipeline juntando o pré-processamento com o modelo de regressão
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        
        # Treina (ajusta) o modelo usando os dados de treino
        model.fit(x_train, y_train)
        trained_models[name] = model
        
        # Avalia o modelo usando os dados de teste
        results.append(evaluate_model(name, model, x_test, y_test))

    # Etapa 3.7: Seleção do Melhor Modelo (neste caso, com menor RMSE)
    metrics = pd.DataFrame(results).sort_values(by="rmse")
    best_model_name = metrics.iloc[0]["modelo"]
    best_model = trained_models[best_model_name]

    # Etapa 3.8: Criação de Diretórios (caso não existam) para salvar os artefatos
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Etapa 3.9: Salvamento dos Artefatos (Dados limpos, Métricas, Relatório e o Modelo)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    cleaning_report.to_csv(CLEANING_REPORT_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)

    # Etapa 3.10: Exibição dos Resultados no Terminal
    print("Relatorio da limpeza:")
    print(cleaning_report.to_string(index=False))
    print(f"\nLinhas usadas: {len(clean_df)}")
    print(f"Colunas finais: {len(clean_df.columns)}")
    print(f"Treino: {len(x_train)} registros | Teste: {len(x_test)} registros")
    print("\nMetricas:")
    print(metrics.to_string(index=False))
    print(f"\nMelhor modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    train()