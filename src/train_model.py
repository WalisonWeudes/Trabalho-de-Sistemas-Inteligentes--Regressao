from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ==========================================
# 1. CONFIGURACOES E CAMINHOS DOS ARQUIVOS
# ==========================================
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "dados_ultrassom_animais.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "dados_limpos.csv"
MODEL_PATH = ROOT_DIR / "models" / "modelo_peso.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metricas_modelo.csv"
CLEANING_REPORT_PATH = ROOT_DIR / "reports" / "relatorio_limpeza.csv"

TARGET = "PESO"
# Mantemos colunas bem esparsas porque a imputacao ainda melhora o ajuste no conjunto atual.
MISSING_THRESHOLD = 0.99
TEST_SIZE = 0.30
RANDOM_STATE = 50

# Campos que nao ajudam no treinamento e costumam agir como identificadores.
IDENTIFIER_COLUMNS = {"ANIMAL", "PAI", "MÃE", "DN"}


# ==========================================
# 2. FUNCOES AUXILIARES E PRE-PROCESSAMENTO
# ==========================================
def read_data(path: Path) -> pd.DataFrame:
    """Carrega os dados brutos a partir de um arquivo CSV."""
    return pd.read_csv(path)


def convert_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas em texto com decimal brasileiro para formato numerico."""
    df = df.copy()

    for column in df.columns:
        if is_object_dtype(df[column]) or is_string_dtype(df[column]):
            values = df[column].astype("string").str.strip()
            numeric_values = pd.to_numeric(
                values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )

            original_not_null = values.notna().sum()
            converted_not_null = numeric_values.notna().sum()

            if original_not_null > 0 and converted_not_null / original_not_null >= 0.80:
                df[column] = numeric_values

    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpa a base e gera um relatorio do que foi removido."""
    df = df.copy()
    rows_original = len(df)

    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed")]
    df = df.drop(columns=unnamed_columns, errors="ignore")

    df = convert_decimal_columns(df)

    df = df.dropna(subset=[TARGET])
    rows_after_target = len(df)

    missing_ratio = df.isna().mean()
    columns_with_many_missing = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=columns_with_many_missing, errors="ignore")

    retained_features = [col for col in df.columns if col != TARGET and col not in IDENTIFIER_COLUMNS]

    columns_before_selection = [col for col in df.columns if col != TARGET]
    unselected_columns = [col for col in columns_before_selection if col not in retained_features]
    df = df[retained_features + [TARGET]].copy()

    cleaning_report = pd.DataFrame(
        [
            {
                "etapa": "colunas_sem_nome",
                "criterio": "nome inicia com Unnamed",
                "linhas_removidas": 0,
                "colunas_removidas": len(unnamed_columns),
                "detalhes": ", ".join(unnamed_columns) if unnamed_columns else "nenhuma",
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
                "detalhes": ", ".join(columns_with_many_missing) if columns_with_many_missing else "nenhuma",
            },
            {
                "etapa": "selecao_variaveis",
                "criterio": "remover identificadores",
                "linhas_removidas": 0,
                "colunas_removidas": len(unselected_columns),
                "detalhes": ", ".join(unselected_columns) if unselected_columns else "nenhuma",
            },
        ]
    )

    return df, cleaning_report


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Constrói o pipeline de transformação dos dados."""
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    """Gera previsoes e calcula as metricas do modelo."""
    predictions = model.predict(x_test)

    return {
        "modelo": name,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": root_mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


# ==========================================
# 3. FUNCAO PRINCIPAL: ORQUESTRACAO DO TREINO
# ==========================================
def train() -> None:
    df = read_data(RAW_DATA_PATH)
    clean_df, cleaning_report = clean_data(df)

    x = clean_df.drop(columns=[TARGET])
    y = clean_df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(x_train)

    candidate_models = {
        "Regressao Linear (log alvo)": LinearRegression(),
        "Ridge (log alvo)": Ridge(alpha=1.0),
        "Elastic Net (log alvo)": ElasticNet(
            alpha=0.0005,
            l1_ratio=0.1,
            max_iter=20000,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    trained_models = {}

    for name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        model.fit(x_train, y_train)
        trained_models[name] = model
        results.append(evaluate_model(name, model, x_test, y_test))

    metrics = pd.DataFrame(results).sort_values(by="r2", ascending=False)
    best_model_name = metrics.iloc[0]["modelo"]
    best_model = trained_models[best_model_name]

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    cleaning_report.to_csv(CLEANING_REPORT_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)

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
