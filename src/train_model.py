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


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "dados_ultrassom_animais.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "dados_limpos.csv"
MODEL_PATH = ROOT_DIR / "models" / "modelo_peso.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metricas_modelo.csv"
CLEANING_REPORT_PATH = ROOT_DIR / "reports" / "relatorio_limpeza.csv"

TARGET = "PESO"
MISSING_THRESHOLD = 0.50
TEST_SIZE = 0.30
RANDOM_STATE = 42

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


def read_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def convert_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
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

    selected_columns = [*SELECTED_FEATURES, TARGET]
    missing_selected_columns = [col for col in selected_columns if col not in df.columns]
    if missing_selected_columns:
        missing_columns = ", ".join(missing_selected_columns)
        raise ValueError(f"Colunas obrigatorias ausentes na base: {missing_columns}")

    columns_before_selection = df.columns.tolist()
    df = df[selected_columns].copy()
    unselected_columns = [col for col in columns_before_selection if col not in selected_columns]

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
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)

    return {
        "modelo": name,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": root_mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


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
        "Regressao Linear": LinearRegression(),
    }

    results = []
    trained_models = {}
    for name, estimator in candidate_models.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        model.fit(x_train, y_train)
        trained_models[name] = model
        results.append(evaluate_model(name, model, x_test, y_test))

    metrics = pd.DataFrame(results).sort_values(by="rmse")
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
