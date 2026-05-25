import pandas as pd
from sklearn.model_selection import train_test_split


def dividir_treino_teste(df: pd.DataFrame, tamanho_teste: float = 0.20, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_treino, df_teste = train_test_split(df, test_size=tamanho_teste, random_state=seed, shuffle=True)
    df_treino = df_treino.reset_index(drop=True)
    df_teste = df_teste.reset_index(drop=True)

    return df_treino, df_teste