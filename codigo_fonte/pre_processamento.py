import pandas as pd
import numpy as np



def padronizar_respostas(df: pd.DataFrame, colunas_itens: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in colunas_itens:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df



def extrair_gabarito(df_gabarito: pd.DataFrame, colunas_itens: list[str]) -> pd.Series:
  gabarito = df_gabarito.iloc[0][colunas_itens].copy()
  gabarito = (gabarito.astype(str).str.strip().str.lower())

  return gabarito



def filtrar_alunos_minimo_respostas(
    df_respostas: pd.DataFrame,
    colunas_itens: list[str],
    minimo_respostas: int = 1
) -> pd.DataFrame:
    """
    Remove alunos com poucas respostas válidas.
    """

    df = df_respostas.copy()

    respostas_validas = ["a", "b", "c", "d", "e"]

    contagem_validas = (
        df[colunas_itens]
        .isin(respostas_validas)
        .sum(axis=1)
    )

    df["n_validas"] = contagem_validas

    df_filtrado = df[df["n_validas"] >= minimo_respostas].copy()

    df_filtrado = df_filtrado.drop(columns=["n_validas"])

    return df_filtrado



def corrigir_respostas(df_respostas: pd.DataFrame, gabarito: pd.Series, colunas_itens: list[str], coluna_id: str) -> pd.DataFrame:
  df = padronizar_respostas(df_respostas, colunas_itens)
  df_corrigido = pd.DataFrame()
  df_corrigido[coluna_id] = df_respostas[coluna_id]

  for col in colunas_itens:
    df_corrigido[col] = np.where(df[col] == gabarito[col], 1, 0)

  return df_corrigido



def gerar_matriz_respostas(df_corrigido: pd.DataFrame, colunas_itens: list[str]) -> np.ndarray:
    return df_corrigido[colunas_itens].astype(int).to_numpy()



def verificar_frequencias(df_respostas: pd.DataFrame, colunas_itens: list[str]) -> pd.DataFrame:
    registros = []

    for col in colunas_itens:
        freq = df_respostas[col].value_counts(dropna=False).to_dict()

        linha = {"item": col}
        linha.update(freq)

        registros.append(linha)

    return pd.DataFrame(registros).fillna(0)