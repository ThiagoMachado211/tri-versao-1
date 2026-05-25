import pandas as pd
from pathlib import Path


def ler_csv(caminho: str | Path, sep: str = ",") -> pd.DataFrame:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, sep=sep, dtype=str)


def salvar_csv(df: pd.DataFrame, caminho: str | Path, sep: str = ";") -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, sep=sep, index=False, encoding="utf-8-sig")


def ler_respostas(caminho) -> pd.DataFrame:
    return ler_csv(caminho)


def ler_gabarito(caminho) -> pd.DataFrame:
    return ler_csv(caminho)


def ler_notas(caminho) -> pd.DataFrame:
    return ler_csv(caminho)