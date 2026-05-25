import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(
    nota_real: np.ndarray,
    nota_estimada: np.ndarray
) -> dict:
    """
    Calcula métricas de validação entre nota real e nota estimada.
    """

    nota_real = np.asarray(nota_real, dtype=float)
    nota_estimada = np.asarray(nota_estimada, dtype=float)

    erro = nota_estimada - nota_real
    erro_abs = np.abs(erro)

    correlacao = np.corrcoef(nota_real, nota_estimada)[0, 1]
    mae = mean_absolute_error(nota_real, nota_estimada)
    rmse = np.sqrt(mean_squared_error(nota_real, nota_estimada))
    r2 = r2_score(nota_real, nota_estimada)

    return {
        "correlacao": float(correlacao),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "erro_medio": float(erro.mean()),
        "erro_absoluto_mediano": float(np.median(erro_abs)),
        "erro_absoluto_maximo": float(erro_abs.max())
    }


def montar_base_validacao(
    df: pd.DataFrame,
    coluna_id: str,
    coluna_nota_real: str,
    theta: np.ndarray,
    erro_theta: np.ndarray | None,
    nota_estimada: np.ndarray
) -> pd.DataFrame:
    """
    Monta uma base final contendo:
    - ID do aluno
    - nota real
    - theta estimado
    - erro padrão do theta
    - nota estimada
    - erro
    - erro absoluto
    """

    resultado = pd.DataFrame()

    resultado[coluna_id] = df[coluna_id].values
    resultado[coluna_nota_real] = df[coluna_nota_real].astype(float).values
    resultado["theta_eap"] = theta

    if erro_theta is not None:
        resultado["erro_theta"] = erro_theta

    resultado["nota_estimada"] = nota_estimada
    resultado["erro"] = resultado["nota_estimada"] - resultado[coluna_nota_real]
    resultado["erro_absoluto"] = resultado["erro"].abs()

    return resultado


def erro_por_faixa(
    df_validacao: pd.DataFrame,
    coluna_nota_real: str,
    tamanho_faixa: int = 50
) -> pd.DataFrame:
    """
    Calcula erro médio e erro absoluto médio por faixa de nota real.
    """

    nota_min = int(df_validacao[coluna_nota_real].min() // tamanho_faixa * tamanho_faixa)
    nota_max = int(df_validacao[coluna_nota_real].max() // tamanho_faixa * tamanho_faixa + tamanho_faixa)

    bins = list(range(nota_min, nota_max + tamanho_faixa, tamanho_faixa))

    df = df_validacao.copy()
    df["faixa_nota"] = pd.cut(
        df[coluna_nota_real],
        bins=bins,
        include_lowest=True
    )

    resumo = (
        df.groupby("faixa_nota", observed=True)
        .agg(
            n=("erro", "size"),
            nota_real_media=(coluna_nota_real, "mean"),
            nota_estimada_media=("nota_estimada", "mean"),
            erro_medio=("erro", "mean"),
            erro_absoluto_medio=("erro_absoluto", "mean"),
            erro_absoluto_mediano=("erro_absoluto", "median")
        )
        .reset_index()
    )

    return resumo


def salvar_metricas(metricas: dict, caminho_saida) -> None:
    """
    Salva as métricas em CSV.
    """

    df = pd.DataFrame([metricas])
    df.to_csv(caminho_saida, sep=";", index=False, encoding="utf-8-sig")