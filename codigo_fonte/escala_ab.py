import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Ajuste da proficiência usando as notas reais:
def ajustar_ab_por_regressao(
    theta: np.ndarray,
    nota_real: np.ndarray
) -> dict:

    theta = np.asarray(theta).reshape(-1, 1)
    nota_real = np.asarray(nota_real, dtype=float)

    modelo = LinearRegression()
    modelo.fit(theta, nota_real)

    A = float(modelo.coef_[0])
    B = float(modelo.intercept_)

    return {
        "A": A,
        "B": B,
        "metodo": "regressao_linear_com_notas_reais"
    }



def aplicar_ab(theta: np.ndarray, A: float, B: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)

    return A * theta + B



# Ajuste da proficiência sem usar as notas reais
def ajustar_ab_por_media_dp(theta: np.ndarray,
    media_alvo: float = 500,
    dp_alvo: float = 100
) -> dict:

    theta = np.asarray(theta, dtype=float)

    media_theta = theta.mean()
    dp_theta = theta.std(ddof=1)

    A = dp_alvo / dp_theta
    B = media_alvo - A * media_theta

    return {
        "A": float(A),
        "B": float(B),
        "media_theta": float(media_theta),
        "dp_theta": float(dp_theta),
        "media_alvo": float(media_alvo),
        "dp_alvo": float(dp_alvo),
        "metodo": "media_dp_alvo"
    }



def salvar_parametros_ab(parametros: dict, caminho_saida) -> None:
    df = pd.DataFrame([parametros])
    df.to_csv(caminho_saida, sep=";", index=False, encoding="utf-8-sig")