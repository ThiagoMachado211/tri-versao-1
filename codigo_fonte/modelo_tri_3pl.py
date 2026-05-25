import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def probabilidade_3pl(theta, a, b, c):
    """
    Modelo logístico de 3 parâmetros:

        P(X=1|theta) = c + (1-c) * logistic(a * (theta - b))
    """

    return c + (1 - c) * expit(a * (theta - b))


def estimar_theta_inicial(matriz_respostas: np.ndarray) -> np.ndarray:
    """
    Estima um theta inicial simples usando proporção de acertos.

    Essa etapa serve apenas como ponto de partida para calibrar os itens.
    """

    proporcao = matriz_respostas.mean(axis=1)

    proporcao = np.clip(proporcao, 0.01, 0.99)

    theta_inicial = np.log(proporcao / (1 - proporcao))

    theta_inicial = (theta_inicial - theta_inicial.mean()) / theta_inicial.std(ddof=1)

    theta_inicial = np.clip(theta_inicial, -4, 4)

    return theta_inicial


def log_verossimilhanca_negativa_item(params, theta, respostas_item):
    """
    Função objetivo para estimar os parâmetros de um item.

    params:
        params[0] = log(a)
        params[1] = b
        params[2] = logit(c)
    """

    log_a, b, logit_c = params

    a = np.exp(log_a)
    c = expit(logit_c)

    p = probabilidade_3pl(theta, a, b, c)

    p = np.clip(p, 1e-8, 1 - 1e-8)

    ll = respostas_item * np.log(p) + (1 - respostas_item) * np.log(1 - p)

    return -np.sum(ll)


def estimar_parametros_item(theta, respostas_item):
    """
    Estima os parâmetros a, b e c de um único item.
    """

    chute_inicial = 0.20

    params_iniciais = np.array([
        np.log(1.0),                # log(a)
        0.0,                        # b
        np.log(chute_inicial / (1 - chute_inicial))  # logit(c)
    ])

    limites = [
        (np.log(0.20), np.log(4.00)),   # a entre 0.20 e 4.00
        (-4.00, 4.00),                 # b entre -4 e 4
        (np.log(0.01 / 0.99), np.log(0.35 / 0.65))  # c entre 0.01 e 0.35
    ]

    resultado = minimize(
        log_verossimilhanca_negativa_item,
        params_iniciais,
        args=(theta, respostas_item),
        method="L-BFGS-B",
        bounds=limites
    )

    if not resultado.success:
        log_a, b, logit_c = params_iniciais
    else:
        log_a, b, logit_c = resultado.x

    a = np.exp(log_a)
    c = expit(logit_c)

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "convergiu": bool(resultado.success),
        "loglike_neg": float(resultado.fun) if resultado.success else np.nan
    }


def calibrar_3pl(
    matriz_respostas: np.ndarray,
    nomes_itens: list[str] | None = None
) -> pd.DataFrame:
    """
    Calibra os parâmetros dos itens pelo modelo 3PL.

    Observação importante:
    Esta implementação estima os itens usando um theta inicial baseado
    na proporção de acertos. Portanto, é uma aproximação prática.

    Para calibração psicométrica mais rigorosa, o ideal é usar EM marginal,
    como no pacote mirt do R.
    """

    matriz_respostas = np.asarray(matriz_respostas, dtype=int)

    n_alunos, n_itens = matriz_respostas.shape

    if nomes_itens is None:
        nomes_itens = [f"Q{i}" for i in range(1, n_itens + 1)]

    theta_inicial = estimar_theta_inicial(matriz_respostas)

    registros = []

    for j in range(n_itens):
        respostas_item = matriz_respostas[:, j]

        parametros = estimar_parametros_item(
            theta=theta_inicial,
            respostas_item=respostas_item
        )

        registros.append({
            "item": nomes_itens[j],
            **parametros
        })

    return pd.DataFrame(registros)


def parametros_para_matriz(df_parametros: pd.DataFrame) -> np.ndarray:
    """
    Converte DataFrame de parâmetros para matriz numpy com colunas:

        a, b, c
    """

    return df_parametros[["a", "b", "c"]].astype(float).to_numpy()


def remover_itens_baixa_discriminacao(
    df_parametros: pd.DataFrame,
    limite_a: float = 0.30
) -> list[str]:
    """
    Identifica itens com baixa discriminação.

    Retorna a lista dos itens cujo parâmetro a é menor que o limite.
    """

    itens_ruins = df_parametros.loc[
        df_parametros["a"] < limite_a,
        "item"
    ].tolist()

    return itens_ruins