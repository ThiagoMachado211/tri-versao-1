import numpy as np


def probabilidade_3pl(theta, a, b, c):
    theta = np.asarray(theta)
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def criar_grid_theta(theta_min=-4, theta_max=4, n_pontos=61):
    return np.linspace(theta_min, theta_max, n_pontos)


def densidade_prior_normal(theta_grid):
    prior = np.exp(-0.5 * theta_grid**2)
    prior = prior / prior.sum()
    return prior


def estimar_theta_eap_individual(respostas, parametros_itens, theta_grid, prior):
    """
    Estima theta por EAP para um único aluno.

    respostas:
        vetor 0/1 com respostas corrigidas.

    parametros_itens:
        matriz com colunas a, b, c.
    """

    verossimilhanca = np.ones_like(theta_grid, dtype=float)

    for j, resposta in enumerate(respostas):
        a, b, c = parametros_itens[j]

        p = probabilidade_3pl(theta_grid, a, b, c)

        p = np.clip(p, 1e-10, 1 - 1e-10)

        if resposta == 1:
            verossimilhanca *= p
        else:
            verossimilhanca *= (1 - p)

    posterior = verossimilhanca * prior

    soma_posterior = posterior.sum()

    if soma_posterior == 0 or np.isnan(soma_posterior):
        return np.nan, np.nan

    posterior = posterior / soma_posterior

    theta_eap = np.sum(theta_grid * posterior)

    variancia = np.sum((theta_grid - theta_eap) ** 2 * posterior)
    erro_padrao = np.sqrt(variancia)

    return theta_eap, erro_padrao


def estimar_theta_eap(
    matriz_respostas,
    parametros_itens,
    theta_min=-4,
    theta_max=4,
    n_pontos=61
):
    """
    Estima theta por EAP para todos os alunos.
    """

    theta_grid = criar_grid_theta(theta_min, theta_max, n_pontos)
    prior = densidade_prior_normal(theta_grid)

    thetas = []
    erros = []

    for respostas in matriz_respostas:
        theta, erro = estimar_theta_eap_individual(
            respostas=respostas,
            parametros_itens=parametros_itens,
            theta_grid=theta_grid,
            prior=prior
        )

        thetas.append(theta)
        erros.append(erro)

    return np.array(thetas), np.array(erros)