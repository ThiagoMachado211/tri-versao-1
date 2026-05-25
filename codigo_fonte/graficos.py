import matplotlib.pyplot as plt
import pandas as pd


def grafico_dispersao_real_vs_estimado(
    df_validacao: pd.DataFrame,
    coluna_nota_real: str,
    caminho_saida
) -> None:
    """
    Gera gráfico de dispersão:
    nota real vs nota estimada.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df_validacao[coluna_nota_real],
        df_validacao["nota_estimada"],
        alpha=0.5
    )

    minimo = min(
        df_validacao[coluna_nota_real].min(),
        df_validacao["nota_estimada"].min()
    )

    maximo = max(
        df_validacao[coluna_nota_real].max(),
        df_validacao["nota_estimada"].max()
    )

    plt.plot([minimo, maximo], [minimo, maximo], linestyle="--")

    plt.xlabel("Nota real")
    plt.ylabel("Nota estimada")
    plt.title("Nota real vs nota estimada")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def histograma_erros(
    df_validacao: pd.DataFrame,
    caminho_saida,
    coluna_erro: str = "erro"
) -> None:
    """
    Gera histograma dos erros:
    erro = nota_estimada - nota_real.
    """

    plt.figure(figsize=(8, 6))

    plt.hist(
        df_validacao[coluna_erro],
        bins=30,
        edgecolor="black"
    )

    plt.axvline(0, linestyle="--")

    plt.xlabel("Erro")
    plt.ylabel("Frequência")
    plt.title("Histograma dos erros")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def histograma_erros_absolutos(
    df_validacao: pd.DataFrame,
    caminho_saida,
    coluna_erro_abs: str = "erro_absoluto"
) -> None:
    """
    Gera histograma dos erros absolutos.
    """

    plt.figure(figsize=(8, 6))

    plt.hist(
        df_validacao[coluna_erro_abs],
        bins=30,
        edgecolor="black"
    )

    plt.xlabel("Erro absoluto")
    plt.ylabel("Frequência")
    plt.title("Histograma dos erros absolutos")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def grafico_erro_por_faixa(
    df_faixas: pd.DataFrame,
    caminho_saida
) -> None:
    """
    Gera gráfico do erro absoluto médio por faixa de nota.
    """

    plt.figure(figsize=(10, 6))

    df_faixas = df_faixas.copy()
    df_faixas["faixa_nota"] = df_faixas["faixa_nota"].astype(str)

    plt.bar(
        df_faixas["faixa_nota"],
        df_faixas["erro_absoluto_medio"]
    )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Faixa de nota real")
    plt.ylabel("Erro absoluto médio")
    plt.title("Erro absoluto médio por faixa de nota")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def histograma_theta(
    df_validacao: pd.DataFrame,
    caminho_saida,
    coluna_theta: str = "theta_eap"
) -> None:
    """
    Gera histograma das proficiências theta.
    """

    plt.figure(figsize=(8, 6))

    plt.hist(
        df_validacao[coluna_theta],
        bins=30,
        edgecolor="black"
    )

    plt.xlabel("Theta estimado")
    plt.ylabel("Frequência")
    plt.title("Distribuição das proficiências estimadas")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()