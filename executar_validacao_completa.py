import pandas as pd

from codigo_fonte.configuracoes import (
    ARQUIVO_RESPOSTAS,
    ARQUIVO_GABARITO,
    ARQUIVO_NOTAS,
    COLUNA_ID,
    COLUNA_NOTA_REAL,
    COLUNAS_ITENS,
    THETA_MIN,
    THETA_MAX,
    N_PONTOS_QUADRATURA,
    PASTA_PARAMETROS,
    PASTA_PROFICIENCIAS,
    PASTA_METRICAS,
    PASTA_GRAFICOS,
)

from codigo_fonte.leitura_escrita import (
    ler_respostas,
    ler_gabarito,
    ler_notas,
    salvar_csv,
)

from codigo_fonte.pre_processamento import (
    extrair_gabarito,
    corrigir_respostas,
    gerar_matriz_respostas,
)

from codigo_fonte.modelo_tri_3pl import (
    calibrar_3pl,
    parametros_para_matriz,
)

from codigo_fonte.estimacao_proficiencia import estimar_theta_eap

from codigo_fonte.escala_ab import (
    ajustar_ab_por_regressao,
    aplicar_ab,
    salvar_parametros_ab,
)

from codigo_fonte.validacao import (
    montar_base_validacao,
    calcular_metricas,
    erro_por_faixa,
    salvar_metricas,
)

from codigo_fonte.graficos import (
    grafico_dispersao_real_vs_estimado,
    histograma_erros,
    histograma_erros_absolutos,
    grafico_erro_por_faixa,
    histograma_theta,
)


def main():
    print("Lendo arquivos...")

    df_respostas = ler_respostas(ARQUIVO_RESPOSTAS)
    df_gabarito = ler_gabarito(ARQUIVO_GABARITO)
    df_notas = ler_notas(ARQUIVO_NOTAS)

    print("Extraindo gabarito...")

    gabarito = extrair_gabarito(df_gabarito, COLUNAS_ITENS)

    print("Corrigindo respostas...")

    df_corrigido = corrigir_respostas(
        df_respostas=df_respostas,
        gabarito=gabarito,
        colunas_itens=COLUNAS_ITENS,
        coluna_id=COLUNA_ID,
    )

    print("Unindo respostas corrigidas com notas reais...")

    df_base = df_corrigido.merge(
        df_notas[[COLUNA_ID, COLUNA_NOTA_REAL]],
        on=COLUNA_ID,
        how="inner",
    )

    df_base[COLUNA_NOTA_REAL] = df_base[COLUNA_NOTA_REAL].astype(float)

    print(f"Total de alunos usados: {len(df_base)}")

    print("Gerando matriz de respostas...")

    matriz_respostas = gerar_matriz_respostas(df_base, COLUNAS_ITENS)

    print("Calibrando parâmetros 3PL usando toda a base...")

    df_parametros = calibrar_3pl(
        matriz_respostas=matriz_respostas,
        nomes_itens=COLUNAS_ITENS,
    )

    caminho_parametros = PASTA_PARAMETROS / "parametros_3pl_completo.csv"
    salvar_csv(df_parametros, caminho_parametros)

    print(f"Parâmetros salvos em: {caminho_parametros}")

    parametros_matriz = parametros_para_matriz(df_parametros)

    print("Estimando theta EAP para toda a base...")

    theta, erro_theta = estimar_theta_eap(
        matriz_respostas=matriz_respostas,
        parametros_itens=parametros_matriz,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_QUADRATURA,
    )

    print("Ajustando A e B usando todas as notas reais...")

    parametros_ab = ajustar_ab_por_regressao(
        theta=theta,
        nota_real=df_base[COLUNA_NOTA_REAL].values,
    )

    caminho_ab = PASTA_PARAMETROS / "parametros_AB_completo.csv"
    salvar_parametros_ab(parametros_ab, caminho_ab)

    print(f"A e B salvos em: {caminho_ab}")
    print(f"A = {parametros_ab['A']:.4f}")
    print(f"B = {parametros_ab['B']:.4f}")

    nota_estimada = aplicar_ab(
        theta=theta,
        A=parametros_ab["A"],
        B=parametros_ab["B"],
    )

    print("Montando base de validação completa...")

    df_validacao = montar_base_validacao(
        df=df_base,
        coluna_id=COLUNA_ID,
        coluna_nota_real=COLUNA_NOTA_REAL,
        theta=theta,
        erro_theta=erro_theta,
        nota_estimada=nota_estimada,
    )

    metricas = calcular_metricas(
        nota_real=df_validacao[COLUNA_NOTA_REAL],
        nota_estimada=df_validacao["nota_estimada"],
    )

    salvar_csv(
        df_validacao,
        PASTA_PROFICIENCIAS / "proficiencias_completo.csv",
    )

    salvar_metricas(
        metricas,
        PASTA_METRICAS / "metricas_completo.csv",
    )

    print("Calculando erro por faixa...")

    df_faixas = erro_por_faixa(
        df_validacao=df_validacao,
        coluna_nota_real=COLUNA_NOTA_REAL,
        tamanho_faixa=50,
    )

    salvar_csv(
        df_faixas,
        PASTA_METRICAS / "erro_por_faixa_completo.csv",
    )

    print("Gerando gráficos...")

    grafico_dispersao_real_vs_estimado(
        df_validacao=df_validacao,
        coluna_nota_real=COLUNA_NOTA_REAL,
        caminho_saida=PASTA_GRAFICOS / "dispersao_real_vs_estimado_completo.png",
    )

    histograma_erros(
        df_validacao=df_validacao,
        caminho_saida=PASTA_GRAFICOS / "histograma_erros_completo.png",
    )

    histograma_erros_absolutos(
        df_validacao=df_validacao,
        caminho_saida=PASTA_GRAFICOS / "histograma_erros_absolutos_completo.png",
    )

    grafico_erro_por_faixa(
        df_faixas=df_faixas,
        caminho_saida=PASTA_GRAFICOS / "erro_por_faixa_completo.png",
    )

    histograma_theta(
        df_validacao=df_validacao,
        caminho_saida=PASTA_GRAFICOS / "histograma_theta_completo.png",
    )

    print("\nMétricas da validação completa:")
    print(pd.DataFrame([metricas]))

    print("\nValidação completa concluída com sucesso!")


if __name__ == "__main__":
    main()