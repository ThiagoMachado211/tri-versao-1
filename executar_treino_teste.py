import pandas as pd

from codigo_fonte.configuracoes import (
    ARQUIVO_RESPOSTAS,
    ARQUIVO_GABARITO,
    ARQUIVO_NOTAS,
    COLUNA_ID,
    COLUNA_NOTA_REAL,
    COLUNAS_ITENS,
    TAMANHO_TESTE,
    SEED,
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
    filtrar_alunos_minimo_respostas,
)

from codigo_fonte.divisao_amostral import dividir_treino_teste

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

    print("Filtrando alunos com poucas respostas válidas...")
    antes = len(df_respostas)
    df_respostas = filtrar_alunos_minimo_respostas(
        df_respostas=df_respostas,
        colunas_itens=COLUNAS_ITENS,
        minimo_respostas=20
    )
    depois = len(df_respostas)
    print(f"Alunos removidos: {antes - depois}")
    print(f"Alunos restantes: {depois}")

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

    df_base[COLUNA_NOTA_REAL] = (
        df_base[COLUNA_NOTA_REAL]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    antes = len(df_base)
    df_base = df_base.dropna(subset=[COLUNA_NOTA_REAL]).reset_index(drop=True)
    depois = len(df_base)
    print(f"Alunos removidos por nota ausente: {antes - depois}")


    print("Dividindo em treino e teste...")

    df_treino, df_teste = dividir_treino_teste(
        df=df_base,
        tamanho_teste=TAMANHO_TESTE,
        seed=SEED,
    )

    print(f"Treino: {len(df_treino)} alunos")
    print(f"Teste: {len(df_teste)} alunos")

    print("Gerando matrizes de respostas...")

    matriz_treino = gerar_matriz_respostas(df_treino, COLUNAS_ITENS)
    matriz_teste = gerar_matriz_respostas(df_teste, COLUNAS_ITENS)

    print("Calibrando parâmetros 3PL no conjunto de treino...")

    df_parametros = calibrar_3pl(
        matriz_respostas=matriz_treino,
        nomes_itens=COLUNAS_ITENS,
    )

    caminho_parametros = PASTA_PARAMETROS / "parametros_3pl_treino.csv"
    salvar_csv(df_parametros, caminho_parametros)

    print(f"Parâmetros salvos em: {caminho_parametros}")

    parametros_matriz = parametros_para_matriz(df_parametros)

    print("Estimando theta EAP no treino...")

    theta_treino, erro_theta_treino = estimar_theta_eap(
        matriz_respostas=matriz_treino,
        parametros_itens=parametros_matriz,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_QUADRATURA,
    )

    print("Ajustando A e B usando notas reais do treino...")

    parametros_ab = ajustar_ab_por_regressao(
        theta=theta_treino,
        nota_real=df_treino[COLUNA_NOTA_REAL].values,
    )

    caminho_ab = PASTA_PARAMETROS / "parametros_AB_treino.csv"
    salvar_parametros_ab(parametros_ab, caminho_ab)

    print(f"A e B salvos em: {caminho_ab}")
    print(f"A = {parametros_ab['A']:.4f}")
    print(f"B = {parametros_ab['B']:.4f}")

    nota_estimada_treino = aplicar_ab(
        theta=theta_treino,
        A=parametros_ab["A"],
        B=parametros_ab["B"],
    )

    print("Montando validação do treino...")

    df_validacao_treino = montar_base_validacao(
        df=df_treino,
        coluna_id=COLUNA_ID,
        coluna_nota_real=COLUNA_NOTA_REAL,
        theta=theta_treino,
        erro_theta=erro_theta_treino,
        nota_estimada=nota_estimada_treino,
    )

    metricas_treino = calcular_metricas(
        nota_real=df_validacao_treino[COLUNA_NOTA_REAL],
        nota_estimada=df_validacao_treino["nota_estimada"],
    )

    salvar_csv(
        df_validacao_treino,
        PASTA_PROFICIENCIAS / "proficiencias_treino.csv",
    )

    salvar_metricas(
        metricas_treino,
        PASTA_METRICAS / "metricas_treino.csv",
    )

    print("Estimando theta EAP no teste com parâmetros fixos...")

    theta_teste, erro_theta_teste = estimar_theta_eap(
        matriz_respostas=matriz_teste,
        parametros_itens=parametros_matriz,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_QUADRATURA,
    )

    nota_estimada_teste = aplicar_ab(
        theta=theta_teste,
        A=parametros_ab["A"],
        B=parametros_ab["B"],
    )

    print("Montando validação do teste...")

    df_validacao_teste = montar_base_validacao(
        df=df_teste,
        coluna_id=COLUNA_ID,
        coluna_nota_real=COLUNA_NOTA_REAL,
        theta=theta_teste,
        erro_theta=erro_theta_teste,
        nota_estimada=nota_estimada_teste,
    )

    metricas_teste = calcular_metricas(
        nota_real=df_validacao_teste[COLUNA_NOTA_REAL],
        nota_estimada=df_validacao_teste["nota_estimada"],
    )

    salvar_csv(
        df_validacao_teste,
        PASTA_PROFICIENCIAS / "proficiencias_teste.csv",
    )

    salvar_metricas(
        metricas_teste,
        PASTA_METRICAS / "metricas_teste.csv",
    )

    print("Calculando erro por faixa no teste...")

    df_faixas_teste = erro_por_faixa(
        df_validacao=df_validacao_teste,
        coluna_nota_real=COLUNA_NOTA_REAL,
        tamanho_faixa=50,
    )

    salvar_csv(
        df_faixas_teste,
        PASTA_METRICAS / "erro_por_faixa_teste.csv",
    )

    print("Gerando gráficos do teste...")

    grafico_dispersao_real_vs_estimado(
        df_validacao=df_validacao_teste,
        coluna_nota_real=COLUNA_NOTA_REAL,
        caminho_saida=PASTA_GRAFICOS / "dispersao_real_vs_estimado_teste.png",
    )

    histograma_erros(
        df_validacao=df_validacao_teste,
        caminho_saida=PASTA_GRAFICOS / "histograma_erros_teste.png",
    )

    histograma_erros_absolutos(
        df_validacao=df_validacao_teste,
        caminho_saida=PASTA_GRAFICOS / "histograma_erros_absolutos_teste.png",
    )

    grafico_erro_por_faixa(
        df_faixas=df_faixas_teste,
        caminho_saida=PASTA_GRAFICOS / "erro_por_faixa_teste.png",
    )

    histograma_theta(
        df_validacao=df_validacao_teste,
        caminho_saida=PASTA_GRAFICOS / "histograma_theta_teste.png",
    )

    print("\nMétricas do treino:")
    print(pd.DataFrame([metricas_treino]))

    print("\nMétricas do teste:")
    print(pd.DataFrame([metricas_teste]))

    print("\nProcesso treino/teste concluído com sucesso!")


if __name__ == "__main__":
    main()