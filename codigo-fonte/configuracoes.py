from pathlib import Path


# Configuração geral:
AREA = "CH"
N_ITENS = 45
SEED = 42
TAMANHO_TESTE = 0.20



# Pastas do projeto:
PASTA_RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = PASTA_RAIZ / "dados"
PASTA_BRUTOS = PASTA_DADOS / "brutos"
PASTA_PROCESSADOS = PASTA_DADOS / "processados"
PASTA_RESULTADOS = PASTA_RAIZ / "resultados"
PASTA_PARAMETROS = PASTA_RESULTADOS / "parametros_itens"
PASTA_PROFICIENCIAS = PASTA_RESULTADOS / "proficiencias"
PASTA_METRICAS = PASTA_RESULTADOS / "metricas"
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"



# Arquivos de entrada:
ARQUIVO_RESPOSTAS = PASTA_BRUTOS / f"RESPOSTAS_{AREA}.csv"
ARQUIVO_GABARITO = PASTA_BRUTOS / f"GABARITO_{AREA}.csv"
ARQUIVO_NOTAS = PASTA_BRUTOS / f"NOTAS_{AREA}.csv"



# Colunas:
COLUNA_ID = "insc"
COLUNA_NOTA_REAL = "nota_tri"
COLUNAS_ITENS = [f"Q{i}" for i in range(1, N_ITENS + 1)]



# Respostas:
ALTERNATIVAS_VALIDAS = ["a", "b", "c", "d", "e"]
VALORES_AUSENTES = ["P", "R", "", " ", "NA", "NaN", None]




# Configurações da TRI:
THETA_MIN = -4
THETA_MAX = 4
N_PONTOS_QUADRATURA = 41



# Escala da nota:
MEDIA_ALVO = 500
DP_ALVO = 100