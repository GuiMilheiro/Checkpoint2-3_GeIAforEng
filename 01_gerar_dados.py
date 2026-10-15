import json
import random
from pathlib import Path

PASTA = Path(__file__).resolve().parent
PASTA_DADOS = PASTA / "dados"
PASTA_DADOS.mkdir(parents=True, exist_ok=True)

PRODUTOS = ["caixas de café", "pacotes de papel A4", "monitores 24p", "teclados mecânicos", "cadeiras ergonômicas", "mouses sem fio", "notebooks i5", "garrafas térmicas"]
ENDERECOS = ["Rua Augusta 1500", "Av Paulista 1000", "Rua da Consolação 450", "Av Brigadeiro Faria Lima 2200", "Rua Oscar Freire 800", "Av Engenheiro Luís Carlos Berrini 500", "Rua Vergueiro 3100", "Av Rebouças 1200"]
PRAZOS = ["até sexta-feira", "amanhã cedo", "até segunda", "em 2 dias úteis", "até o fim do mês", "hoje à tarde", "na próxima quarta", "até quinta-feira"]

TEMPLATES = [
    "Quero {qtd} {produto}, entrega na {endereco} {prazo}.",
    "Por favor, envie {qtd} unidades de {produto} para {endereco} ({prazo}).",
    "Pedido: {qtd}x {produto}. Local de entrega: {endereco}. Prazo: {prazo}.",
    "Preciso receber {qtd} {produto} no endereço {endereco} {prazo}."
]

def montar_prompt(texto_pedido: str) -> str:
    return (
        "### Tarefa:\n"
        "Extraia os dados do pedido abaixo em formato JSON com as chaves: produto, qtd, endereco, prazo.\n"
        f"Pedido: \"{texto_pedido}\"\n"
        "### JSON:\n"
    )

def gerar_dataset():
    rng = random.Random(42)
    combinacoes = []

    for prod in PRODUTOS:
        for end in ENDERECOS:
            for prazo in PRAZOS:
                qtd = rng.randint(1, 50)
                combinacoes.append((prod, qtd, end, prazo))

    rng.shuffle(combinacoes)

    grupos = {"treino": [], "validacao": [], "teste": []}

    # 300 combinações base -> Treino (com 2 variações de frase = 600), 40 Validação, 50 Teste
    for i, (prod, qtd, end, prazo) in enumerate(combinacoes[:390]):
        if i < 300:
            particao = "treino"
            templates_usados = TEMPLATES[:2]
        elif i < 340:
            particao = "validacao"
            templates_usados = [TEMPLATES[2]]
        else:
            particao = "teste"
            templates_usados = [TEMPLATES[3]] # Frase inédita no teste!

        for temp in templates_usados:
            texto = temp.format(qtd=qtd, produto=prod, endereco=end, prazo=prazo)
            gabarito_dict = {
                "produto": prod,
                "qtd": qtd,
                "endereco": end,
                "prazo": prazo
            }
            resposta_json = json.dumps(gabarito_dict, ensure_ascii=False)

            grupos[particao].append({
                "texto_pedido": texto,
                "prompt": montar_prompt(texto),
                "resposta": resposta_json,
                "gabarito": gabarito_dict
            })

    for nome, lista in grupos.items():
        rng.shuffle(lista)
        caminho = PASTA_DADOS / f"{nome}.jsonl"
        with open(caminho, "w", encoding="utf-8") as f:
            for item in lista:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{nome}: {len(lista)} exemplos salvos em {caminho}")

if __name__ == "__main__":
    gerar_dataset()