import json
import time
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PASTA = Path(__file__).resolve().parent
PASTA_RESULTADOS = PASTA / "resultados"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Você pode trocar por "Qwen/Qwen2.5-0.5B" ou "HuggingFaceTB/SmolLM2-360M"
MODELO_BASE = "HuggingFaceTB/SmolLM2-135M"
MAX_TOKENS = 160

def verificar_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("ATENÇÃO: GPU CUDA não detectada! O trabalho exige execução em GPU.")
    device = "cuda"
    nome_gpu = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("=" * 60)
    print(f"GPU ATIVA: {nome_gpu} ({vram_total:.1f} GB VRAM) | CUDA: {torch.version.cuda}")
    print("=" * 60)
    return device

def ler_jsonl(nome):
    caminho = PASTA / "dados" / f"{nome}.jsonl"
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]

def carregar_modelo(caminho, device):
    tok = AutoTokenizer.from_pretrained(str(caminho))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(caminho)).to(device)
    model.eval()
    return model, tok

def gerar_resposta(model, tok, device, prompt_texto, temp=0.0):
    entrada = tok(prompt_texto, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        saida = model.generate(
            **entrada,
            max_new_tokens=90,
            do_sample=(temp > 0.0),
            temperature=temp if temp > 0.0 else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    novos = saida[0, entrada["input_ids"].shape[1]:]
    texto = tok.decode(novos, skip_special_tokens=True).strip()
    # Corta na primeira linha ou fechamento de chave para evitar texto extra
    if "}" in texto:
        texto = texto[:texto.find("}") + 1]
    return texto

def avaliar_modelo(model, tok, device, nome_avaliacao):
    exemplos = ler_jsonl("teste")
    total_exemplos = len(exemplos)
    resultados = []
    acertos_totais = 0
    soma_campos = 0.0
    inicio = time.time()

    print(f"\nIniciando avaliação [{nome_avaliacao}] em {total_exemplos} exemplos de teste...", flush=True)

    for i, ex in enumerate(exemplos, start=1):
        gerado = gerar_resposta(model, tok, device, ex["prompt"])
        gabarito = ex["gabarito"]
        acertou_tudo = False
        campos_certos = 0
        erro = ""

        try:
            obj = json.loads(gerado)
            if isinstance(obj, dict):
                for chave in ["produto", "qtd", "endereco", "prazo"]:
                    if obj.get(chave) == gabarito[chave]:
                        campos_certos += 1
                acertou_tudo = (campos_certos == 4)
            else:
                erro = "JSON gerado não é um dicionário"
        except Exception as e:
            erro = f"JSON inválido: {str(e)}"

        acertos_totais += int(acertou_tudo)
        soma_campos += (campos_certos / 4.0)

        # Mostra o progresso a cada 10 exemplos (ou no último)
        if i % 10 == 0 or i == total_exemplos:
            taxa_parcial = (acertos_totais / i) * 100
            print(f"[{i:02d}/{total_exemplos}] Avaliados... Acerto parcial: {taxa_parcial:.1f}%", flush=True)

        resultados.append({
            "pedido": ex["texto_pedido"],
            "esperado": ex["resposta"],
            "gerado": gerado,
            "campos_corretos": f"{campos_certos}/4",
            "passou": acertou_tudo,
            "erro": erro
        })

    taxa_exata = acertos_totais / total_exemplos
    taxa_campos = soma_campos / total_exemplos
    tempo = time.time() - inicio

    resumo = {
        "avaliacao": nome_avaliacao,
        "acertos_exatos": f"{acertos_totais}/{total_exemplos}",
        "taxa_acerto_exato": round(taxa_exata * 100, 2),
        "taxa_acerto_por_campo": round(taxa_campos * 100, 2),
        "tempo_segundos": round(tempo, 2),
        "amostras": resultados[:5]
    }

    caminho_saida = PASTA_RESULTADOS / f"avaliacao_{nome_avaliacao}.json"
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"-> Resultado [{nome_avaliacao}]: Acerto Exato = {taxa_exata:.1%} | Acerto por Campo = {taxa_campos:.1%} ({tempo:.1f}s)\n", flush=True)
    return resumo

def treinar_experimento(nome_exp, epocas, lote, lr, device):
    print(f"\n" + "-" * 60)
    print(f"TREINANDO EXPERIMENTO: {nome_exp} | Épocas={epocas}, Lote={lote}, LR={lr}")
    print("-" * 60)

    random.seed(42)
    torch.manual_seed(42)

    model, tok = carregar_modelo(MODELO_BASE, device)
    dados_treino = ler_jsonl("treino")
    dados_val = ler_jsonl("validacao")

    def tokenizar(ex):
        ids_p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        ids_r = tok(ex["resposta"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        entrada = ids_p + ids_r
        rotulos = [-100] * len(ids_p) + ids_r
        return entrada[:MAX_TOKENS], rotulos[:MAX_TOKENS]

    treino = [tokenizar(ex) for ex in dados_treino]
    validacao = [tokenizar(ex) for ex in dados_val]

    def preparar_lote(lote_ex):
        tam = max(len(e) for e, _ in lote_ex)
        entradas, mascaras, rotulos = [], [], []
        for e, r in lote_ex:
            f = tam - len(e)
            entradas.append(e + [tok.pad_token_id] * f)
            mascaras.append([1] * len(e) + [0] * f)
            rotulos.append(r + [-100] * f)
        return {
            "input_ids": torch.tensor(entradas, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(mascaras, dtype=torch.long, device=device),
            "labels": torch.tensor(rotulos, dtype=torch.long, device=device),
        }

    def calc_val_loss():
        model.eval()
        soma, total = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(validacao), lote):
                b = preparar_lote(validacao[i:i+lote])
                loss = model(**b).loss
                qtd = (b["labels"][:, 1:] != -100).sum().item()
                soma += loss.item() * qtd
                total += qtd
        return soma / total

    model.config.use_cache = False
    otimizador = torch.optim.AdamW(model.parameters(), lr=lr)
    historico = []
    melhor_loss = float("inf")
    pasta_destino = PASTA / "modelos" / nome_exp

    for ep in range(1, epocas + 1):
        random.shuffle(treino)
        model.train()
        soma_t, total_t = 0.0, 0

        for passo, i in enumerate(range(0, len(treino), lote), start=1):
            b = preparar_lote(treino[i:i+lote])
            otimizador.zero_grad(set_to_none=True)
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            otimizador.step()

            qtd = (b["labels"][:, 1:] != -100).sum().item()
            soma_t += loss.item() * qtd
            total_t += qtd

        loss_tr = soma_t / total_t
        loss_val = calc_val_loss()
        historico.append({"epoca": ep, "loss_treino": round(loss_tr, 4), "loss_validacao": round(loss_val, 4)})
        print(f"Época {ep}/{epocas} — Loss Treino: {loss_tr:.4f} | Loss Validação: {loss_val:.4f}")

        if loss_val < melhor_loss:
            melhor_loss = loss_val
            model.config.use_cache = True
            model.save_pretrained(pasta_destino)
            tok.save_pretrained(pasta_destino)
            model.config.use_cache = False

    resumo_av = avaliar_modelo(model, tok, device, nome_exp)
    return {"experimento": nome_exp, "parametros": {"epocas": epocas, "lote": lote, "lr": lr}, "historico_loss": historico, "avaliacao": resumo_av}

if __name__ == "__main__":
    device = verificar_gpu()

    # 1. Avaliar modelo base (Antes do treino)
    model_base, tok_base = carregar_modelo(MODELO_BASE, device)
    res_base = avaliar_modelo(model_base, tok_base, device, "modelo_base")
    del model_base
    torch.cuda.empty_cache()

    # 2. Testar diferentes parâmetros (Requisito obrigatório do CP2/CP3)
    exp1 = treinar_experimento("exp1_lr_1e-5_ep2", epocas=2, lote=4, lr=1e-5, device=device)
    torch.cuda.empty_cache()

    exp2 = treinar_experimento("exp2_lr_5e-5_ep4", epocas=4, lote=4, lr=5e-5, device=device)
    torch.cuda.empty_cache()

    relatorio = {"modelo_base": MODELO_BASE, "avaliacao_base": res_base, "experimentos": [exp1, exp2]}
    with open(PASTA_RESULTADOS / "relatorio_comparativo.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print("\nRelatório completo salvo em resultados/relatorio_comparativo.json!")