import json
from pathlib import Path
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PASTA = Path(__file__).resolve().parent
MODELO_BASE = "HuggingFaceTB/SmolLM2-135M"
CAMINHO_TREINADO = PASTA / "modelos" / "exp2_lr_5e-5_ep4"

st.set_page_config(page_title="CP2/CP3 - Extrator JSON com LLM Local", layout="wide")
st.title("📦 Extração Estruturada de Pedidos (Texto → JSON)")
st.caption("CP2/CP3 — Generative AI for Engineering | Fine-tuning Local em GPU")

# Barra lateral com info da GPU e parâmetros
with st.sidebar:
    st.header("⚙️ Configuração & Hardware")
    if torch.cuda.is_available():
        st.success(f"GPU Ativa: {torch.cuda.get_device_name(0)}")
        vram_alocada = torch.cuda.memory_allocated(0) / (1024**2)
        st.caption(f"CUDA {torch.version.cuda} | VRAM em uso: {vram_alocada:.1f} MB")
    else:
        st.warning("Executando em CPU")

    escolha_modelo = st.selectbox(
        "Selecione o Modelo:",
        ["Modelo Treinado (exp2_lr_5e-5_ep4)", "Modelo Treinado (exp1_lr_1e-5_ep2)", "Modelo Base (Sem Treino)"]
    )
    temperatura = st.slider("Temperatura (Criatividade):", 0.0, 1.0, 0.0, 0.1)

@st.cache_resource
def carregar(caminho):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(caminho))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(caminho)).to(device)
    model.eval()
    return model, tok, device

# Define qual caminho carregar
if "exp2" in escolha_modelo and CAMINHO_TREINADO.exists():
    caminho_atual = CAMINHO_TREINADO
elif "exp1" in escolha_modelo and (PASTA / "modelos" / "exp1_lr_1e-5_ep2").exists():
    caminho_atual = PASTA / "modelos" / "exp1_lr_1e-5_ep2"
else:
    caminho_atual = MODELO_BASE

model, tok, device = carregar(caminho_atual)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Inserir Pedido em Linguagem Natural")
    exemplo_padrao = "Quero 12 cadeiras ergonômicas, entrega na Av Paulista 1000 até sexta-feira."
    texto_usuario = st.text_area("Digite o texto livre do pedido:", value=exemplo_padrao, height=120)

    prompt_final = (
        "### Tarefa:\n"
        "Extraia os dados do pedido abaixo em formato JSON com as chaves: produto, qtd, endereco, prazo.\n"
        f"Pedido: \"{texto_usuario}\"\n"
        "### JSON:\n"
    )
    with st.expander("Ver Prompt Formatado enviado à LLM"):
        st.code(prompt_final, language="markdown")

    btn_executar = st.button("🚀 Extrair JSON na GPU", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Resposta da LLM e Validação")
    if btn_executar:
        entrada = tok(prompt_final, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            saida = model.generate(
                **entrada,
                max_new_tokens=90,
                do_sample=(temperatura > 0.0),
                temperature=temperatura if temperatura > 0.0 else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        novos = saida[0, entrada["input_ids"].shape[1]:]
        resposta_bruta = tok.decode(novos, skip_special_tokens=True).strip()
        if "}" in resposta_bruta:
            resposta_limpa = resposta_bruta[:resposta_bruta.find("}") + 1]
        else:
            resposta_limpa = resposta_bruta

        st.markdown("**Saída Bruta do Modelo:**")
        st.code(resposta_limpa, language="json")

        try:
            dados_json = json.loads(resposta_limpa)
            st.success("✅ JSON Válido e Estruturado com Sucesso!")
            st.json(dados_json)
        except Exception as e:
            st.error(f"❌ Falha ao validar JSON: {e}")