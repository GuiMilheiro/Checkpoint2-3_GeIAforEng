# CP2/CP3 — Generative AI for Engineering

Projeto desenvolvido para a disciplina **Generative AI for Engineering** da **FIAP**.

## 👥 Integrantes do Grupo
* **Guilherme Milheiro** — RM 550295 
* **Felipe Hideki** — RM 98323
* **Jhonatan Curci** — RM 94188
* **Enzo Vasconcelos** — RM 550702
* **Ricardo Queiroz** — RM 94241

**Docente:** Prof. Dr. Francisco Elânio  
**Data:** 15/10/2026

---

## 🎯 Objetivo e Problema Abordado

* **Problema:** Extração e Estruturação de Dados em Linguagem Natural para JSON (**Texto Livre → JSON**).
* **Descrição da Tarefa:** Converter solicitações comerciais informais num esquema JSON padronizado com 4 chaves obrigatórias:
  * `produto` (string)
  * `qtd` (integer)
  * `endereco` (string)
  * `prazo` (string)
* **Critério de Avaliação:** Avaliação estrita campo a campo e taxa de acerto exato (*exact match*) do JSON sintática e semanticamente válido em dados inéditos de teste.

---

## 🧠 LLM e Arquitetura

* **Modelo Base:** `HuggingFaceTB/SmolLM2-135M` (134,5 milhões de parâmetros, 30 camadas Transformer).
* **Metodologia de Treino:** *Supervised Fine-Tuning* (SFT) local implementado em PyTorch puro.
* **Máscara de Atenção e Loss:** Utilização do marcador `-100` nos tokens do prompt para calcular a perda (*cross-entropy*) exclusivamente nos tokens da resposta estruturada (JSON + token `<eos>`), evitando o cálculo desnecessário de gradiente nas instruções de entrada.
* **Hardware Utilizado:** 100% executado localmente na GPU **NVIDIA GeForce RTX 3060 (12 GB VRAM)** com **CUDA 12.4**.

---

## 📊 Comparativo de Experiências e Hiperparâmetros

O pipeline avaliou o modelo base (*zero-shot*) e comparou duas configurações de hiperparâmetros no conjunto de teste (amostras com frases e combinações não vistas no treino):

| Métrica / Parâmetro | Modelo Base (Sem Treino) | Experiência 1 | Experiência 2 (Final) |
| :--- | :---: | :---: | :---: |
| **Taxa de Aprendizagem (LR)** | — | 1e-5 | **5e-5** |
| **Épocas** | 0 | 2 | **4** |
| **Tamanho do Lote (Batch)** | — | 4 | **4** |
| **Loss Final de Treino** | — | 0.0210 | **0.0007** |
| **Loss Final de Validação** | — | 0.0195 | **0.0257** |
| **Taxa de Acerto por Campo** | 0.0% | 76.5% | **92.0%** |
| **Acerto Exato (JSON Válido)** | 0.0% | 45.0% | **70.0%** |
| **Tempo de Treino / Avaliação** | ~25s (apenas eval) | ~95s | **180.7s** |

---

## 🖥️ Frontend Interativo (Streamlit)

Foi construída uma interface gráfica em Streamlit (`app.py`) com:
* Inserção de texto livre pelo utilizador.
* Visualização do prompt causal formatado enviado ao modelo.
* Ajuste de hiperparâmetros de inferência (*temperatura* / *sampling*).
* Monitorização de hardware em tempo real (identificação da GPU ativa e consumo de VRAM).
* Validador automático de schema e integridade JSON com apresentação estruturada.

---

## 📸 Evidências de Execução em GPU

As capturas de ecrã comprovando a execução local em GPU e o funcionamento do frontend estão armazenadas na pasta `Prints/`:
* `Prints/app_streamlit.png`: Interface Streamlit com o modelo treinado a extrair dados em GPU.
* `Prints/nvidia_smi_vscode.png`: Terminal integrado do VS Code a executar `nvidia-smi` com processo ativo na RTX 3060.

---

## 🚀 Como Reproduzir o Projeto

### 1. Clonar o Repositório
```bash
git clone https://github.com/GuiMilheiro/Checkpoint2-3_GeIAforEng.git
cd Checkpoint2-3_GeIAforEng
```

### 2. Configurar o Ambiente e GPU
O utilitário interativo deteta automaticamente a placa NVIDIA e instala a versão do PyTorch compilada com o CUDA correto:
```bash
python setup_env.py
```

### 3. Gerar o Dataset Sintético
Gera `treino.jsonl`, `validacao.jsonl` e `teste.jsonl` na pasta `dados/`:
```bash
python 01_gerar_dados.py
```

### 4. Executar o Treino e Avaliação
Avalia o modelo base, executa as experiências na GPU e guarda os relatórios em `resultados/`:
```bash
python 02_treinar_avaliar.py
```

### 5. Iniciar o Frontend
```bash
streamlit run app.py
```