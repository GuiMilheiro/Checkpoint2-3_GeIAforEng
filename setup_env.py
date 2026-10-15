import subprocess
import sys
import shutil

def executar(comando):
    print(f"\n[A executar] {comando}")
    subprocess.check_call(comando, shell=True)

def detectar_gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        saida = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        return saida
    except Exception:
        return None

def main():
    print("=" * 70)
    print(" Verificador de Ambiente e GPU — CP2/CP3 Generative AI")
    print("=" * 70)

    info_gpu = detectar_gpu()

    if info_gpu:
        print(f" Placa NVIDIA detetada: {info_gpu}")
        print("\nGuia de compatibilidade de placas gráficas (GPUs):")
        print("  • Ideais para treino local (>= 8 GB VRAM): RTX 3060, RTX 4060, RTX 3070, RTX 4070, RTX 3080/4080/4090")
        print("  • Suportadas com lotes reduzidos (4 GB a 6 GB VRAM): RTX 2060, RTX 3050, GTX 1660 Ti/Super")
        print("  • Legadas / Arquitetura Pascal: Séries GTX 10 (1060, 1070, 1080)")
        
        print("\nEscolha a versão do PyTorch de acordo com a sua GPU:")
        print("1) CUDA 12.1 [Recomendado] — Séries RTX 40, RTX 30, RTX 20 e drivers atualizados")
        print("2) CUDA 11.8 — Séries GTX 16, GTX 10 e sistemas com drivers mais antigos")
        print("3) Manter o PyTorch atual — Caso já tenha uma versão funcional com CUDA")
        print("4) Versão apenas CPU — Caso não pretenda utilizar aceleração por hardware")
        
        opcao = input("\nOpção desejada (predefinição 1): ").strip() or "1"
        
        if opcao == "1":
            executar(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        elif opcao == "2":
            executar(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        elif opcao == "4":
            executar(f"{sys.executable} -m pip install torch torchvision")
        else:
            print("A manter a instalação atual do PyTorch.")
    else:
        print(" Nenhuma GPU NVIDIA detetada com o utilitário 'nvidia-smi'.")
        resp = input("Deseja instalar a versão apenas para CPU? (s/n): ").strip().lower()
        if resp == "s":
            executar(f"{sys.executable} -m pip install torch torchvision")

    print("\n A instalar dependências adicionais (transformers, streamlit, accelerate)...")
    executar(f"{sys.executable} -m pip install -r requirements.txt")

    print("\n" + "=" * 70)
    print(" Ambiente configurado com sucesso!")
    print("=" * 70)

if __name__ == "__main__":
    main()