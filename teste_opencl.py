import cv2
import numpy as np
import time
import os

def test_opencl_speed():
    print("=== Testando OpenCV com OpenCL ===")
    
    # 1. Verificar se o OpenCL está disponível e ativado
    cv2.ocl.setUseOpenCL(True)
    enabled = cv2.ocl.useOpenCL()
    print(f"OpenCL ativado no OpenCV: {enabled}")
    
    if not enabled:
        print("Aviso: OpenCL não pôde ser ativado. O teste continuará na CPU.")
    
    # Tenta obter o nome do dispositivo OpenCL
    try:
        device = cv2.ocl.Device.getDefault()
        print(f"Dispositivo OpenCL: {device.name()} ({device.vendor()})")
    except Exception as e:
        print(f"Não foi possível obter detalhes do dispositivo: {e}")

    # 2. Criar imagens de teste (simulando prints grandes)
    # Vamos usar uma imagem grande para o teste fazer sentido
    img_h, img_w = 2000, 1000
    template_h, template_w = 100, 300
    
    print(f"\nCriando imagens sintéticas para o teste...")
    print(f"Imagem principal: {img_w}x{img_h}")
    print(f"Template: {template_w}x{template_h}")
    
    # Imagem base aleatória (para simular conteúdo)
    np.random.seed(42)
    img = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
    
    # Cria um template específico e insere na imagem
    template = np.zeros((template_h, template_w, 3), dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (template_w-10, template_h-10), (0, 255, 0), -1)
    cv2.putText(template, "TESTE", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Inserir o template na imagem em uma posição conhecida
    pos_y, pos_x = 1500, 500
    img[pos_y:pos_y+template_h, pos_x:pos_x+template_w] = template

    # --- TESTE 1: CPU (NumPy Arrays) ---
    print("\n--- Executando na CPU ---")
    
    # Forçar uso de CPU desativando OpenCL temporariamente
    cv2.ocl.setUseOpenCL(False)
    
    start_time = time.time()
    for i in range(5): # Executa 5 vezes para média
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
    cpu_time = (time.time() - start_time) / 5
    
    print(f"Resultado CPU: Score={max_val:.4f}, Loc={max_loc}")
    print(f"Tempo médio na CPU: {cpu_time:.4f} segundos")

    # --- TESTE 2: GPU (UMat / OpenCL) ---
    print("\n--- Executando na GPU (OpenCL) ---")
    
    cv2.ocl.setUseOpenCL(True)
    
    # Converter para UMat
    img_umat = cv2.UMat(img)
    template_umat = cv2.UMat(template)
    
    # Warm-up (a primeira execução do OpenCL costuma compilar o kernel e demorar mais)
    print("Fazendo warm-up do OpenCL...")
    res_umat = cv2.matchTemplate(img_umat, template_umat, cv2.TM_CCOEFF_NORMED)
    res_umat.get() # Força a sincronização
    
    start_time = time.time()
    for i in range(5):
        res_umat = cv2.matchTemplate(img_umat, template_umat, cv2.TM_CCOEFF_NORMED)
        # Para medir o tempo real, precisamos forçar o OpenCV a trazer o dado de volta
        # ou garantir que a operação assíncrona terminou. .get() faz isso.
        resultado_final = res_umat.get()
        _, max_val, _, max_loc = cv2.minMaxLoc(resultado_final)
        
    gpu_time = (time.time() - start_time) / 5
    
    print(f"Resultado GPU: Score={max_val:.4f}, Loc={max_loc}")
    print(f"Tempo médio na GPU: {gpu_time:.4f} segundos")
    
    # 3. Conclusão
    if gpu_time < cpu_time:
        print(f"\nSucesso! GPU foi {cpu_time/gpu_time:.2f}x mais rápida que a CPU.")
    else:
        print(f"\nCPU foi mais rápida (ou empate). Diferença: {gpu_time/cpu_time:.2f}x.")
        print("Isso pode acontecer devido ao custo de transferência de memória ou tamanho da imagem.")

if __name__ == "__main__":
    test_opencl_speed()
