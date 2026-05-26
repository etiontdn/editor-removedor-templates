import os
import sys
import django
import time
import cv2
import numpy as np

# Configura Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from limpador.models import ImageTemplate
from limpador.engine import process_single_image, hex_to_bgr, imread_unicode, imwrite_unicode

def main():
    # Carrega templates
    templates_db = ImageTemplate.objects.all()
    templates_info = []
    for t in templates_db:
        if not t.image: continue
        template_img = imread_unicode(t.image.path)
        if template_img is not None:
            templates_info.append({
                'id': t.id,
                'img': template_img,
                'original_width': t.original_width,
                'action_type': t.action_type,
                'fill_color': hex_to_bgr(t.fill_color),
                'padding': t.padding,
                'name': t.name
            })

    print(f"--- DIAGNÓSTICO DE VELOCIDADE (CLI) ---")
    print(f"Templates carregados do banco de dados: {len(templates_info)}")
    for t in templates_info:
        print(f"  - {t['name']} ({t['action_type']}) - Tamanho original: {t['img'].shape}")

    # Pede os parâmetros
    folder_path = input("\nDigite ou arraste o caminho da pasta com imagens: ").strip('"')
    if not os.path.isdir(folder_path):
        print("Pasta inválida!")
        return

    downsample = int(input("Fator de downsampling (1, 2, 4): ") or 1)

    arquivos = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    print(f"\nEncontrados {len(arquivos)} arquivos para processar.")
    print("Iniciando processamento sequencial direto (sem Django/Tkinter overhead)...")

    start_time = time.time()
    processed_count = 0
    
    for f_path in arquivos:
        t0 = time.time()
        img = imread_unicode(f_path)
        if img is not None:
            shape_orig = img.shape
            # Executa o matching
            _, img_final, alterou = process_single_image(img, templates_info, 0.92, downsample_factor=downsample)
            # Salva
            if alterou and img_final is not None:
                imwrite_unicode(f_path, img_final)
            
            processed_count += 1
            print(f"[{processed_count}/{len(arquivos)}] {os.path.basename(f_path)} ({shape_orig[1]}x{shape_orig[0]}) -> Processado em {time.time() - t0:.3f}s (Alterou: {alterou})")
        else:
            print(f"Erro ao ler imagem: {os.path.basename(f_path)}")

    total_duration = time.time() - start_time
    print(f"\n✓ Processamento concluído em {total_duration:.2f} segundos!")
    print(f"Média por imagem: {total_duration / len(arquivos):.3f}s")

if __name__ == "__main__":
    main()
