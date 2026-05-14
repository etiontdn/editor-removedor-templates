import os
import cv2
import numpy as np
import uuid
import threading
from django.conf import settings
from .models import ImageTemplate

# Dicionário global para controlar o status de cada thread em andamento
PROCESSING_TASKS = {}

def imread_unicode(path):
    """ Lê uma imagem de um caminho que contém acentos ou caracteres especiais. """
    try:
        with open(path, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(chunk, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[DEBUG] Erro: Falha ao decodificar imagem {path}")
            return img
    except Exception as e:
        print(f"[DEBUG] Erro ao ler arquivo {path}: {e}")
        return None

def imwrite_unicode(path, img):
    """ Salva uma imagem em um caminho que contém acentos ou caracteres especiais. """
    try:
        ext = os.path.splitext(path)[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            im_buf_arr.tofile(path)
            return True
        print(f"[DEBUG] Erro: Falha ao codificar imagem {path}")
        return False
    except Exception as e:
        print(f"[DEBUG] Erro ao salvar arquivo {path}: {e}")
        return False

def scan_folders(mother_path):
    """Rastreia pastas em busca daquelas que contêm imagens"""
    if not os.path.isdir(mother_path):
        return []

    valid_folders = []
    try:
        for entry in os.scandir(mother_path):
            if entry.is_dir():
                has_image = False
                try:
                    for file_entry in os.scandir(entry.path):
                        if file_entry.is_file() and file_entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            has_image = True
                            break
                except PermissionError:
                    pass
                if has_image:
                    valid_folders.append(entry.name)
    except Exception as e:
        print(f"Erro ao escanear pasta: {e}")
    return sorted(valid_folders)

def hex_to_bgr(hex_color):
    """Converte '#RRGGBB' para uma tupla (B, G, R) para o OpenCV"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)
    return (255, 255, 255)

def start_processing_thread(mother_path, selected_folders):
    task_id = str(uuid.uuid4())
    PROCESSING_TASKS[task_id] = {
        'status': 'in_progress',
        'progress': 0,
        'total': 0,
        'current_file': '',
        'message': 'Inicializando...',
        'error': None
    }
    thread = threading.Thread(target=process_task, args=(task_id, mother_path, selected_folders))
    thread.daemon = True
    thread.start()
    return task_id

def process_task(task_id, mother_path, selected_folders):
    try:
        templates_db = ImageTemplate.objects.all()
        templates_info = []
        print(f"[DEBUG] Buscando templates no banco... Encontrados: {len(templates_db)}")
        
        for t in templates_db:
            if not t.image: continue
            img_path = t.image.path
            template_img = imread_unicode(img_path)
            if template_img is not None:
                templates_info.append({
                    'img': template_img,
                    'original_width': t.original_width,
                    'action_type': t.action_type,
                    'fill_color': hex_to_bgr(t.fill_color),
                    'padding': t.padding,
                    'name': t.name
                })
        
        print(f"[DEBUG] Templates carregados com sucesso: {len(templates_info)}")

        if not templates_info:
            print("[DEBUG] Nenhum template válido. Abortando.")
            PROCESSING_TASKS[task_id]['status'] = 'error'
            PROCESSING_TASKS[task_id]['message'] = 'Nenhum template válido encontrado.'
            return

        total_arquivos = 0
        for folder in selected_folders:
            f_path = os.path.join(mother_path, folder)
            if os.path.isdir(f_path):
                files = [f for f in os.listdir(f_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                total_arquivos += len(files)
                print(f"[DEBUG] Pasta '{folder}': {len(files)} arquivos encontrados.")
        
        print(f"[DEBUG] Total de arquivos para processar: {total_arquivos}")
        PROCESSING_TASKS[task_id]['total'] = total_arquivos
        if total_arquivos == 0:
            print("[DEBUG] Nada para processar.")
            PROCESSING_TASKS[task_id]['status'] = 'completed'
            PROCESSING_TASKS[task_id]['progress'] = 100
            return

        THRESHOLD = 0.92  # Aumentado de 0.75 para evitar matches falsos
        MAX_MATCHES_PER_IMAGE = 50
        arquivos_processados_count = 0

        for folder_name in selected_folders:
            folder_path = os.path.join(mother_path, folder_name)
            arquivos = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

            # Fase 1: Individual
            print(f"[DEBUG] Iniciando Fase 1 (Individual) para pasta: {folder_name}")
            for file_path in arquivos:
                PROCESSING_TASKS[task_id]['current_file'] = f"[{folder_name}] {os.path.basename(file_path)}"
                img = imread_unicode(file_path)
                if img is not None:
                    _, img_final, alterou = process_single_image(img, templates_info, THRESHOLD)
                    if alterou:
                        print(f"[DEBUG] Imagem alterada e salva: {os.path.basename(file_path)}")
                        imwrite_unicode(file_path, img_final)
                
                arquivos_processados_count += 1
                PROCESSING_TASKS[task_id]['progress'] = int((arquivos_processados_count / total_arquivos) * 100)

            # Fase 2: Transição
            PROCESSING_TASKS[task_id]['message'] = f"Analisando transições na pasta: {folder_name}"
            for i in range(len(arquivos) - 1):
                process_transition(arquivos[i], arquivos[i+1], templates_info, THRESHOLD)

        PROCESSING_TASKS[task_id]['status'] = 'completed'
        PROCESSING_TASKS[task_id]['message'] = 'Processamento finalizado com sucesso!'
        print("[DEBUG] Tarefa finalizada com sucesso.")
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[DEBUG] ERRO FATAL: {error_msg}")
        PROCESSING_TASKS[task_id]['status'] = 'error'
        PROCESSING_TASKS[task_id]['error'] = str(e)
        PROCESSING_TASKS[task_id]['message'] = 'Erro fatal no motor.'

def process_single_image(img_original, templates_info, threshold):
    if len(img_original.shape) == 3 and img_original.shape[2] == 4:
        img_trabalho = img_original[:, :, :3].copy()
    elif len(img_original.shape) == 2:
        img_trabalho = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        img_original = img_trabalho.copy()
    else:
        img_trabalho = img_original.copy()
        
    width_real = img_trabalho.shape[1]
    alterou = False
    
    for t_info in sorted(templates_info, key=lambda x: 1 if x['action_type'] == 'cut_y' else 0):
        proporcao = width_real / t_info['original_width'] if t_info['original_width'] > 0 else 1.0
        th, tw = int(t_info['img'].shape[0] * proporcao), int(t_info['img'].shape[1] * proporcao)
        if tw <= 0 or th <= 0 or tw > img_trabalho.shape[1] or th > img_trabalho.shape[0]: continue
        
        temp_rescalado = cv2.resize(t_info['img'], (tw, th))
        temp_bgr, temp_alpha = (temp_rescalado[:, :, :3], temp_rescalado[:, :, 3]) if len(temp_rescalado.shape) == 3 and temp_rescalado.shape[2] == 4 else (temp_rescalado, None)
        
        match_count = 0
        while match_count < 100:  # Limite de segurança contra loop infinito
            # Verifica se o template ainda cabe na imagem (importante após cortes)
            if tw > img_trabalho.shape[1] or th > img_trabalho.shape[0]:
                print(f"[DEBUG] Template {t_info['name']} maior que a imagem restante. Pulando.")
                break

            res = cv2.matchTemplate(img_trabalho, temp_bgr, cv2.TM_CCORR_NORMED, mask=temp_alpha) if temp_alpha is not None else cv2.matchTemplate(img_trabalho, temp_bgr, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < threshold: break
            
            match_count += 1
            
            y_topo, x_esq = max_loc[1], max_loc[0]
            padding = int(t_info['padding'] * proporcao)
            y_ini, y_fim = max(0, y_topo - padding), min(img_trabalho.shape[0], y_topo + th + padding)
            x_ini, x_fim = max(0, x_esq - padding), min(img_trabalho.shape[1], x_esq + tw + padding)
            
            if t_info['action_type'] == 'fill':
                print(f"[DEBUG] Match found! Template: {t_info['name']} (Score: {max_val:.4f}). Action: FILL em {x_ini, y_ini}")
                cv2.rectangle(img_trabalho, (x_ini, y_ini), (x_fim, y_fim), t_info['fill_color'], -1)
                if img_original is not img_trabalho: cv2.rectangle(img_original, (x_ini, y_ini), (x_fim, y_fim), t_info['fill_color'], -1)
                alterou = True
            elif t_info['action_type'] == 'cut_y':
                print(f"[DEBUG] Match found! Template: {t_info['name']} (Score: {max_val:.4f}). Action: CUT_Y em y={y_ini}")
                img_trabalho = np.vstack([img_trabalho[:y_ini, :], img_trabalho[y_fim:, :]])
                img_original = np.vstack([img_original[:y_ini, :], img_original[y_fim:, :]])
                alterou = True
    return img_trabalho, img_original, alterou

def process_transition(path_a, path_b, templates_info, threshold):
    img_a, img_b = imread_unicode(path_a), imread_unicode(path_b)
    if img_a is None or img_b is None or img_a.shape[1] != img_b.shape[1]: return

    # Garantir que ambas tenham o mesmo número de canais para o vstack
    if img_a.shape[2] != img_b.shape[2]:
        if img_a.shape[2] == 4 and img_b.shape[2] == 3:
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2BGRA)
        elif img_a.shape[2] == 3 and img_b.shape[2] == 4:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2BGRA)

    h_a, w_a = img_a.shape[:2]
    ponte = np.vstack([img_a, img_b])
    ponte_trabalho = ponte[:, :, :3].copy() if len(ponte.shape) == 3 and ponte.shape[2] == 4 else ponte.copy()
    alterou_a, alterou_b = False, False

    for t_info in templates_info:
        proporcao = w_a / t_info['original_width'] if t_info['original_width'] > 0 else 1.0
        th, tw = int(t_info['img'].shape[0] * proporcao), int(t_info['img'].shape[1] * proporcao)
        temp_rescalado = cv2.resize(t_info['img'], (tw, th))
        temp_bgr, temp_alpha = (temp_rescalado[:, :, :3], temp_rescalado[:, :, 3]) if len(temp_rescalado.shape) == 3 and temp_rescalado.shape[2] == 4 else (temp_rescalado, None)
        
        # Verifica se cabe na ponte
        if tw > ponte_trabalho.shape[1] or th > ponte_trabalho.shape[0]:
            continue

        res = cv2.matchTemplate(ponte_trabalho, temp_bgr, cv2.TM_CCORR_NORMED, mask=temp_alpha) if temp_alpha is not None else cv2.matchTemplate(ponte_trabalho, temp_bgr, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            y_topo = max_loc[1]
            if y_topo < h_a and (y_topo + th) > h_a:
                padding = int(t_info['padding'] * proporcao)
                y_ini, y_fim = y_topo - padding, y_topo + th + padding
                if t_info['action_type'] == 'fill':
                    print(f"[DEBUG] Transition match! Template: {t_info['name']} (Score: {max_val:.4f}). Action: FILL transição")
                    if y_ini < h_a:
                        cv2.rectangle(img_a, (0, max(0, y_ini)), (w_a, min(h_a, y_fim)), t_info['fill_color'], -1)
                        alterou_a = True
                    if y_fim > h_a:
                        cv2.rectangle(img_b, (0, max(0, y_ini - h_a)), (w_a, min(img_b.shape[0], y_fim - h_a)), t_info['fill_color'], -1)
                        alterou_b = True
                elif t_info['action_type'] == 'cut_y':
                    print(f"[DEBUG] Transition match! Template: {t_info['name']} (Score: {max_val:.4f}). Action: CUT_Y transição")
                    if y_ini < h_a:
                        img_a = img_a[0 : max(0, y_ini), :]
                        alterou_a = True
                    if y_fim > h_a:
                        img_b = img_b[min(img_b.shape[0], int(y_fim - h_a)) : img_b.shape[0], :]
                        alterou_b = True
                break

    if alterou_a: imwrite_unicode(path_a, img_a)
    if alterou_b: imwrite_unicode(path_b, img_b)
