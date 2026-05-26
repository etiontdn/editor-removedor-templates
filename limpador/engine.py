import os
import cv2
import numpy as np
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from django.conf import settings
from .models import ImageTemplate

# Ativa o suporte a OpenCL para aceleração por GPU
cv2.ocl.setUseOpenCL(True)
print(f"[DEBUG] OpenCL ativado no motor: {cv2.ocl.useOpenCL()}")

# Dicionário global para controlar o status de cada thread em andamento
PROCESSING_TASKS = {}

# Cache global para templates redimensionados e UMat na GPU (Chave: (template_name, target_width))
TEMPLATE_CACHE = {}
TEMPLATE_CACHE_LOCK = threading.Lock()

def get_cached_template(t_info, target_width):
    """
    Retorna o template redimensionado e preparado para matching (com cache thread-safe).
    """
    cache_key = (t_info['name'], target_width)
    
    with TEMPLATE_CACHE_LOCK:
        cached = TEMPLATE_CACHE.get(cache_key)
        
    if cached is None:
        proporcao = target_width / t_info['original_width'] if t_info['original_width'] > 0 else 1.0
        th, tw = int(t_info['img'].shape[0] * proporcao), int(t_info['img'].shape[1] * proporcao)
        
        if tw > 0 and th > 0:
            interp = cv2.INTER_AREA if proporcao < 1.0 else cv2.INTER_CUBIC
            temp_rescalado = cv2.resize(t_info['img'], (tw, th), interpolation=interp)
            
            # Ignora transparência e foca apenas nas cores (BGR)
            if len(temp_rescalado.shape) == 3 and temp_rescalado.shape[2] == 4:
                temp_bgr = temp_rescalado[:, :, :3].copy()
                temp_alpha_mask = temp_rescalado[:, :, 3].copy()
            else:
                temp_bgr = temp_rescalado
                temp_alpha_mask = None
            
            has_transparency = False
            if temp_alpha_mask is not None:
                if np.any(temp_alpha_mask < 255):
                    has_transparency = True
            
            # Converte e mantém na GPU (UMat)
            temp_umat = cv2.UMat(temp_bgr)
            
            cached = {
                'temp_bgr': temp_bgr,
                'temp_alpha_mask': temp_alpha_mask,
                'temp_umat': temp_umat,
                'has_transparency': has_transparency,
                'tw': tw,
                'th': th,
                'proporcao': proporcao
            }
            
            with TEMPLATE_CACHE_LOCK:
                if cache_key not in TEMPLATE_CACHE:
                    TEMPLATE_CACHE[cache_key] = cached
                else:
                    cached = TEMPLATE_CACHE[cache_key]
        else:
            return None
            
    return cached

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

def start_processing_thread(mother_path, selected_folders, process_transitions=True):
    task_id = str(uuid.uuid4())
    PROCESSING_TASKS[task_id] = {
        'status': 'in_progress',
        'progress': 0,
        'total': 0,
        'processed_count': 0,
        'current_file': '',
        'message': 'Inicializando...',
        'error': None
    }
    thread = threading.Thread(target=process_task, args=(task_id, mother_path, selected_folders, process_transitions))
    thread.daemon = True
    thread.start()
    return task_id

def process_single_file(file_path, templates_info, threshold, task_id, progress_lock, folder_name):
    try:
        PROCESSING_TASKS[task_id]['current_file'] = f"[{folder_name}] {os.path.basename(file_path)}"
        img = imread_unicode(file_path)
        if img is not None:
            _, img_final, alterou = process_single_image(img, templates_info, threshold)
            if alterou:
                print(f"[DEBUG] Imagem alterada e salva: {os.path.basename(file_path)}")
                imwrite_unicode(file_path, img_final)
        
        with progress_lock:
            PROCESSING_TASKS[task_id]['processed_count'] += 1
            total = PROCESSING_TASKS[task_id]['total']
            if total > 0:
                PROCESSING_TASKS[task_id]['progress'] = int((PROCESSING_TASKS[task_id]['processed_count'] / total) * 100)
    except Exception as e:
        print(f"[DEBUG] Erro ao processar arquivo {file_path}: {e}")


def process_task(task_id, mother_path, selected_folders, process_transitions=True):
    start_time = time.time()
    
    # Invalida o cache global no início da execução de cada lote
    global TEMPLATE_CACHE
    with TEMPLATE_CACHE_LOCK:
        TEMPLATE_CACHE.clear()
        
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
            duration = time.time() - start_time
            print(f"[DEBUG] Tarefa abortada (sem templates). Tempo decorrido: {duration:.2f} segundos.")
            return

        total_arquivos = 0
        total_transicoes = 0
        for folder in selected_folders:
            f_path = os.path.join(mother_path, folder)
            if os.path.isdir(f_path):
                files = [f for f in os.listdir(f_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                total_arquivos += len(files)
                if process_transitions:
                    total_transicoes += max(0, len(files) - 1)
                print(f"[DEBUG] Pasta '{folder}': {len(files)} arquivos encontrados.")
        
        total_operacoes = total_arquivos + total_transicoes
        print(f"[DEBUG] Total de arquivos: {total_arquivos}, Total de transições: {total_transicoes}")
        PROCESSING_TASKS[task_id]['total'] = total_operacoes
        if total_arquivos == 0:
            print("[DEBUG] Nada para processar.")
            PROCESSING_TASKS[task_id]['status'] = 'completed'
            PROCESSING_TASKS[task_id]['progress'] = 100
            duration = time.time() - start_time
            print(f"[DEBUG] Tarefa finalizada (nada para processar). Tempo decorrido: {duration:.2f} segundos.")
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
            
            progress_lock = threading.Lock()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                worker = partial(process_single_file, templates_info=templates_info, threshold=THRESHOLD, task_id=task_id, progress_lock=progress_lock, folder_name=folder_name)
                executor.map(worker, arquivos)

            # Fase 2: Transição
            if process_transitions:
                PROCESSING_TASKS[task_id]['message'] = f"Analisando transições na pasta: {folder_name}"
                for i in range(len(arquivos) - 1):
                    process_transition(arquivos[i], arquivos[i+1], templates_info, THRESHOLD)
                    
                    PROCESSING_TASKS[task_id]['processed_count'] += 1
                    total = PROCESSING_TASKS[task_id]['total']
                    if total > 0:
                        PROCESSING_TASKS[task_id]['progress'] = int((PROCESSING_TASKS[task_id]['processed_count'] / total) * 100)

        PROCESSING_TASKS[task_id]['status'] = 'completed'
        count = PROCESSING_TASKS[task_id]['processed_count']
        PROCESSING_TASKS[task_id]['message'] = f'Processamento finalizado com sucesso! {count} operações concluídas.'
        duration = time.time() - start_time
        print(f"[DEBUG] Tarefa finalizada com sucesso. Tempo decorrido: {duration:.2f} segundos.")
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[DEBUG] ERRO FATAL: {error_msg}")
        PROCESSING_TASKS[task_id]['status'] = 'error'
        PROCESSING_TASKS[task_id]['error'] = str(e)
        PROCESSING_TASKS[task_id]['message'] = 'Erro fatal no motor.'
        duration = time.time() - start_time
        print(f"[DEBUG] Tarefa interrompida com erro após {duration:.2f} segundos.")

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
        cached = get_cached_template(t_info, width_real)
        if cached is None: continue
        
        tw = cached['tw']
        th = cached['th']
        proporcao = cached['proporcao']
        temp_bgr = cached['temp_bgr']
        temp_alpha_mask = cached['temp_alpha_mask']
        temp_umat = cached['temp_umat']
        has_transparency = cached['has_transparency']

        if tw > img_trabalho.shape[1] or th > img_trabalho.shape[0]: continue

        if has_transparency:
            print(f"[DEBUG] Template '{t_info['name']}' possui transparência. Usando matching com máscara (CPU).")

        match_count = 0
        while match_count < 100:  # Limite de segurança contra loop infinito
            # Verifica se o template ainda cabe na imagem (importante após cortes)
            if tw > img_trabalho.shape[1] or th > img_trabalho.shape[0]:
                print(f"[DEBUG] Template {t_info['name']} maior que a imagem restante. Pulando.")
                break

            if has_transparency:
                # Usa TM_CCORR_NORMED que aceita máscara (geralmente CPU)
                res = cv2.matchTemplate(img_trabalho, temp_bgr, cv2.TM_CCORR_NORMED, mask=temp_alpha_mask)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
            else:
                # Usa TM_CCOEFF_NORMED com aceleração GPU (OpenCL)
                img_umat = cv2.UMat(img_trabalho)
                res_umat = cv2.matchTemplate(img_umat, temp_umat, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res_umat)
            if max_val < threshold: break
            
            match_count += 1
            
            y_topo, x_esq = max_loc[1], max_loc[0]
            padding = int(t_info['padding'] * proporcao)
            y_ini, y_fim = max(0, y_topo - padding), min(img_trabalho.shape[0], y_topo + th + padding)
            x_ini, x_fim = max(0, x_esq - padding), min(img_trabalho.shape[1], x_esq + tw + padding)
            
            if t_info['action_type'] == 'fill':
                print(f"[DEBUG] Match found! Template: {t_info['name']} (Score: {max_val:.4f}). Action: FILL em {x_ini, y_ini}")
                
                if temp_alpha_mask is not None:
                    # Aplicação mascarada para formatos variados
                    roi_trabalho = img_trabalho[y_ini:y_fim, x_ini:x_fim]
                    roi_original = img_original[y_ini:y_fim, x_ini:x_fim]
                    
                    # Redimensionar máscara se o padding alterou o tamanho esperado
                    m_h, m_w = roi_trabalho.shape[:2]
                    mask_resized = cv2.resize(temp_alpha_mask, (m_w, m_h), interpolation=cv2.INTER_AREA)
                    mask_bool = mask_resized > 10 # Limiar para a máscara
                    
                    # Criar uma cor sólida compatível com o número de canais
                    if roi_original.shape[2] == 4:
                        # Para RGBA, preenchemos BGR e definimos Alpha como 255
                        roi_original[mask_bool, :3] = t_info['fill_color']
                        roi_original[mask_bool, 3] = 255
                    else:
                        roi_original[mask_bool] = t_info['fill_color']
                    
                    # Atualiza imagem de trabalho (sempre BGR)
                    roi_trabalho[mask_bool] = t_info['fill_color']
                else:
                    # Retangular padrão
                    cv2.rectangle(img_trabalho, (x_ini, y_ini), (x_fim, y_fim), t_info['fill_color'], -1)
                    if img_original is not img_trabalho: cv2.rectangle(img_original, (x_ini, y_ini), (x_fim, y_fim), t_info['fill_color'], -1)
                
                alterou = True
            elif t_info['action_type'] == 'cut_y':
                print(f"[DEBUG] Match found! Template: {t_info['name']} (Score: {max_val:.4f}). Action: CUT_Y em y={y_ini}")
                
                # Se o template tiver máscara (alpha), usamos ela para um corte mais preciso se possível,
                # mas para CUT_Y (que remove a linha inteira) o retângulo costuma ser suficiente.
                # Se quiser um corte que respeite a forma lateral, seria 'cut_shape', mas aqui focamos no eixo Y.
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
        cached = get_cached_template(t_info, w_a)
        if cached is None: continue
        
        tw = cached['tw']
        th = cached['th']
        proporcao = cached['proporcao']
        temp_bgr = cached['temp_bgr']
        temp_alpha_mask = cached['temp_alpha_mask']
        temp_umat = cached['temp_umat']
        has_transparency = cached['has_transparency']

        # Verifica se cabe na ponte
        if tw > ponte_trabalho.shape[1] or th > ponte_trabalho.shape[0]:
            continue

        if has_transparency:
            print(f"[DEBUG] [Transição] Template '{t_info['name']}' possui transparência. Usando matching com máscara (CPU).")

        if has_transparency:
            # Usa TM_CCORR_NORMED que aceita máscara (geralmente CPU)
            res = cv2.matchTemplate(ponte_trabalho, temp_bgr, cv2.TM_CCORR_NORMED, mask=temp_alpha_mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        else:
            # Usa TM_CCOEFF_NORMED com aceleração GPU (OpenCL)
            ponte_umat = cv2.UMat(ponte_trabalho)
            res_umat = cv2.matchTemplate(ponte_umat, temp_umat, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res_umat)

        if max_val >= threshold:
            y_topo = max_loc[1]
            if y_topo < h_a and (y_topo + th) > h_a:
                padding = int(t_info['padding'] * proporcao)
                y_ini, y_fim = y_topo - padding, y_topo + th + padding
                if t_info['action_type'] == 'fill':
                    print(f"[DEBUG] Transition match! Template: {t_info['name']} (Score: {max_val:.4f}). Action: FILL transição")
                    
                    if temp_alpha_mask is not None:
                        # Aplicação mascarada na transição
                        # Cria uma máscara para a área da transição na ponte
                        y_fim = y_topo + th
                        padding = int(t_info['padding'] * proporcao)
                        y_ini_p, y_fim_p = max(0, y_topo - padding), min(ponte.shape[0], y_topo + th + padding)
                        
                        mask_resized = cv2.resize(temp_alpha_mask, (w_a, y_fim_p - y_ini_p), interpolation=cv2.INTER_AREA)
                        color_fill = np.full((y_fim_p - y_ini_p, w_a, 3), t_info['fill_color'], dtype=np.uint8)
                        mask_bool = mask_resized > 10
                        
                        # Divide a aplicação entre imagem A e B
                        if y_ini_p < h_a:
                            h_in_a = min(h_a, y_fim_p) - y_ini_p
                            roi_a = img_a[y_ini_p : y_ini_p + h_in_a, :]
                            m_a = mask_bool[0 : h_in_a, :]
                            
                            if roi_a.shape[2] == 4:
                                roi_a[m_a, :3] = t_info['fill_color']
                                roi_a[m_a, 3] = 255
                            else:
                                roi_a[m_a] = t_info['fill_color']
                            alterou_a = True
                            
                        if y_fim_p > h_a:
                            h_in_b = y_fim_p - max(h_a, y_ini_p)
                            start_in_b = max(0, y_ini_p - h_a)
                            roi_b = img_b[start_in_b : start_in_b + h_in_b, :]
                            m_b = mask_bool[mask_bool.shape[0] - h_in_b : , :]
                            
                            if roi_b.shape[2] == 4:
                                roi_b[m_b, :3] = t_info['fill_color']
                                roi_b[m_b, 3] = 255
                            else:
                                roi_b[m_b] = t_info['fill_color']
                            alterou_b = True
                    else:
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
