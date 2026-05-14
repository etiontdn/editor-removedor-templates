import os
import cv2
import numpy as np
import uuid
import threading
from django.conf import settings
from .models import ImageTemplate

# Dicionário global para controlar o status de cada thread em andamento
# Em uma aplicação de produção real, usaríamos Celery e Redis
PROCESSING_TASKS = {}

def imread_unicode(path):
    """ Lê uma imagem de um caminho que contém acentos ou caracteres especiais. """
    try:
        with open(path, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(chunk, cv2.IMREAD_UNCHANGED) # Mantém todos os canais
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {e}")
        return None

def imwrite_unicode(path, img):
    """ Salva uma imagem em um caminho que contém acentos ou caracteres especiais. """
    try:
        ext = os.path.splitext(path)[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            im_buf_arr.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"Erro ao salvar arquivo {path}: {e}")
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
    return (255, 255, 255) # default white

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
    
    # Inicia a thread
    thread = threading.Thread(target=process_task, args=(task_id, mother_path, selected_folders))
    thread.daemon = True
    thread.start()
    
    return task_id

def process_task(task_id, mother_path, selected_folders):
    try:
        # Prepara templates
        templates_db = ImageTemplate.objects.all()
        templates_info = []
        for t in templates_db:
            if not t.image:
                continue
                
            img_path = t.image.path
            template_img = imread_unicode(img_path)
            
            if template_img is not None:
                templates_info.append({
                    'img': template_img, # Imagem completa (provavelmente 4 canais)
                    'original_width': t.original_width,
                    'action_type': t.action_type,
                    'fill_color': hex_to_bgr(t.fill_color),
                    'padding': t.padding,
                    'name': t.name
                })
        
        if not templates_info:
            PROCESSING_TASKS[task_id]['status'] = 'error'
            PROCESSING_TASKS[task_id]['message'] = 'Nenhum template válido encontrado no banco de dados.'
            return

        # Coletar arquivos
        arquivos = []
        for folder in selected_folders:
            folder_path = os.path.join(mother_path, folder)
            if not os.path.isdir(folder_path):
                continue
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    arquivos.append(os.path.join(folder_path, f))
                    
        total_arquivos = len(arquivos)
        PROCESSING_TASKS[task_id]['total'] = total_arquivos
        
        if total_arquivos == 0:
            PROCESSING_TASKS[task_id]['status'] = 'completed'
            PROCESSING_TASKS[task_id]['progress'] = 100
            PROCESSING_TASKS[task_id]['message'] = 'Nenhuma imagem encontrada nas pastas selecionadas.'
            return
            
        THRESHOLD = 0.75 # Limiar de similaridade
        
        for idx, file_path in enumerate(arquivos):
            PROCESSING_TASKS[task_id]['current_file'] = os.path.basename(file_path)
            
            img_original = imread_unicode(file_path)
            if img_original is None:
                continue
                
            # Verifica e padroniza para 3 canais de trabalho
            if len(img_original.shape) == 3 and img_original.shape[2] == 4:
                # Remove o canal alpha se tiver (geralmente não queremos o alpha no match principal do alvo)
                img_trabalho = img_original[:, :, :3].copy()
            elif len(img_original.shape) == 2:
                # Imagem em tons de cinza
                img_trabalho = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
                img_original = img_trabalho.copy()
            else:
                img_trabalho = img_original.copy()
                
            width_real = img_trabalho.shape[1]
            
            alterou = False
            
            # Vamos ordenar para processar preenchimentos e, só no final, os cortes 
            # (cortes modificam a estrutura da imagem, afetando as coordenadas)
            for t_info in sorted(templates_info, key=lambda x: 1 if x['action_type'] == 'cut_y' else 0):
                # Calcular proporção baseada na largura original do template
                if t_info['original_width'] > 0:
                    proporcao = width_real / t_info['original_width']
                else:
                    proporcao = 1.0 # Fallback caso falte o dado original
                    
                template_h, template_w = t_info['img'].shape[:2]
                
                th = int(template_h * proporcao)
                tw = int(template_w * proporcao)
                
                # Previne erro se proporção deixar muito pequeno
                if tw <= 0 or th <= 0 or tw > img_trabalho.shape[1] or th > img_trabalho.shape[0]:
                    continue
                    
                # Redimensionar template
                temp_rescalado = cv2.resize(t_info['img'], (tw, th))
                
                # Extrair canais
                if len(temp_rescalado.shape) == 3 and temp_rescalado.shape[2] == 4:
                    temp_bgr = temp_rescalado[:, :, :3]
                    temp_alpha = temp_rescalado[:, :, 3]
                else:
                    temp_bgr = temp_rescalado
                    temp_alpha = None
                    
                # Template matching
                while True:
                    if temp_alpha is not None:
                        # TM_CCORR_NORMED suporta máscara
                        res = cv2.matchTemplate(img_trabalho, temp_bgr, cv2.TM_CCORR_NORMED, mask=temp_alpha)
                    else:
                        res = cv2.matchTemplate(img_trabalho, temp_bgr, cv2.TM_CCOEFF_NORMED)
                        
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    if max_val < THRESHOLD:
                        break # Nenhum outro template encontrado
                        
                    # Achou um template. As coordenadas são x,y no canto superior esquerdo
                    y_topo = max_loc[1]
                    x_esq = max_loc[0]
                    
                    padding = int(t_info['padding'] * proporcao)
                    
                    # Define a área afetada considerando padding
                    y_ini = max(0, y_topo - padding)
                    y_fim = min(img_trabalho.shape[0], y_topo + th + padding)
                    x_ini = max(0, x_esq - padding)
                    x_fim = min(img_trabalho.shape[1], x_esq + tw + padding)
                    
                    if t_info['action_type'] == 'fill':
                        # Preenchimento: substitui a região pela cor definida e obscurece o local na imagem de busca para não rodar infinito
                        cv2.rectangle(img_trabalho, (x_ini, y_ini), (x_fim, y_fim), t_info['fill_color'], -1)
                        if img_original is not img_trabalho:
                            cv2.rectangle(img_original, (x_ini, y_ini), (x_fim, y_fim), t_info['fill_color'], -1)
                        alterou = True
                        
                    elif t_info['action_type'] == 'cut_y':
                        # Para não ferrar o loop ao alterar a altura da imagem e os y das próximas buscas,
                        # em caso de corte no Y fazemos a junção logo na original e na de trabalho
                        if y_ini > 0 and y_fim < img_trabalho.shape[0]:
                            img_trabalho = np.vstack([img_trabalho[:y_ini, :], img_trabalho[y_fim:, :]])
                            if img_original is not img_trabalho: # Mantém sincrono
                                # Aqui assumimos que se cortou na trabalho, tem que cortar na original
                                img_original = np.vstack([img_original[:y_ini, :], img_original[y_fim:, :]])
                        elif y_ini == 0:
                            img_trabalho = img_trabalho[y_fim:, :]
                            img_original = img_original[y_fim:, :]
                        else:
                            img_trabalho = img_trabalho[:y_ini, :]
                            img_original = img_original[:y_ini, :]
                        alterou = True
                        
            if alterou:
                imwrite_unicode(file_path, img_original)
                
            # Atualizar progresso
            PROCESSING_TASKS[task_id]['progress'] = int(((idx + 1) / total_arquivos) * 100)
            
        PROCESSING_TASKS[task_id]['status'] = 'completed'
        PROCESSING_TASKS[task_id]['message'] = 'Processamento finalizado com sucesso!'
        
    except Exception as e:
        PROCESSING_TASKS[task_id]['status'] = 'error'
        PROCESSING_TASKS[task_id]['error'] = str(e)
        PROCESSING_TASKS[task_id]['message'] = 'Ocorreu um erro no processamento.'
        print(f"Erro no process_task: {e}")
