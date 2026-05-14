from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ProcessedImage, ImageTemplate
from .engine import scan_folders, start_processing_thread, PROCESSING_TASKS
import json
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image

def home_view(request):
    templates = ImageTemplate.objects.all().order_by('-created_at')
    return render(request, 'limpador/home.html', {'templates': templates})

def editor_view(request):
    return render(request, 'limpador/editor.html')

def processor_view(request):
    return render(request, 'limpador/processor.html')

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        # Cria um novo registro no banco de dados com a imagem
        processed_img = ProcessedImage.objects.create(image=image_file)
        
        # Retorna a URL da imagem para o frontend renderizar no Fabric.js
        return JsonResponse({
            'success': True,
            'image_id': processed_img.id,
            'image_url': processed_img.image.url
        })
    return JsonResponse({'success': False, 'error': 'Nenhuma imagem enviada.'}, status=400)

@csrf_exempt
def save_templates(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get('image_id')
            templates = data.get('templates', [])
            
            if not image_id:
                return JsonResponse({'success': False, 'error': 'ID da imagem não fornecido.'}, status=400)
                
            try:
                processed_img = ProcessedImage.objects.get(id=image_id)
            except ProcessedImage.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Imagem não encontrada.'}, status=404)
                
            processed_img.templates_data = templates
            processed_img.save()
            
            original_image = Image.open(processed_img.image.path)
            img_width, img_height = original_image.size
            
            saved_templates = []
            
            for tpl in templates:
                left = tpl.get('left', 0)
                top = tpl.get('top', 0)
                width = tpl.get('width', 0)
                height = tpl.get('height', 0)
                scaleX = tpl.get('scaleX', 1)
                scaleY = tpl.get('scaleY', 1)
                
                name = tpl.get('templateName', 'template')
                action_type = tpl.get('actionType', 'fill')
                fill_color = tpl.get('fillColor', '#ffffff')
                padding = int(tpl.get('padding', 0))
                
                actual_w = width * scaleX
                actual_h = height * scaleY
                
                box_left = max(0, int(left))
                box_top = max(0, int(top))
                box_right = min(img_width, int(left + actual_w))
                box_bottom = min(img_height, int(top + actual_h))
                
                if box_right > box_left and box_bottom > box_top:
                    cropped = original_image.crop((box_left, box_top, box_right, box_bottom)).convert("RGBA")
                    
                    mask_base64 = tpl.get('mask_base64')
                    if mask_base64:
                        import base64
                        if ',' in mask_base64:
                            mask_base64 = mask_base64.split(',')[1]
                        
                        mask_data = base64.b64decode(mask_base64)
                        mask_img = Image.open(BytesIO(mask_data)).convert("RGBA")
                        mask_img = mask_img.resize(cropped.size, Image.Resampling.LANCZOS)
                        
                        alpha_mask = mask_img.split()[3]
                        cropped.putalpha(alpha_mask)
                    
                    img_io = BytesIO()
                    cropped.save(img_io, format='PNG')
                    img_file = ContentFile(img_io.getvalue(), name=f"{name}_{processed_img.id}.png")
                    
                    new_template = ImageTemplate.objects.create(
                        processed_image=processed_img,
                        name=name,
                        image=img_file,
                        action_type=action_type,
                        fill_color=fill_color,
                        padding=padding,
                        original_width=img_width
                    )
                    
                    saved_templates.append({
                        'id': new_template.id,
                        'name': new_template.name,
                        'url': new_template.image.url
                    })
            
            if processed_img.image:
                processed_img.image.delete()
            processed_img.delete()
            
            return JsonResponse({'success': True, 'templates': saved_templates})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
            
    return JsonResponse({'success': False, 'error': 'Método não permitido.'}, status=405)

@csrf_exempt
def delete_template(request, template_id):
    if request.method == 'DELETE':
        try:
            template = ImageTemplate.objects.get(id=template_id)
            if template.image:
                template.image.delete()
            template.delete()
            return JsonResponse({'success': True})
        except ImageTemplate.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Template não encontrado.'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    return JsonResponse({'success': False, 'error': 'Método não permitido.'}, status=405)

@csrf_exempt
def scan_folders_api(request):
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            mother_path = data.get('path', '').strip()
            folders = scan_folders(mother_path)
            return JsonResponse({'success': True, 'folders': folders})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)

@csrf_exempt
def start_processing_api(request):
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            mother_path = data.get('path', '').strip()
            selected_folders = data.get('folders', [])
            
            if not mother_path or not selected_folders:
                return JsonResponse({'success': False, 'error': 'Caminho ou pastas não selecionadas.'})
                
            task_id = start_processing_thread(mother_path, selected_folders)
            return JsonResponse({'success': True, 'task_id': task_id})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)

def get_progress_api(request, task_id):
    task = PROCESSING_TASKS.get(task_id)
    if not task:
        return JsonResponse({'success': False, 'error': 'Tarefa não encontrada.'}, status=404)
    return JsonResponse({'success': True, 'task': task})

