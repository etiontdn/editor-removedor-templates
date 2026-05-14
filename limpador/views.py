from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ProcessedImage, ImageTemplate
import json
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image

def editor_view(request):
    return render(request, 'limpador/editor.html')

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
                
                box_left = max(0, int(left - padding))
                box_top = max(0, int(top - padding))
                box_right = min(img_width, int(left + actual_w + padding))
                box_bottom = min(img_height, int(top + actual_h + padding))
                
                if box_right > box_left and box_bottom > box_top:
                    cropped = original_image.crop((box_left, box_top, box_right, box_bottom))
                    
                    img_io = BytesIO()
                    cropped.save(img_io, format='PNG')
                    img_file = ContentFile(img_io.getvalue(), name=f"{name}_{processed_img.id}.png")
                    
                    new_template = ImageTemplate.objects.create(
                        processed_image=processed_img,
                        name=name,
                        image=img_file,
                        action_type=action_type,
                        fill_color=fill_color,
                        padding=padding
                    )
                    
                    saved_templates.append({
                        'id': new_template.id,
                        'name': new_template.name,
                        'url': new_template.image.url
                    })
            
            return JsonResponse({'success': True, 'templates': saved_templates})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
            
    return JsonResponse({'success': False, 'error': 'Método não permitido.'}, status=405)
