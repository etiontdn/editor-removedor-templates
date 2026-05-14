from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ProcessedImage

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
