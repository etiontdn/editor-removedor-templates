from django.db import models

class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Campo JSON para armazenar os templates (coordenadas x, y, width, height) 
    # gerados pelo Fabric.js para aplicação posterior do OpenCV
    templates_data = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Image {self.id} - {self.uploaded_at}"

class ImageTemplate(models.Model):
    processed_image = models.ForeignKey(ProcessedImage, related_name='templates', null=True, blank=True, on_delete=models.SET_NULL)
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='templates/')
    action_type = models.CharField(max_length=50, default='fill')
    fill_color = models.CharField(max_length=20, default='#ffffff')
    padding = models.IntegerField(default=0)
    original_width = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} (from Image {self.processed_image.id})"
