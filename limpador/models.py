from django.db import models

class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Campo JSON para armazenar os templates (coordenadas x, y, width, height) 
    # gerados pelo Fabric.js para aplicação posterior do OpenCV
    templates_data = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Image {self.id} - {self.uploaded_at}"
