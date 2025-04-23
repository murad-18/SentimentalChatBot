# main/apps.py

from django.apps import AppConfig
from django.conf import settings
# from . import models_loader


class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        from . import model_loader
        print("🚀 Starting async model loading...")
        model_loader.start_model_loading_async()
