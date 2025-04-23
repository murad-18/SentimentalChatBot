from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat_view'),  # main chat interface
    # path('api/chatbot/', views.chatbot_api, name='chatbot_api'),  # chatbot API endpoint
    # path('api/predict/convo-style/', views.predict_convo_style_api, name='predict_convo_style_api'),
    # path("status/", views.model_status_view, name="model_status"),
    path("ajax/chat/", views.ajax_chat, name="ajax_chat"),
    path("ajax/predict/", views.ajax_chat),
    
]
