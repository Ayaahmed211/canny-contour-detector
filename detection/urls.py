from django.urls import path
from .views import upload_and_detect

urlpatterns = [
    # We changed 'ImageUploadView.as_view()' to 'upload_and_detect'
    path('upload/', upload_and_detect, name='upload'),
]