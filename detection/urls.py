from django.urls import path
from .views import upload_and_detect, run_active_contour


urlpatterns = [
    # We changed 'ImageUploadView.as_view()' to 'upload_and_detect'
    path('upload/', upload_and_detect, name='upload'),
    path('snake/',run_active_contour, name='run_active_contour')
]