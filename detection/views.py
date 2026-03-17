import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .process_image import process_image


@csrf_exempt
@require_http_methods(["POST"])
def upload_and_detect(request):
    """
    POST /api/upload/
    Body: multipart/form-data with:
      - image        : image file (required)
      - canny_low    : float  (optional, default 0.05)
      - canny_high   : float  (optional, default 0.15)
      - canny_sigma  : float  (optional, default 1.4)
      - lines_thresh : int    (optional, default 80)
      - circles_p2   : int    (optional, default 30)

    Returns JSON:
      { original, edges, result, lines, circles, ellipses }
    """
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image file provided."}, status=400)

    file = request.FILES["image"]
    file_bytes = file.read()

    # Pull optional tuning params from the form
    def _float(key, default):
        try:
            return float(request.POST.get(key, default))
        except (TypeError, ValueError):
            return default

    def _int(key, default):
        try:
            return int(request.POST.get(key, default))
        except (TypeError, ValueError):
            return default

    try:
        result = process_image(
            file_bytes,
            canny_low_ratio=_float("canny_low", 0.05),
            canny_high_ratio=_float("canny_high", 0.15),
            canny_sigma=_float("canny_sigma", 1.4),
            lines_threshold=_int("lines_thresh", 80),
            circles_param2=_int("circles_p2", 30),
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=422)
    except Exception as exc:
        return JsonResponse({"error": f"Processing failed: {exc}"}, status=500)

    return JsonResponse(result)