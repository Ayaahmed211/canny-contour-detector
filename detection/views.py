import json
import base64
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .snake_contour import GreedySnake 
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




@csrf_exempt # Disables CSRF for this API endpoint so React can easily POST to it
@require_http_methods(["POST"])
def run_active_contour(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # 1. Extract parameters from React
            image_data_uri = data.get('image')
            initial_contour = data.get('initial_contour')
            alpha = float(data.get('alpha', 0.3))
            beta = float(data.get('beta', 0.5))
            gamma = float(data.get('gamma', 1.5))
            
            # --- THE FIX: Extract the new parameters here ---
            max_iterations = int(data.get('max_iterations', 200))
            convergence_threshold = float(data.get('convergence_threshold', 0.5))

            if not image_data_uri or not initial_contour:
                return JsonResponse({'error': 'Missing image or contour points'}, status=400)

            # 2. Decode the Base64 image into an OpenCV numpy array
            header, encoded = image_data_uri.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 3. Initialize and run the algorithm with ALL parameters
            snake = GreedySnake(
                image=img, 
                initial_contour=initial_contour, 
                alpha=alpha, 
                beta=beta, 
                gamma=gamma,
                max_iterations=max_iterations,             # Passed here
                convergence_threshold=convergence_threshold # Passed here
            )
            snake.evolve()

            # 4. Gather the results and convert NumPy types to Python primitives
            perimeter = float(snake.compute_perimeter())
            area = float(snake.compute_area())

            # Ensure everything in the list is a standard Python int
            chain_code = [int(x) for x in snake.get_chain_code()]
            
            # get_visualization returns raw base64, we need to re-wrap it for the browser
            vis_raw_base64 = snake.get_visualization()
            vis_data_uri = f"data:image/png;base64,{vis_raw_base64}"

            # 5. Send it back to React
            return JsonResponse({
                'visualization': vis_data_uri,
                'perimeter': perimeter,
                'area': area,
                'chainCode': chain_code
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)