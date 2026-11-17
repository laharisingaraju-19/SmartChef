import base64
import io

from django.shortcuts import render
from PIL import Image, ImageOps

from .forms import UploadForm
from .predictor import predict


def home(request):
    """Show the upload form; on submit, classify the photo and suggest a smoothie."""
    context = {"form": UploadForm()}

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if form.is_valid():
            image = Image.open(form.cleaned_data["photo"])
            image = ImageOps.exif_transpose(image).convert("RGB")  # fixes sideways phone photos

            guesses, recipe = predict(image)

            # Send a small copy of the photo back to the page without saving it on the server
            preview = image.copy()
            preview.thumbnail((640, 640))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=85)

            context.update({
                "guesses": guesses,
                "best": guesses[0],
                "recipe": recipe,
                "photo_data": base64.b64encode(buffer.getvalue()).decode(),
            })

    return render(request, "classifier/index.html", context)
