from django import forms
from django.conf import settings


class UploadForm(forms.Form):
    # ImageField uses Pillow to check that the upload really is an image
    photo = forms.ImageField(label="Photo of a fruit or vegetable")

    def clean_photo(self):
        photo = self.cleaned_data["photo"]
        if photo.size > settings.MAX_UPLOAD_BYTES:
            raise forms.ValidationError("Please upload an image under 8 MB.")
        return photo
