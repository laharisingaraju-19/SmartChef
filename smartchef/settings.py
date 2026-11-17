"""Settings for the SmartChef Django project.

Values that change between your laptop and a server (secret key, debug mode,
allowed hosts) are read from environment variables, with safe defaults for
running locally.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-key-change-me-in-production")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Render tells the app its public address through this variable
RENDER_HOST = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
if RENDER_HOST:
    ALLOWED_HOSTS.append(RENDER_HOST)

# The app has no database models, so only the pieces it uses are installed
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "classifier",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # Serves the CSS when the site is deployed (Django only does this itself in debug mode)
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# The host serves the app over HTTPS through a proxy
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

ROOT_URLCONF = "smartchef.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    },
]

WSGI_APPLICATION = "smartchef.wsgi.application"

# No models are stored, but Django expects a database setting
DATABASES = {
    "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Where the trained model, class names and recipes live
MODEL_DIR = BASE_DIR / "model"

# Where to download the ONNX model from if model/smartchef.onnx is missing.
# Change "laharisingaraju" to your Hugging Face username if it is different.
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/laharisingaraju/smartchef-resnet18/resolve/main/smartchef.onnx",
)

# Reject uploads larger than 8 MB
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
