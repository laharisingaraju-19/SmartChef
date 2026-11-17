"""WSGI entry point, used by servers such as gunicorn when the site is deployed."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smartchef.settings")

application = get_wsgi_application()
