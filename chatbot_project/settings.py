import os, pathlib
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
# Pull from an environment variable, or fall back to localhost for development
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/trait_classification_model_DB"
)
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# directory for local JSONL logs
TRAIT_LOG_DIR = BASE_DIR / "conversation_logs"
TRAIT_LOG_DIR.mkdir(exist_ok=True)

SECRET_KEY = 'your-secret-key'

DEBUG = True

ALLOWED_HOSTS = ["127.0.0.1"]

# Add your app to INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main', 
    'rest_framework'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',  # CSRF protection
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chatbot_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # Set the templates directory (placed at the project root)
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'chatbot_project.wsgi.application'

# --- MongoDB configuration using djongo ---
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'chatbot_db',  # name of your MongoDB database
        'ENFORCE_SCHEMA': False,
        'CLIENT': {
            'host': 'mongodb://localhost:27017',  # update if your MongoDB is on a different host/port
        }  
    }
}

# Password validation (default settings)
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization settings
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# print("STATIC URL IS: ", STATIC_URL)
# print("STATICFILES DIRS ARE: ", STATICFILES_DIRS)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'