from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import Config
import os

app = Flask(__name__)

app.config.from_object(Config)
db = SQLAlchemy(app)

os.environ["SPOTIPY_CLIENT_ID"] = os.environ.get('SPOTIPY_CLIENT_ID') or \
                    "75c5af0b7a014b95b47f4a12870378a9"
os.environ["SPOTIPY_CLIENT_SECRET"] = os.environ.get('SPOTIPY_CLIENT_SECRET') or \
                        "732602451837434388fe0f3c9e9da31a"

from app import routes