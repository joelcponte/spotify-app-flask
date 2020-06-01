from app import db
from datetime import datetime


class Searches(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_field = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"User('{self.search_field}', '{self.date}')"

class RemovedSongs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    # key = db.Column(db.Integer, nullable=False)
    # danceability = db.Column(db.Float(precision=5), nullable=False)
    # energy = db.Column(db.Float(precision=5), nullable=False)
    # loudness = db.Column(db.Float(precision=5), nullable=False)
    # mode = db.Column(db.Integer, nullable=False)
    # speechiness = db.Column(db.Float(precision=5), nullable=False)
    # acousticness = db.Column(db.Float(precision=5), nullable=False)
    # instrumentalness = db.Column(db.Float(precision=5), nullable=False)
    # liveness = db.Column(db.Float(precision=5), nullable=False)
    # valence = db.Column(db.Float(precision=5), nullable=False)
    # tempo = db.Column(db.Float(precision=5), nullable=False)
    playlist_id = db.Column(db.Integer)
    uri = db.Column(db.String(50), nullable=False)
    label = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"RemovedSong('{self.playlist_id}', '{self.name}', '{self.uri}', '{self.label}', '{self.date}')"

class ExportedSongs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    playlist_id = db.Column(db.Integer)
    name = db.Column(db.String(80), nullable=False)
    uri = db.Column(db.String(50), nullable=False)
    label = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"ExportedSong('{self.playlist_id}', '{self.name}', '{self.uri}', '{self.label}', '{self.date}')"

class Posts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20), nullable=False)
    image_link = db.Column(db.String(400), unique=True, nullable=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Post('{self.title}', '{self.image_link}', '{self.date}')"