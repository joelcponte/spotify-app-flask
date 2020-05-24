from app import db
from datetime import datetime


class Searches(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_field = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"User('{self.search_field}', '{self.date}')"

class Posts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20), nullable=False)
    image_link = db.Column(db.String(400), unique=True, nullable=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Post('{self.title}', '{self.image_link}', '{self.date}')"