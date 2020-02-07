from flask import Flask, render_template
from forms import SearchForm
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config["SECRET_KEY"] = "0e63eda1ff2d33926eab19a123e1118b"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

db = SQLAlchemy(app)

class Searches(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_field = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"User('{self.search_field}', '{self.date}')"

artists = [
    {
        "name": "Grandma",
        "picture": "https://image.shutterstock.com/image-photo/old-woman-talking-phone-600w-1510216667.jpg"},
    {
        "name": "Banksy",
        "picture": "https://image.shutterstock.com/image-photo/speaking-loudly-megaphone-600w-1026929530.jpg"}
]


@app.route('/')
@app.route('/home')
def home():
    form = SearchForm()
    return render_template('home.html', artists=artists, form=form)


if __name__ == '__main__':
    app.run()
