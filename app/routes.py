from flask import render_template, flash, redirect, url_for
from app.forms import SearchForm
from app.models import Searches
from app import app, db

from datetime import datetime
from app.spotify import SpotifySearcher, Classifier
artists = [
    {
        "name": "Grandma",
        "picture": "https://image.shutterstock.com/image-photo/old-woman-talking-phone-600w-1510216667.jpg"},
    {
        "name": "Banksy",
        "picture": "https://image.shutterstock.com/image-photo/speaking-loudly-megaphone-600w-1026929530.jpg"}
]



@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    searcher = SpotifySearcher()
    form = SearchForm()
    clf = Classifier()
    songs, happy_songs, sad_songs = None, None, None

    if form.validate_on_submit():
        search = Searches(search_field=form.artist.data, date=datetime.now())
        db.session.add(search)
        db.session.commit()
        songs = searcher.search_artist_songs(form.artist.data)

        happy_songs = clf.get_happy(songs)
        sad_songs = clf.get_sad(songs)
        flash(f'Done!', 'success')
        return render_template('home.html', artists=artists, form=form,
                               happy_songs=happy_songs, sad_songs=sad_songs)
    return render_template('home.html', artists=artists, form=form,
                           happy_songs=happy_songs, sad_songs=sad_songs)
