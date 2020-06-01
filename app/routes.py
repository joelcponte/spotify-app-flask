from flask import render_template, flash, session, redirect
from app.forms import SearchForm
from app.models import Searches, RemovedSongs, ExportedSongs
from app import app, db

from datetime import datetime
from app.spotify import SpotifySearcher, Classifier
from app.helpers import get_session_var


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():

    session["playlist_id"] = max(
        db.session.query(db.func.max(ExportedSongs.playlist_id)).scalar() or 0,
        db.session.query(db.func.max(RemovedSongs.playlist_id)).scalar() or 0
    ) + 1
    searcher = SpotifySearcher()
    form = SearchForm()
    clf = Classifier()
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(columns=["name", "label"])
    # happy_songs = get_session_var("happy_songs", session, return_if_missing=df)
    # sad_songs = get_session_var("sad_songs", session,  return_if_missing=df)
    session_songs = get_session_var("session_songs", session,  return_if_missing=df)
    if form.validate_on_submit():
        search = Searches(search_field=form.artist.data, date=datetime.now())
        db.session.add(search)
        db.session.commit()
        songs = searcher.search_artist_songs(form.artist.data)

        songs["label"] = clf.get_labels(songs)
        songs = songs.dropna(subset=["label"])[["name",  "label", "uri"]]
        session_songs = pd.concat([session_songs, songs]).drop_duplicates().reset_index(drop=True)
        session["session_songs"] = session_songs.to_json()

        flash(f'Done!', 'success')
        return render_template('home.html', form=form,
                               songs=session_songs,
                               happy_songs=session_songs[session_songs.label=="happy"],
                               sad_songs=session_songs[session_songs.label=="sad"])
    return render_template('home.html', form=form,
                           songs=session_songs,
                           happy_songs=session_songs[session_songs.label=="happy"],
                           sad_songs=session_songs[session_songs.label=="sad"])

@app.route('/delete/<string:uri>')
def remove(uri):

    try:
        session_songs = get_session_var("session_songs", session)
        s = session_songs[session_songs.uri == uri]
        session_songs = session_songs[session_songs.uri != uri]
        session["session_songs"] = session_songs.to_json()

        song = RemovedSongs(
            playlist_id=session["playlist_id"],
            name=s["name"].values[0],
            # key=s["key"],
            # danceability=s["danceability"],
            # energy=s["energy"],
            # loudness=s["loudness"],
            # mode=s["mode"],
            # speechiness=s["speechiness"],
            # acousticness=s["acousticness"],
            # instrumentalness=s["instrumentalness"],
            # liveness=s["liveness"],
            # valence=s["valenc"],
            # tempo=s["tempo"],
            uri=s["uri"].values[0],
            label=s["label"].values[0],
        )
        db.session.add(song)
        db.session.commit()

        return redirect("/")
    except:
        return "There was a problem removing the song."


@app.route('/export_playlist/<string:label>')
def export_playlist(label):
    print(label)
    try:
        session_songs = get_session_var("session_songs", session,
                                        return_if_missing=None)
        session_songs = session_songs.query("label == label")
        session_songs = session_songs.dropna()
    except:
        return "There was a problem exporting the playlist."

    if len(session_songs) < 1:
        flash("The playlist can't be empty", "error")
    try:
        for _, s in session_songs.iterrows():
            song = RemovedSongs(
                playlist_id=session["playlist_id"],
                name=s["name"].values[0],
                uri=s["uri"].values[0],
                label=s["label"].values[0],
            )
            db.session.add(song)
        db.session.commit()
        return redirect("/")
    except:
        return "There was a problem exporting the playlist."



@app.route('/remove_all/<string:label>')
def remove_all(label):
    try:
        import pandas as pd
        if label is not "all":
            session["session_songs"] = (get_session_var("session_songs",
                                                       session,
                                                       return_if_missing=None)
                                        .loc[lambda x: x.label != label]
                                        .to_json())
        else:
            del session["session_songs"]
        return redirect("/")
    except:
        return "There was a problem deleting the playlist."