import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd


class SpotifySearcher():

    def __init__(self):
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials())

    def search_artist_songs(self, artist):
        artist_search_results = self.sp.search(artist, type="artist")

        artist_uri = list(artist_search_results.values())[0]["items"][0]["external_urls"]["spotify"]


        albums_search_results = self.sp.artist_albums(artist_uri)

        album_uris = [i["uri"] for i in list(albums_search_results.items())[1][1]]

        songs = [[i["name"], i["uri"]] for j in album_uris for i in list(self.sp.album_tracks(j).values())[1]]
        songs_features = [i + [self.sp.audio_features(i[1])[0]] for i in songs]
        songs_features = [dict(name=i[0], **i[2]) for i in songs_features]

        songs_df = pd.DataFrame(songs_features)

        return songs_df



class Classifier:

    def __init__(self):
        pass

    def get_happy(self, df):
        if df is not None:
            return df[df.danceability > 0.8]
        else:
            None

    def get_sad(self, df):
        if df is not None:
            return df[df.danceability < 0.2]
        else:
            None
