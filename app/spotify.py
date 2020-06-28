import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

from app import db
from app.models import RemovedSongs, ExportedSongs

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


class SpotifySearcher:

    def __init__(self):
        self.connection = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials())

    def search_artist_songs(self, artist):
        artist_search_results = self.connection.search(artist, type="artist")

        artist_uri = list(artist_search_results.values())[0]["items"][0]["external_urls"]["spotify"]


        albums_search_results = self.connection.artist_albums(artist_uri)

        album_uris = [i["uri"] for i in list(albums_search_results.items())[1][1]]

        songs = [[i["name"], i["uri"]] for j in album_uris for i in
                 list(self.connection.album_tracks(j).values())[1]]
        songs_features = [i + [self.connection.audio_features(i[1])[0]] for i in songs]
        songs_features = [dict(name=i[0], **i[2]) for i in songs_features]

        songs_df = pd.DataFrame(songs_features)

        return songs_df

class Classifier:

    def __init__(self):
        pass

    def get_labels(self, df):
        if df is not None:
            df["label"] = None
            df.loc[df.danceability > 0.8, "label"] = "happy"
            df.loc[df.danceability < 0.2, "label"] = "sad"
            return df["label"]
        else:
            np.nan

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

class PlaylistClassifier:

    def __init__(self, classifier):

        self.hyperparameter_space = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elastic_net"],
            "l1_ratio": [0.2, 0.4, 0.6, 0.8],
        }

        self.probability_threshold = 0.8

        self.classifier_class = classifier
        self.classifiers = None

        self.features = [
            'danceability',
            'energy',
            'key',
            'loudness',
            'mode',
            'speechiness',
            'acousticness',
            'instrumentalness',
            'liveness',
            'valence',
            'tempo',
        ]

        self.target = 'target'
        self.random_search_tries = 20

    def fit(self, data):

        self.classifiers = dict()
        for label in data.keys():
            df = data[label]
            df[self.features] = df[self.features].fillna(0)
            rscv = RandomizedSearchCV(
                self.classifier_class(),
                param_distributions=self.hyperparameter_space,
                n_iter=self.random_search_tries,
                scoring="roc_auc").fit(df[self.features], df[self.target])
            params = rscv.best_params_
            clf = self.classifier_class(**params)
            clf.fit(df[self.features], df[self.target])
            self.classifiers[label] = clf
        return self

    def predict_labels(self, data, label=None, probability_threshold=None):

        valid_labels = self.classifiers.keys()

        if label is not None:
            assert label in valid_labels, "Unexisting label"
            return self.classifiers[label].predict(data[self.features])
        else:
            predictions = dict()
            for label in valid_labels:
                predictions[label] = (
                    self.classifiers[label]
                        .predict_proba(data[self.features])[:, 1])
                predictions = pd.DataFrame(predictions)
                max_probs = predictions.max(1)
                labels = predictions.idxmax(1).values
                probability_threshold = probability_threshold or self.probability_threshold
                labels[max_probs < probability_threshold] = np.nan
            return labels

    def predict_proba(self, data, label=None):

        valid_labels = self.classifiers.keys()

        if label is not None:
            assert label in valid_labels, "Unexisting label"
            return self.classifiers[label].predict_proba(data[self.features])[:, 1]
        else:
            predictions = dict()
            for label in valid_labels:
                predictions[label] = (
                    self.classifiers[label].predict_proba(data[self.features]))[:, 1]
            return predictions


class TrainingSetCreator:

    min_size = 100
    labels = ["happy", "sad", "dance"]

    def __init__(self):
        self.connection = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials())

    def create_dataset(self):
        data = self.get_datasets()
        class_datasets = self.split_dataset_per_class(data)
        class_datasets = self.apply_to_dict(class_datasets, self.get_relative_weights)
        class_datasets = self.apply_to_dict(class_datasets, self.get_sample_weights)
        return class_datasets

    @staticmethod
    def apply_to_dict(data, fun):
        return {k: fun(v) for k, v in data.items()}

    def get_sample_weights(self, data):
        target_weights = data.groupby("target")["relative_weights"].transform("sum")
        data["class_weights"] = data["relative_weights"] * data["relative_weights"].sum() / target_weights
        return data

    def get_relative_weights(self, data):
        data["relative_weights"] = -1
        data.loc[(data.action_type == "exported") &
                 (data.target == 1), "relative_weights"] = 1
        data.loc[(data.action_type == "cold_start") &
                 (data.target == 1), "relative_weights"] = 1/10
        data.loc[data.action_type == "removed",
                 "relative_weights"] = 2
        data.loc[(data.action_type == "cold_start") &
                 (data.target == 0), "relative_weights"] = 1/10
        data.loc[(data.action_type == "exported") &
                 (data.target == 0), "relative_weights"] = 1/10

        if len(data[data.relative_weights < 0]) > 0:
            print("NEGATIVE WEIGHT FOUND")
        return data

    def get_cold_start_training_set(self):
        features_df = []
        for label in self.labels:
            playlists = self.get_playlists_containing_label(label)
            track_uris = [self.get_tracks_from_playlists(i) for i in playlists]
            track_uris = [k for i in track_uris for k in i]
            track_uris = np.random.choice(track_uris, self.min_size, replace=False)
            df = self.get_features_from_uris(track_uris)
            df["label"] = label
            df["action_type"] = "cold_start"
            features_df.append(df)
        features_df = pd.concat(features_df)
        return features_df

    def get_playlists_containing_label(self, label):
        playlists = [i["uri"] for i in self.connection.search(
            label, type="playlist", limit=20)["playlists"]["items"]]
        return playlists

    def get_tracks_from_playlists(self, playlist_uri):
        tracks = [i["track"]["uri"] for i in self.connection.playlist_tracks(
            playlist_uri)["items"]]
        return tracks

    def split_dataset_per_class(self, data):
        datasets = dict()
        for label, df in data.groupby("label"):

            """
            Get negative samples from other classes. Give priority for positive
            feedback from other classes as negative feedback for the current
            class. If data not available, then use the cold start dataset.
            """
            df_other = data[data.label != label]
            other_exported = (df_other.action_type == "exported").sum()

            df_other = pd.concat([
                df_other.query("action_type == 'exported'")
                    .sample(min(other_exported, self.min_size), replace=True),
                df_other.query("action_type == 'cold_start'")
                    .sample(self.min_size, replace=True),
            ])
            d = pd.concat([
                df.query("action_type == 'exported'").assign(target=1),
                df.query("action_type == 'cold_start'")
                    .sample(self.min_size, replace=True).assign(target=1),
                df.query("action_type == 'removed'").assign(target=0),
                df_other.assign(target=0),
                ])
            datasets[label] = d
        return datasets

    def get_datasets(self):
        return pd.concat([
            self.get_cold_start_training_set(),
            self.get_feedback_from_exported(),
            self.get_feedback_from_removed(),
        ])

    def get_feedback_from_exported(self):
        features_df = self.get_features_from_db_table(ExportedSongs)
        features_df["action_type"] = "exported"
        return features_df

    def get_feedback_from_removed(self):
        features_df = self.get_features_from_db_table(RemovedSongs)
        features_df["action_type"] = "removed"
        return features_df

    def get_features_from_db_table(self, table):
        uris = [i[0] for i in
                table.query.with_entities(table.uri).all()][:5]

        labels = [i[0] for i in
                  table.query.with_entities(table.label).all()][:5]

        features_df = self.get_features_from_uris(uris)
        features_df["label"] = labels
        return features_df

    def get_features_from_uris(self, uris):
        songs_features = [self.connection.audio_features(i)[0]
                          for i in uris]

        return pd.DataFrame(songs_features)


def train_classifier():
    data = TrainingSetCreator().create_dataset()
    return PlaylistClassifier(LogisticRegression).fit(data)

