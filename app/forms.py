from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length, ValidationError
from app.models import Searches


class SearchForm(FlaskForm):
    artist = StringField("Artist's Name",
                         validators=[DataRequired(), Length(min=2, max=20)])
    search = SubmitField("Search")

    def validate_artist(self, artist):

        if artist.data == "Coldplay":
            raise ValidationError(f"Not able to search for {artist.data}.")