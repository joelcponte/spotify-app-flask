from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length


class SearchForm(FlaskForm):
    artist = StringField("Artist's Name",
                         validators=[DataRequired(), Length(min=2, max=20)])
    search = SubmitField("Search")