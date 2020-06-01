import pandas as pd

def get_session_var(var, session, type="data_frame", return_if_missing=None):
    if var in session:
        if type == "data_frame":
            return pd.read_json(session.get(var))
        else:
            raise NotImplementedError()
    else:
        return return_if_missing
