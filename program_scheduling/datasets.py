import os


def create_dataset(dataset_id, n_sessions, only_session_name=False):
    """
    Method that creates the dataset dictionary. Contains either (incomplete) paths to config files or only session
    names. Paths are incomplete because then it is possible to specify whether it should be the configuration file
    for Alice or Bob.

    :param dataset_id: ID of the dataset. Possible values are integers from 0 to 6.
    :param n_sessions: Total number of sessions in the dataset.
    :param only_session_name: Should be True if one wants the dictionary to contain only the session names.
    :return: Dictionary with sessions (paths to configs or names) and number of sessions.
    """
    # note that this will be different whether called from PyCharm (the entire path) vs command-line (relative path)
    config_path = os.path.dirname(__file__).rstrip("program_scheduling") + "configs/"

    path = "" if only_session_name else config_path

    if dataset_id == 0:
        return {path + "bqc": n_sessions}
    elif dataset_id == 1:
        return {path + "pingpong": n_sessions}
    elif dataset_id == 2:
        return {path + "qkd": n_sessions}
    elif dataset_id == 3:
        if n_sessions % 2 != 0:
            raise ValueError(f"Number of session is not divisible by 2 ({n_sessions}), pick a different dataset.")
        return {
            path + "bqc": int(n_sessions / 2),
            path + "pingpong": int(n_sessions / 2),
        }
    elif dataset_id == 4:
        if n_sessions % 2 != 0:
            raise ValueError(f"Number of session is not divisible by 2 ({n_sessions}), pick a different dataset.")
        return {
            path + "bqc": int(n_sessions / 2),
            path + "qkd": int(n_sessions / 2),
        }
    elif dataset_id == 5:
        if n_sessions % 2 != 0:
            raise ValueError(f"Number of session is not divisible by 2 ({n_sessions}), pick a different dataset.")
        return {
            path + "pingpong": int(n_sessions / 2),
            path + "qkd": int(n_sessions / 2),
        }
    elif dataset_id == 6:
        if n_sessions % 3 != 0:
            raise ValueError(f"Number of session is not divisible by 3 ({n_sessions}), pick a different dataset.")
        return {
            path + "bqc": int(n_sessions / 3),
            path + "pingpong": int(n_sessions / 3),
            path + "qkd": int(n_sessions / 3)
        }
    else:
        print(f"Dataset ID {dataset_id} not recognised.")
