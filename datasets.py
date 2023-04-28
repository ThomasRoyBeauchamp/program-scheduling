import os


def create_dataset(id, n_sessions):
    path = os.path.dirname(__file__) + "/configs/"

    if id == 0:
        return {path + "bqc": n_sessions}
    elif id == 1:
        return {path + "pingpong": n_sessions}
    elif id == 2:
        return {path + "qkd": n_sessions}
    elif id == 3:
        if n_sessions % 2 != 0:
            raise ValueError(f"Number of session is not divisible by 2 ({n_sessions}), pick a different dataset.")
        return {
            path + "bqc": int(n_sessions / 2),
            path + "pingpong": int(n_sessions / 2),
        }
    elif id == 4:
        if n_sessions % 2 != 0:
            raise ValueError(f"Number of session is not divisible by 2 ({n_sessions}), pick a different dataset.")
        return {
            path + "bqc": int(n_sessions / 2),
            path + "qkd": int(n_sessions / 2),
        }
    elif id == 5:
        if n_sessions % 2 != 0:
            raise ValueError(f"Number of session is not divisible by 2 ({n_sessions}), pick a different dataset.")
        return {
            path + "qkd": int(n_sessions / 2),
            path + "pingpong": int(n_sessions / 2),
        }
    elif id == 6:
        if n_sessions % 3 != 0:
            raise ValueError(f"Number of session is not divisible by 3 ({n_sessions}), pick a different dataset.")
        return {
            path + "bqc": int(n_sessions / 3),
            path + "pingpong": int(n_sessions / 3),
            path + "qkd": int(n_sessions / 3)
        }
    else:
        print(f"Dataset ID {id} not recognised.")
