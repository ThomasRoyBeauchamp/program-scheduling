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
        return {
            path + "bqc": n_sessions / 2,
            path + "pingpong": n_sessions / 2,
        }
    elif id == 4:
        return {
            path + "bqc": n_sessions / 2,
            path + "qkd": n_sessions / 2,
        }
    elif id == 5:
        return {
            path + "qkd": n_sessions / 2,
            path + "pingpong": n_sessions / 2,
        }
    elif id == 6:
        return {
            path + "bqc": n_sessions / 3,
            path + "pingpong": n_sessions / 3,
            path + "qkd": n_sessions / 3
        }
    else:
        print(f"Dataset ID {id} not recognised.")
