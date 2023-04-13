def create_dataset(id, n_sessions):
    if id == 0:
        return {"../configs/bqc": n_sessions}
    elif id == 1:
        return {"../configs/pingpong": n_sessions}
    elif id == 2:
        return {"../configs/qkd": n_sessions}
    elif id == 3:
        return {
            "../configs/bqc": n_sessions / 2,
            "../configs/pingpong": n_sessions / 2,
        }
    elif id == 4:
        return {
            "../configs/bqc": n_sessions / 2,
            "../configs/qkd": n_sessions / 2,
        }
    elif id == 5:
        return {
            "../configs/qkd": n_sessions / 2,
            "../configs/pingpong": n_sessions / 2,
        }
    elif id == 6:
        return {
            "../configs/bqc": n_sessions / 3,
            "../configs/pingpong": n_sessions / 3,
            "../configs/qkd": n_sessions / 3
        }
    else:
        print(f"Dataset ID {id} not recognised.")
