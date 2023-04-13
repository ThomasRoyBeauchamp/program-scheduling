class NetworkSchedule:

    def __init__(self, start_times=None, durations=None, sessions=None):
        self.is_defined = start_times is not None and durations is not None and sessions is not None
        self.start_times = start_times
        self.durations = durations
        self.sessions = sessions

    # TODO: save a network schedule
    # TODO: calculate length and probabilities given app types and number of sessions
    # TODO: generate a random network schedule given the length and probabilities

    def get_session_start_times(self, session_id):
        try:
            self.sessions.index(session_id)
        except AttributeError:
            return None
        except ValueError:
            raise ValueError(f"Network schedule is incomplete and does not define a time slot for session {session_id}")
        if self.sessions.count(session_id) > 1:
            start_times = []
            for i, s in enumerate(self.sessions):
                if s == session_id:
                    start_times.append(self.start_times[i])
            return start_times
        else:
            return self.start_times[self.sessions.index(session_id)]

    def get_session_durations(self, session_id):
        try:
            self.sessions.index(session_id)
        except AttributeError:
            return None
        except ValueError:
            raise ValueError(f"Network schedule is incomplete and does not define a time slot for session {session_id}")
        if self.sessions.count(session_id) > 1:
            durations = []
            for i, s in enumerate(self.sessions):
                if s == session_id:
                    durations.append(self.durations[i])
            return durations
        else:
            return self.durations[self.sessions.index(session_id)]
