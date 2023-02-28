class NetworkSchedule:

    def __init__(self, start_times=None, durations=None, sessions=None):
        self.is_defined = start_times is not None and durations is not None and sessions is not None
        self.start_times = start_times
        self.durations = durations
        self.sessions = sessions

    def get_session_start_time(self, session_id):
        try:
            self.sessions.index(session_id)
        except AttributeError:
            return None
        except ValueError:
            raise ValueError(f"Network schedule is incomplete and does not define a time slot for session {session_id}")
        return self.start_times[self.sessions.index(session_id)]

    def get_session_duration(self, session_id):
        try:
            self.sessions.index(session_id)
        except AttributeError:
            return None
        except ValueError:
            raise ValueError(f"Network schedule is incomplete and does not define a time slot for session {session_id}")
        return self.durations[self.sessions.index(session_id)]
