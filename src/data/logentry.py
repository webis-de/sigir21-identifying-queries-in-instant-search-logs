import datetime
import json


class LogEntry:
    """Represents a LogEntry """

    def __init__(self, json_str):
        parsed = json.loads(json_str)
        parsed['time'] = parsed['time'].replace('.', ',')
        try:
            timestamp = datetime.datetime.strptime(parsed['date']+" "+parsed['time'], "%Y-%m-%d %H:%M:%S,%f")
        except Exception:
            timestamp = datetime.datetime.strptime(parsed['date']+" "+parsed['time'], "%Y-%m-%d %H:%M:%S")
        if 'log_version' in parsed:
            self.log_version = parsed['log_version']
        else:
            self.log_version = None
        self.timestamp = timestamp
        
        if 'ip' in parsed:
            self.ip = parsed['ip']
        else:
            self.ip = None
        
        if 'uid' in parsed:
            self.uid = parsed['uid']
        else:
            self.uid = None
        
        if 'info' in parsed:
            self.info = parsed['info']
        else:
            self.info = None
        
        if 'query' in parsed:
            self.interaction = parsed['query']
        else:
            self.interaction = parsed['interaction']
        
        if 'border' in parsed:
            self.boundary = parsed['border']
        elif 'boundary' in parsed:
            self.boundary = parsed['boundary']
        else:
            self.boundary = False

        if self.log_version == "v1.2":
            self.boundary = True

        self.boundary_prediction = None

        self.features = None
        self.features_time = None
        self.features_lookahead = None

        if 'agent' in parsed:
            self.agent = parsed['agent']
        else:
            self.agent = None

        self.time_difference_next_entry = None

        if 'empty_results' in parsed:
            self.empty_response = parsed['empty_results']
        else:
            self.empty_response = None

        if 'results' in parsed:
            self.netspeak_results = parsed['results']
            if self.empty_response is None:
                self.empty_response = len(self.netspeak_results) == 0
        else:
            self.netspeak_results = None

        self.occurred_before = 0

    def create_json(self, boundary=False, privacy=False, boundary_annotated=False):
        if boundary:
            output = {"log_version": self.log_version, "date": self.timestamp.strftime("%Y-%m-%d"),
                      "time": self.timestamp.strftime("%H:%M:%S,%f"), "ip": self.ip,
                      "info": self.info, "agent": self.agent, "query": self.interaction, "border": self.boundary_prediction}
        elif privacy and boundary_annotated:
            output = {"uid": self.uid,
                      "date": self.timestamp.strftime("%Y-%m-%d"),
                      "time": self.timestamp.strftime("%H:%M:%S,%f")[:-3],
                      "boundary": self.boundary,
                      "interaction": self.interaction, }
        else:
            output = {"log_version": self.log_version, "date": self.timestamp.strftime("%Y-%m-%d"),
                      "time": self.timestamp.strftime("%H:%M:%S,%f"), "ip": self.ip,
                      "info": self.info, "agent": self.agent, "query": self.interaction}


        return str(json.dumps(output))

    def __le__(self, other):
        return self.timestamp <= other.timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __ge__(self, other):
        return self.timestamp >= other.timestamp

    def __gt__(self, other):
        return self.timestamp > other.timestamp

    def __repr__(self):
        return f"<{self.timestamp} from {self.ip}: {self.interaction}|{self.boundary}>"

    def __eq__(self, other):
        return self.timestamp == other.timestamp and self.interaction == other.interaction

    def __hash__(self):
        return hash((self.timestamp, self.interaction))