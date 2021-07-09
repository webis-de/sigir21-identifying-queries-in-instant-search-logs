from datetime import timedelta


def times_occuring(user, threshold=timedelta(hours=24)):
    """Counts how often a user occurred after the threshold time"""
    counter = 1
    last_time = user[0].timestamp
    for i in range(1, len(user)):
        if abs((user[i].timestamp - last_time).total_seconds()) > threshold.total_seconds():
            counter += 1
        last_time = user[i].timestamp
    return counter


def time_between_first_and_last_query(user):
    """returns the time difference between the first log entry and the last one of this user"""
    return abs((user[-1].timestamp - user[0].timestamp).total_seconds())


def operator_use_count(user):
    """returns how often the user uses a netspeak operator"""
    operators = 0
    unary_operators = ["...", "?", "#"]
    binary_operators = [["{", "}"], ["[", "]"]]
    for query in user:
        has_operator = False
        for operator in unary_operators:
            if operator in query.interaction:
                has_operator = True
        for operator in binary_operators:
            if operator[0] in query.interaction and operator[1] in query.interaction:
                has_operator = True
        if has_operator:
            operators += 1
    return operators


def count_see_saw(user, ignore_example_queries=True):
    """Count how often the user shows a see-saw pattern"""
    if len(user) < 3:
        return 0
    example_queries = ["how to ? this", "see ... works", "it's [ great well ]", "and knows #much", "{ more show me }",
                       "m...d ? g?p", "waiting ? response", "waiting + response", "waiting ? ? response",
                       "waiting * response", "the same [ like as ]", "{ only for members }", "waiting * #response",
                       "waiting ? ? response | waiting ? response"]
    A = None
    B = None
    indices_to_ignore = []
    counts = 0
    for i in range(len(user)-2):
        if i in indices_to_ignore:
            continue
        A = user[i]
        B = user[i + 1]
        if A.interaction in example_queries or B.interaction in example_queries:
            continue
        if (user[i + 2].interaction == A.interaction):
            # state 1 found query A, -1 found query B last time
            state = -1
            for j in range(i + 2, len(user)):
                if state == -1 and user[j].interaction == A.interaction:
                    indices_to_ignore.append(j - 2)
                    state = 1
                elif state == 1 and user[j].interaction == B.interaction:
                    indices_to_ignore.append(j - 2)
                    state = -1
                else:
                    counts += 1
                    break
    return counts


def count_log_entries_in_dict(dict):
    """Count the number of log entries in the dictionary"""
    entries = 0
    for user in dict:
        entries += len(dict[user])
    return entries


def calculate_average_query_length(users):
    """Computes the average query length (number of log entries per query)"""
    query_length = []
    queries = 0
    for user_key in users.keys():
        current_length = 1
        for entry in users[user_key]:
            current_length += 1
            if entry.boundary:
                query_length.append(current_length)
                queries += 1
                current_length = 1
    return sum(query_length)/float(queries)