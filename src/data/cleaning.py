def is_full_non_ascii(string):
    """Checks if the given string consists only out of non ascii characters"""
    full_non_ascii = True
    for char in string:
        if char.isspace():
            continue
        elif char.isascii():
            full_non_ascii = False
    return full_non_ascii


def clean_line(line):
    """Removes all non ascii characters in a string"""
    result = ""
    last_was_non_ascii = False
    for char in line:
        if char.isascii():
            result += char
            if last_was_non_ascii:
                last_was_non_ascii = False
        else:
            last_was_non_ascii = True
            continue
    return result


def remove_non_ascii_strings(users, remove_space=False, return_stats=False):
    """Cleans the strings of a user by removing non ascii characters/lines

    :param users: dictionary of users which should be cleaned
    :param remove_space: if true lines with only whitespace characters are tread as non ascii lines are therefore also removed
    :param return_stats: return some basic stats of the cleaning process
    :return: dictionary with cleaned users
    """
    cleaned_users = {}
    full_non_ascii_queries = 0
    part_non_ascii_queries = 0
    for user_key in users.keys():
        user = users[user_key]
        user.sort()
        cleaned_user = []
        for entry in user:
            if is_full_non_ascii(entry.interaction) or (entry.interaction.isspace() and remove_space):
                full_non_ascii_queries += 1
                if len(cleaned_user) > 0 and entry.boundary:
                    cleaned_user[-1].boundary = True
            else:
                old = entry.interaction
                entry.interaction = clean_line(entry.interaction)
                if old != entry.interaction:
                    part_non_ascii_queries += 1
                cleaned_user.append(entry)
        cleaned_users[user_key] = cleaned_user
    if not return_stats:
        return cleaned_users
    else:
        return cleaned_users, full_non_ascii_queries, part_non_ascii_queries


def remove_users_with_single_entry(users):
    """Removes all users which have only one entry"""
    return_users = {}
    for user in users:
        if len(users[user]) > 1:
            return_users[user] = users[user]
    return return_users
