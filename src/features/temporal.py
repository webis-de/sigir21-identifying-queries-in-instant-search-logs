def calc_time_gap(interaction, next_interaction):
    """computes the time gap between two interactions"""
    return abs((next_interaction.timestamp - interaction.timestamp).total_seconds())


def time_gap(timestamp1, timestamp2):
    """computes the time gap between two timestamps"""
    return abs((timestamp2 - timestamp1).total_seconds())