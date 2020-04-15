def epoch_time(start_time, end_time):
    elapsed_secs = end_time - start_time
    elapsed_mins = elapsed_secs / 60
    return elapsed_mins, elapsed_secs