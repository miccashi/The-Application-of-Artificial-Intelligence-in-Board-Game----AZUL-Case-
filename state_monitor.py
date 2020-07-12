from collections import deque, defaultdict

states = deque(maxlen=1)
search_time = deque(maxlen=1)

def set_state(state):
    states.append(state)

def get_state():
    return states[0]

# def set_time(time):
#     search_time.append(time)
#
# def get_time():
#     return search_time[0]
times = defaultdict(float)
def set_search_time(name, time):
    times[name] += time

def get_search_time():
    return times

speical_actions = defaultdict(int)

def increase_actions(name):
    speical_actions[name]+=1

def get_special_action():
    return speical_actions