import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
import os

from util import dist
from constants import OR_ACC_COLORS

class Trial():
    N_TRIALS = 80
    
    def __init__(self, match_id, trial_idx=0):
        self.match_id = match_id
        self.trial_idx = trial_idx
        self.problem = None
        self.trial_data = None
        self.metadata = None
        self.p_data = None
        self.load_metadata(match_id)
        self.load_trial(trial_idx)
    
    @staticmethod
    def All(match_id):
        all_trials = []
        for i in range(Trial.N_TRIALS):
            try:
                t = Trial(match_id, trial_idx=i)
            except Exception as e:
                print("Error loading trial data: %s, trial %s: %s" % (match_id, i, e))
            else:
                all_trials.append(t)
        return all_trials

    def load_metadata(self, match_id):
        with open("../data/%s/metadata.json" % (match_id), 'r') as f:
            self.metadata = json.load(f)
    
    def load_trial(self, trial_idx):
        # Get trial metadata
        self.trial_data = self.metadata.get('trials').get(str(int(trial_idx)))
        # Get problem / map
        map_name = self.metadata.get("matchData").get("map_order")[trial_idx]
        with open("../map_creation/maps_v1/%s.json" % (map_name), 'r') as f:
            self.problem = json.load(f)
        # Get each player's stored data
        with open("../data/%s/trial%d.json" % (self.match_id, trial_idx), 'r') as f:
            data = json.load(f)
            self.p_data = data
        self.problem['mapName'] = map_name
        
    def node_loc(self, node_id):
        n = self.problem.get("nodes").get(str(node_id))
        return (n.get('X'), n.get('Y'))
    
    def get_duration(self):
        """
        Use trial start and last host move to overcome out-of-sync issues
        with ts_finished
        """
#         ts_started = self.trial_data.get('ts_started')
#         return (ts_finished - ts_started) / 1000.
        last_event = self.events(moves_only=False)[-1]
        return last_event.get("time")

    def state_timeseries(self):
        """
        Using event data, reconstruct state timeseries for player
        """
        middle_loc = self.node_loc(self.problem.get("middle_node"))
        # Start at middle
        state = {
            'eventType': 'start',
            'loc': middle_loc,
            'time': 0
        }  
        states = [state]
        p_events = self.events()
        done = False
        playerLoc = middle_loc
        for e in p_events:
            next_state = dict(states[-1])    
            nextMove = e.get("nodeId")
            if nextMove is not None:
                playerLoc = self.node_loc(nextMove)
            next_state['loc'] = playerLoc
            next_state['nodeId'] = nextMove
            next_state['time'] = e.get("time")
            next_state['eventType'] = e.get("eventType")
            states.append(next_state)
        return states
    
    def oracle_request_index(self, states):
        """
        Return number of gems collected when oracle called
        -1 if oracle not requested
        """
        idx = -1
        n_gems = 0
        for s in states:
            if s['eventType'] == "collect":
                n_gems += 1
            elif s['eventType'] == "oracle_request":
                idx = n_gems
                break
        return idx
                
        
    def start_node(self, pid=0):
        return self.problem.get("middle_node")
    
    def move_dir(self, from_node_id, to_node_id):
        loc0 = np.array(self.node_loc(from_node_id))
        loc1 = np.array(self.node_loc(to_node_id))
        vector = loc1 - loc0
        assert abs(vector.sum()) == 1
        if np.array_equal(vector, [1, 0]):
            return 'right'
        elif np.array_equal(vector, [-1, 0]):
            return 'left'
        elif np.array_equal(vector, [0, 1]):
            return 'up'
        elif np.array_equal(vector, [0, -1]):
            return 'down'
    
    def events(self, moves_only=False):
        data = self.p_data
        events = data.get("trial_data").get("TrialEventData", [])
        if events is None:
            events = []
        if not events:
            print("M %s T %d has no events" % (self.match_id, self.trial_idx))
        if moves_only:
            events = [e for e in events if e.get("eventType") == "move"]
        return events
    
    def render_timing_hist(self, ax=None):
        fig, ax = plt.subplots(dpi=144, figsize=(8, 5))
        lastTime = 0
        deltas = []
        collect_idxs = []
        for i, state in enumerate(self.state_timeseries()):
            eventType = state.get("eventType")
            time = state.get("time")
            if eventType == "collect":
                collect_idxs.append(i)
            delta = time - lastTime
            deltas.append(delta)
            lastTime = time
        barlist = ax.bar(list(range(len(deltas))), deltas)
        for idx in collect_idxs:
            barlist[idx].set_color('green')
        ax.set_ylabel("Delay (s)")
        
    def render_problem(self, ax=None, title=None, no_rewards=False):
        if ax is None:
            fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
        X = []
        Y = []
        C = []
        acc = self.problem.get("conditions").get("oracle")
        color = OR_ACC_COLORS[acc]
        for node_id, n in self.problem.get('nodes').items():
            X.append(n.get('X'))
            Y.append(n.get('Y'))  
            reward_rank, reward_value = n.get("reward_rank", 0), n.get("reward_value", 0)
            has_reward = reward_value > 0
            c = color if has_reward else "gray"
            C.append(c)
        for link_key in self.problem.get('links').keys():
            src_id, tgt_id = link_key.split('_')
            src = self.problem.get('nodes')[src_id]
            tgt = self.problem.get('nodes')[tgt_id]        
            sx, sy = src.get('X'), src.get('Y')
            tx, ty = tgt.get('X'), tgt.get('Y')
            ax.plot([sx, tx], [sy, ty], c='gray', lw=1.7, zorder=1)
        ax.scatter(X, Y, c=C, s=50, zorder=10)
        if title:
            ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        return ax
        
    def render(self, ax=None):
        """
        Render player moves during trial, on top of map
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
        self.render_problem(ax=ax)
        X, Y, C = [], [], []
        MW = 0.15
        pcolor = '#222'
        patches = []
        pkwargs = {'zorder': 100, 'fill': True}
        for state in self.state_timeseries():
            nx, ny = state.get("loc")
            eventType = state.get("eventType")
            if eventType == "oracle_request":
                patches.append(Ellipse((nx, ny), MW, MW, color='yellow', **pkwargs))
            elif eventType == "collect":
                patches.append(Ellipse((nx, ny), MW, MW, color='green', **pkwargs))
            X.append(nx)
            Y.append(ny)                
            C.append(pcolor)
        ax.plot(X, Y, lw=5, c=pcolor, alpha=1.0)
        for patch in patches:
            ax.add_patch(patch)
        oracleCalled = self.trial_data.get("oracleRequested")
        nGems = len(self.trial_data.get("gems_collected"))
        score = self.trial_data.get("score")
        ax.set_title(self.problem['mapName'] + " Oracle: %s, Gems: %d (%d points)" % ("Yes" if oracleCalled else "No", nGems, score))
