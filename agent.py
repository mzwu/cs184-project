import utils
import time

class HeuristicAgent:
    def __init__(self, params = [-0.510066, -0.35663, -0.184483, 0.760777], trials = 3):
        utils.initialize()
        self.params = params 
        self.trials = trials
    
    def eval_board(self, state):
        if state is None:
            return -1e9

        height = [20]*10
        for col in range(10):
            while height[col] > 0 and state[(20 - height[col])*10 + col] == 0:
                height[col] -= 1 

        agg = sum(height)
        agg_squared = 0
        for h in height:
            agg_squared += h ** 2
        bump = sum([abs(height[i+1] - height[i]) for i in range(9)])
        holes = 0
        holes_with_depth = 0
        for y in range(10):
            for x in range(20 - height[y], 20):
                if state[x*10 + y] == 0:
                    holes += 1
                    holes_with_depth += x - (20 - height[y])
        return self.params[0] * agg + self.params[1] * holes + self.params[2] * bump
    
    def eval_action(self, state, act, height = []):
        nxt, cleared, done = utils.transition(state, act, height)
        if nxt is None:
            return -1e9
                
        return 0.760777 * cleared + self.eval_board(nxt)
    
    def eval_agent(self, print_flag = False):
        total_cost = 0
        for trial in range(self.trials):
            state = utils.starting_position()
            if print_flag:
                utils.print_info(state)
            
            moves = 0
            while True:
                best_score, best_act = -1e9, -1
                height = utils.get_heights(state)
                for act in range(80):
                    cur_score = self.eval_action(state, act, height)

                    if cur_score > best_score:
                        best_score, best_act = cur_score, act

                moves += 1
                state, cleared, done = utils.transition(state, best_act)
                if state is None:
                    total_cost += 1e9
                    break

                if print_flag:
                    utils.print_info(state)

                if done:
                    total_cost += moves
                    if print_flag:
                        print("Finished in", moves, "moves")
                    break
        return total_cost / self.trials
    

if __name__ == "__main__":
    agent = HeuristicAgent(trials = 10)
    start = time.time()
    print("Average:", agent.eval_agent())
    print(time.time() - start)