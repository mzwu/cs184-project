import utils
import time

class HeuristicAgent:
    def __init__(self, params = [-0.510066, -0.35663, -0.184483, 0.760777, 0, 0], trials = 3):
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
        return self.params[0] * agg + self.params[1] * holes + self.params[2] * bump + self.params[4] * agg_squared + self.params[5] * holes_with_depth
    
    def eval_action(self, state, act, height = []):
        nxt, cleared, done = utils.transition(state, act, height)
        if nxt is None:
            return -1e9
        
        # cur_score = -1e9
        # height2 = utils.get_heights(nxt)
        # for act2 in range(80):
        #     nxt2, cleared2, done2 = utils.transition(nxt, act2, height2)
        #     if nxt2 is None:
        #         continue 
        #     cur_score = max(cur_score, self.params[3] * (cleared + cleared2) + self.eval_board(nxt2))
        # return cur_score
                
        return self.params[3] * cleared + self.eval_board(nxt)
    
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
                    total_cost += 500
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
    params1 = [-0.510066, -0.35663, -0.184483, 0.760777, 0, 0]
    params2 = [-0.7814559850152453, -0.023394217039222485, -0.1961356687120126, 0.591869963380358, 0, 0]
    params3 = [-0.24751747760725967, -0.05946665996584107, -0.7707773413418773, 0.1227229261690893, -0.2553315719496989, -0.5107308258403698]
    params4 = [-0.6676757011739746, -0.31034700285664457, -0.32138862669390333, 0.0009965842688914792, -0.467225432474488, -0.36919188585391305]
    params5 = [-0.8209688443754912, -0.24739868845190294, -0.420685070320271, 0.08634861516012744, -0.2428236403018467, -0.14631716747246387]

    params_set = [params1, params2, params3, params4, params5]
    
    for i in range(5):
        agent = HeuristicAgent(params = params_set[i], trials = 30)
        print("Average " + str(i+1) + ": " + str(agent.eval_agent()))
    # agent = HeuristicAgent(params = [-0.7814559850152453, -0.023394217039222485, -0.1961356687120126, 0.591869963380358, 0, 0], trials = 30)
    # print("Average 2:", agent.eval_agent())
    # agent = HeuristicAgent(params = [-0.24751747760725967, -0.05946665996584107, -0.7707773413418773, 0.1227229261690893, -0.2553315719496989, -0.5107308258403698], trials = 30)
    # print("Average 3:", agent.eval_agent())

    # agent = HeuristicAgent(params = params5, trials = 30)
    # print("Average 5:", agent.eval_agent())
    # agent = HeuristicAgent(params = [-0.24751747760725967, -0.05946665996584107, -0.7707773413418773, 0.1227229261690893, -0.2553315719496989, -0.5107308258403698], trials = 1)
    # agent.eval_agent(print_flag = True)