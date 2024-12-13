import utils 
import agent 
import random
import math

def normalize(p):
    norm = math.sqrt(sum([x*x for x in p]))
    return [x/norm for x in p]

# uses the SPSA optimizer to find the ideal parameters for the minimum eigenvalue
# 
def SPSA(p, trials = 3, bounds = [(-10,0), (-10,0), (-10,0), (0,10), (-10,0), (-10,0)]):

    # set starting parameters
    dim = len(p)
    # p=[0 for i in range(dim)]
    alpha = 0.602
    gamma = 0.101
    a = 0.001
    c = 0.01
    A = 100

    for k in range(100):
        # alter parameters in the final stretch
        if k > 35:
            alpha = 1
            gamma = 1/6.0

        # pick a random direction
        delta = [random.randint(0, 1)*2 - 1 for i in range(dim)]
        ak = a / ((A + k + 1) ** alpha)
        ck = c / ((k+1) ** gamma)

        # calculate and measure in both directions
        yplus = [p[i] + ck*delta[i] for i in range(dim)]
        yminus = [p[i] - ck*delta[i] for i in range(dim)]
        
        agentplus = agent.HeuristicAgent(yplus, trials)
        mplus = agentplus.eval_agent()
        agentminus = agent.HeuristicAgent(yminus, trials)
        mminus = agentminus.eval_agent()


        # update parameters based on results
        ghat = [(mplus - mminus) / (2 * ck * delta[i]) for i in range(dim)]
        print("yplus, mplus:", yplus, mplus)
        print("yminus, mminus:", yminus, mminus)
        print("ak * ghat:", [ak*x for x in ghat])
        p = [p[i] - ak*ghat[i] for i in range(dim)]
        p = [max(bounds[i][0], min(bounds[i][1], p[i])) for i in range(dim)]
        # norm = math.sqrt(sum([x*x for x in p]))
        # p = [x/norm for x in p]
        print("Iteration " + str(k+1) + ": " + str(p) + " " + str(min(mplus, mminus)))
        print()
    return normalize(p)

if __name__ == "__main__":
    print("Final Result:", SPSA(p = [-0.6676757011739746, -0.31034700285664457, -0.32138862669390333, 0.0009965842688914792, -0.467225432474488, -0.36919188585391305]))