# coding: utf-8
import math

states = ('Healthy', 'Fever')

observations = ('dizzy', 'dizzy', 'cold', 'dizzy', 'cold', 'cold', 'normal')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6},
   }

emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }


# Helps visualize the steps of Viterbi.
def print_dptable(V):
    print "    ",
    for i in range(len(V)): print "%7d" % i,
    print

    for y in V[0].keys():
        print "%7s: " % y,
        for t in range(len(V)):
            print "%.7s" % ("%f" % V[t][y]),
        print


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        # V[0][y] = start_p[y] * emit_p[y][obs[0]]
        V[0][y] = math.log(start_p[y]) + math.log(emit_p[y][obs[0]])
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max([(V[t-1][y0] + sum(map( math.log, (trans_p[y0][y], emit_p[y][obs[t]]))), y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    print_dptable(V)
    (prob, state) = max([(V[-1][y], y) for y in states])
    return (prob, path[state])

def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)

print "Observations: %s" % ' -> '.join(observations)
prob, path = example()
print "Max prob: %.5f" % prob
print "Chain: %s" % ' -> '.join(path)
