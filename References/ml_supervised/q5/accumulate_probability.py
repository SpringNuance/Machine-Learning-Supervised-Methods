import scipy.special
import math

def accumulate_probability(leaners, error):
    major = math.ceil(leaners / 2)
    sum = 0
    for k in range(major, leaners):
        sum += scipy.special.binom(leaners, k) * (error**k) * ((1-error)**(leaners-k))
    return sum

print(accumulate_probability(15, 0.4))
