import numpy as np
import pulp
import matplotlib.pyplot as plt
import pandas as pd

varcount = 0

EPS = 0.0000005


def get_neg(bkt):
    neg = 0
    for prob, dgp in bkt:
        neg = neg+(1 - prob) * dgp
    return neg


def get_pos(bkt):
    pos = 0
    for prob, dgp in bkt:
        pos = pos + prob * dgp
    return pos


def resource_monotonicity(k, bkt_gr):
    poss = [get_pos(bkt) for bkt in bkt_gr]
    negs = [get_neg(bkt) for bkt in bkt_gr]

    totp = sum(poss)
    totn = sum(negs)

    xs = []
    ys = []
    xcods = []
    ycods = []
    for bkts, pos, neg in zip(bkt_gr, poss, negs):
        xcod, ycod = solve(k, bkts, pos, neg, totp - pos, totn - neg)
        xcods.append(xcod)
        ycods.append(ycod)
        xs.append(xcod / pos)
        ys.append(ycod / neg)

    optx = min(xs)
    alcns = []
    for pos, neg, x, xcod, ycod, bkts in zip(poss, negs, xs, xcods, ycods, bkt_gr):
        thld = []
        card = xcod + ycod
        rem = card
        for (prob, dgp) in bkts:
            thld.append(min(rem, dgp))
            rem -= min(rem, dgp)

        urate = k / (totp + totn)
        if x <= urate + EPS:
            rho = 0
        else:
            rho = (optx - urate) / (x - urate)

        alcn = []
        for (prob, dgp), thld_alloc in zip(bkts, thld):
            alcn.append((prob, rho * thld_alloc + (1 - rho) * urate * dgp))

        alcns.append(alcn)

    return alcns


def solve(k, bkts, tp, tn, opos, oneg):
    xcod = 0
    ycod = 0
    for (prob, dgp) in bkts:
        dx = prob * dgp
        dy = (1 - prob) * dgp

        k_prev = (xcod * (1 + opos / tp) + ycod * (1 + oneg / tn))
        k_mid = (dx * (1 + opos / tp) + dy * (1 + oneg / tn))

        k_new = k_prev + k_mid

        if k_new <= k:
            xcod += dx
            ycod += dy
        else:
            needed_increase = k - k_prev
            xcod += dx * needed_increase / k_mid
            ycod += dy * needed_increase / k_mid

    return xcod, ycod


def unfair(k, bkt_gr):
    tot_bkt = [pair for bkt in bkt_gr for pair in bkt]
    tot_bkt.sort(reverse=True)
    rem = k
    eff = 0

    for prob, bkt_size in tot_bkt:
        al = min(rem, bkt_size)
        eff += prob * al
        rem -= al
        if rem <= EPS:
            return eff

    print("Sonething wrong in unfair()")
    exit()


def cleanup_frame(frame):
    frame = frame.rename(columns={'Non- Hispanic white': 'White'})
    frame = frame.reindex(['Asian', 'Black', 'Hispanic', 'White'], axis=1)
    return frame


def lpcons(gr_bkt, kcons):
    global varcount
    num_groups = len(gr_bkt)

    lpprob = pulp.LpProblem("ValueFn", pulp.LpMaximize)

    allcns = [[] for _ in range(num_groups)]
    allcn_sum = 0
    wpb = [0 for _ in range(num_groups)]
    wntpb = [0 for _ in range(num_groups)]
    wpbar = [0 for _ in range(num_groups)]
    wntpbar = [0 for _ in range(num_groups)]

    for group in range(num_groups):
        for prob, group_size in gr_bkt[group]:
            var = pulp.LpVariable(name= f"V{varcount}",upBound=group_size)
            allcns[group].append((prob, var))
            varcount+=1
            allcn_sum += var
            wpb[group] += group_size * prob
            wntpb[group] += group_size * (1 - prob)
            wpbar[group] += var * prob
            wntpbar[group] += var * (1 - prob)

    lpprob += allcn_sum == kcons

    for group in range(num_groups):
        
        lpprob += wpbar[0] * wpb[group] == wpbar[group] * wpb[0]
        
        lpprob += wntpbar[0] * wntpb[group] == wntpbar[group] * wntpb[0]
    return lpprob, allcns, wpbar


def population_monotonicity(gr_bkt, kcons):
    global varcount
    lpprob, allcns, wpbar = lpcons(gr_bkt, kcons)

    num_groups = len(gr_bkt)
    grp_len = [sum(group_size for _, group_size in clf)
                   for clf in gr_bkt]
    for group in range(num_groups):
        for (_, group_size), (_, var) in zip(gr_bkt[group], allcns[group]):
            agents_after_removal = group_size + sum(grp_len[:group]) + sum(grp_len[group+1:])
            lpprob += var * agents_after_removal <= kcons * group_size

    lpprob += sum(wpbar)
    lpprob.solve(pulp.PULP_CBC_CMD(msg=0))
    varcount = 0
    return lpprob


if __name__ == "__main__":

    risk = cleanup_frame(pd.read_csv("data/transrisk_performance_by_race_ssa.csv"))
    totals = cleanup_frame(pd.read_csv("data/totals.csv"))
    cdf = cleanup_frame(pd.read_csv("data/transrisk_cdf_by_race_ssa.csv"))

    N = sum(totals[race][0] for race in totals.columns)

    grp_bkt = []
    for race in risk.columns:
        tot = totals[race][0]
        buckets = []
        old_cd = 0
        for default_percent, cd in zip(risk[race], cdf[race]):
            buckets.append((1 - float(default_percent)/100, int(cd * tot / 100) - int(old_cd * tot / 100))) 
            old_cd = cd
        buckets = [(b1, b2) for b1, b2 in buckets if b2 > 0]
        buckets.sort(reverse=True)
        grp_bkt.append(buckets)

    effdata = []

    for k in range(1, int(N + 1), 10):
        print(f"Processing {k*100//N}%", end='\r')
        base = unfair(k, grp_bkt)
        
        reseff = sum(get_pos(alloc)
                  for alloc in resource_monotonicity(k, grp_bkt))
        resdata = {"num_loans": k, "eff_frac": reseff /base, "alg": "Resource Monotonicity"}
        popeff = pulp.value(population_monotonicity(grp_bkt, k).objective)
        popdata = {"num_loans": k,  "eff_frac": popeff /
                 base, "alg": "Population Monotonicity"}
        coneff = k / N * sum(get_pos(bucket) for bucket in grp_bkt)
        condata = {"num_loans": k, 
                 "eff_frac": coneff / base, "alg": "Consistency"}
        effdata.append(resdata)
        effdata.append(popdata)
        effdata.append(condata)

    datadf = pd.DataFrame(effdata)
    datadf.to_csv("result.csv", index=False)

    temp1 = datadf[datadf['alg'] == 'Resource Monotonicity']
    num_loans = np.array(temp1['num_loans'])
    res = np.array(temp1["eff_frac"])
    popl_mon = datadf[datadf['alg'] ==
                    'Population Monotonicity']["eff_frac"]
    consis = datadf[datadf['alg'] ==
                    'Consistency']["eff_frac"]
    plt.figure(figsize=(10, 5))
    plt.plot(num_loans, res, label="Resource Monotonicity", color='blue')
    plt.plot(num_loans, popl_mon, label="Population Monotonicity", color='green')
    plt.plot(num_loans, consis, label="Consistency", color='orange')
    plt.xlabel("num_loans")
    plt.ylabel("Efficiency fraction of Unfair Allocation")
    plt.legend(bbox_to_anchor=(0.75, 1.15), ncol=2)
    plt.savefig("figure.jpg")
    plt.show()
    
