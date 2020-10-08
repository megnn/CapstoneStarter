#Working on refactoring code from the link below:
#http://www.degeneratestate.org/posts/2018/Jul/10/causal-inference-with-python-part-2-causal-graphical-models/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def f1():
    return np.random.binomial(n=1, p=0.3)


def f2(x1):
    return np.random.normal(loc=x1, scale=0.1)


def f3(x2):
    return x2 ** 2


x1 = f1()
x2 = f2(x1)
x3 = f3(x2)

print("x1 = {}, x2 = {:.2f}, x3 = {:.2f}".format(x1, x2, x3))

from causalgraphicalmodels import CausalGraphicalModel

sprinkler = CausalGraphicalModel(
    nodes=["season", "rain", "sprinkler", "wet", "slippery"],
    edges=[
        ("season", "rain"),
        ("season", "sprinkler"),
        ("rain", "wet"),
        ("sprinkler", "wet"),
        ("wet", "slippery")
    ]
)

# draw return a graphviz `dot` object, which jupyter can render
sprinkler.draw()

print(sprinkler.get_distribution())


from causalgraphicalmodels.examples import fork, chain, collider

print("Implied conditional Independence Relationship: ",
      fork.get_all_independence_relationships())
fork.draw()

print("Implied conditional Independence Relationship: ",
      chain.get_all_independence_relationships())
chain.draw()


print("Implied conditional Independence Relationship: ",
      collider.get_all_independence_relationships())
collider.draw()

path = CausalGraphicalModel(
    nodes = ["x1", "x2", "x3", "x4", "x5"],
    edges = [("x1", "x2"), ("x3", "x2"), ("x4", "x3"), ("x4", "x5")]
)

path.draw()

print("Are x1 and x5 unconditional independent? {} "
      .format(path.is_d_separated("x1", "x5", {})))

print("Are x1 and x5 conditional independent when conditioning on x2? {} "
      .format(path.is_d_separated("x1", "x5", {"x2"})))

print("Are x1 and x5 conditional independent when conditioning on x2 and x3? {} "
      .format(path.is_d_separated("x1", "x5", {"x2", "x3"})))


sprinkler.get_all_independence_relationships()

sprinkler_do = sprinkler.do("rain")

print(sprinkler_do.get_distribution())

sprinkler_do.draw()

from causalgraphicalmodels.examples import simple_confounded

simple_confounded.draw()

simple_confounded.do("x").draw()



from causalgraphicalmodels.examples import big_csm

example_cgm = big_csm.cgm
example_cgm.draw()

example_cgm.get_all_backdoor_paths("x", "y")

example_cgm.is_valid_backdoor_adjustment_set("x", "y", {"b", "d", "e"})

example_cgm.is_valid_backdoor_adjustment_set("x", "y", {"b", "d", "e", "h"})

example_cgm.get_all_backdoor_adjustment_sets("x", "y")

from causalgraphicalmodels.examples import simple_confounded_potential_outcomes

simple_confounded_potential_outcomes.draw()

big_csm.cgm.draw()

big_csm.sample(5)

from causalinference import CausalModel


def estimate_ate(dataset, adjustment_set=None, method="matching"):
    """
    Estimate the ATE of X on Y from from dataset when
    adjusting using adjustment_set.

    Arguments
    ---------
    dataset: pd.DateFrame
        dataframe of observations

    adjustment_set: iterable of variables or None

    method: str
        adjustment method to use.
    """

    if adjustment_set is None:
        y0 = dataset.loc[lambda df: df.x == 0].y.mean()
        y1 = dataset.loc[lambda df: df.x == 1].y.mean()

        y0_var = dataset.loc[lambda df: df.x == 0].y.var()
        y1_var = dataset.loc[lambda df: df.x == 1].y.var()

        y0_n = dataset.loc[lambda df: df.x == 0].shape[0]
        y1_n = dataset.loc[lambda df: df.x == 1].shape[0]

        return {
            "ate": y1 - y0,
            "ate_se": 2 * np.sqrt(y0_var / y0_n + y1_var / y1_n)
        }

    cm = CausalModel(
        Y=dataset.y.values,
        D=dataset.x.values,
        X=dataset[adjustment_set].values
    )
    #TODO
    cm.est_via_ols()
    cm.est_via_matching()
    cm.est_propensity_s()
    cm.est_via_weighting()

    cm.stratify_s()
    cm.est_via_blocking()

    results = {
        "ate": cm.estimates[method]["ate"],
        "ate_se": cm.estimates[method]["ate_se"]
    }

    return results


n_samples = 10000

ds = big_csm.sample(n_samples)

# this allows us to generate samples from an interventional distribution
# where the value of X is assigned randomly as in an A/B test.
ds_intervention = (
    big_csm
    .do("x")
    .sample(
        n_samples=1000000,
        set_values={"x": np.random.binomial(p=0.5, n=1, size=1000000)})
)

true_ate = estimate_ate(ds_intervention)["ate"]

# generate results for a number of different adjustment sets
results = {
    "no_adjustment": estimate_ate(ds),
    "adjustment_b": estimate_ate(ds, ["b"]),
    "adjustment_bde": estimate_ate(ds, ["b", "d", "e"]),
    "adjustment_bh": estimate_ate(ds, ["b", "h"]),
    "adjustment_bc": estimate_ate(ds, ["b", "c"]),
    "adjustment_everything": estimate_ate(ds, ["a", "b", "c", "d", "e", "f", "h"]),

}
print(results)

# plot the results
x_label = list(results.keys())
x = np.arange(len(x_label))
y = [results[l]["ate"] for l in x_label]
yerr = [results[l]["ate_se"] for l in x_label]

plt.figure(figsize=(10,6))
plt.errorbar(x=x, y=y, yerr=yerr, linestyle="none", capsize=5, marker="o")
plt.xticks(x, x_label, rotation=45, fontsize=16)
plt.title("Estimated ATE Size", fontsize=18)
xmin, xmax = plt.xlim()
plt.hlines(true_ate, xmin, xmax, linestyles="dashed")
plt.show()
