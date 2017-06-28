import cvxpy
import numpy as np 

class MySolver(object):
    def __init__(self, budget=None):
        self.budget = budget
        self.x = {}
        self.payouts = {}
        self.amounts = {}
        self.obj = None
        self.problem = None
        self.consts = []

    def add(self, key, payouts, amounts):
        self.payouts[key] = payouts
        self.amounts[key] = amounts

    def _setup(self):
        all_payouts = None
        for k in self.payouts.keys():
            self.x[k] = cvxpy.Bool(self.payouts[k].shape[0])
            self.consts.append(
                    cvxpy.norm1(self.x[k]) <= 1  # このうち選ぶのはひとつのみ
            )
            if all_payouts is None:
                all_payouts = self.payouts[k] * self.x[k]
            else:
                all_payouts += self.payouts[k] * self.x[k]
            if self.obj is None:
                self.obj = self.amounts[k] * self.x[k]
            else:
                self.obj += self.amounts[k] * self.x[k]

        self.obj = cvxpy.Maximize(self.obj)             # 合計売上の最大化
        self.consts.append(self.budget >= all_payouts)  # 合計費用の予算上限の制約

    def solve(self):
        self._setup()
        self.problem = cvxpy.Problem(self.obj, self.consts)
        self.problem.solve(solver=cvxpy.ECOS_BB)        # MIP 混合整数問題用ソルバ
        return self.problem, self.x

    def payouts(self):
        res = 0
        for k in self.payouts.keys():
            res += self.payouts[k] * self.x
        return res


if __name__ == "__main__":
    a = MySolver(budget=500)
    a.add("frame_1", np.array([100, 200, 300]), np.array([20, 35, 25]))
    a.add("frame_2", np.array([100, 200, 300]), np.array([20, 30, 28]))
    a.add("frame_3", np.array([100, 200, 300]), np.array([23, 30, 25]))
    a.solve()

    selected = {}
    payouts = 0
    for k in a.x.keys():
        selected[k] = [round(ix[0, 0]) for ix in a.x[k].value]
        print(k, selected[k])
        payouts += np.sum(selected[k] * a.payouts[k])

    print("optimal amount: ", a.problem.value)
    print("capped payout : ", payouts)

