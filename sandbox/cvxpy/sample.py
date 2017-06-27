# utf-8

import cvxpy
import numpy as np

#  frame_payout = np.array([22, 12, 19, 10, 35, 32, 42, 53])  # 消化金額
#  frame_amount = np.array([22, 12, 16, 10, 35, 26, 42, 53])  # 売上
#  campaign_budget = 100                                      # 予算
#  
#  x = cvxpy.Bool(frame_payout.shape[0])  # 変数の設定
#  objective = cvxpy.Maximize(frame_amount * x)
#  aaa = frame_payout * x,
#  constraints = [
#      campaign_budget >= sum(aaa),
#      cvxpy.norm1(x) <= 1  # x はどれかひとつだけしか選べない
#  ]
#  
#  
#  prob = cvxpy.Problem(objective, constraints)
#  prob.solve(solver=cvxpy.ECOS_BB)
#  
#  result = [round(ix[0, 0]) for ix in x.value]
#  
#  print("status:", prob.status)
#  print("optimal value", prob.value)
#  print("result x:", result)

class MySolver(object):
    def __init__(self, budget=None):
        self.budget = budget
        self.x = {}
        self.payouts = {}
        self.amounts = {}
        self.obj= None
        self.prob= None
        self.consts= []

    def add(self, key, payouts, amounts):
        self.payouts[key] = payouts
        self.amounts[key] = amounts

    def solve(self):
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
                self.obj+= self.amounts[k] * self.x[k]

        self.obj= cvxpy.Maximize(self.obj)
        self.consts.append(
                self.budget >= all_payouts
        )
        self.problem = cvxpy.Problem(self.obj, self.consts)
        self.problem.solve(solver=cvxpy.ECOS_BB)
        return self.problem, self.x


if __name__ == "__main__":
    a = MySolver(budget=100)
    a.add("frame_1", np.array([1,2,3]), np.array([20,30,25]))
    a.add("frame_2", np.array([1,2,3]), np.array([20,30,25]))
    a.add("frame_3", np.array([1,2,3]), np.array([20,30,25]))
    a.solve()
    print(a.x["frame_1"].value)
