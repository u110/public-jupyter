# utf-8

# https://github.com/coin-or/pulp
# http://www.fujilab.dnj.ynu.ac.jp/lecture/system2.pdf


"""
あるレストランで，手持ちの材料からハンバーグとオムレツを
作って利益を最大にしたいと考えている．手持ちの材料は，
• ひき肉 3800 [g]
• タマネギ 2100 [g]
• ケチャップ 1200 [g]
であり，それぞれの品を作るのに必要な材料の量は，
• ハンバーグ 1 個あたり，ひき肉 60 [g]，タマネギ 20 [g]，ケチャップ 20 [g]
• オムレツ 1 個あたり，ひき肉 40 [g]，タマネギ 30 [g]，ケチャップ 10 [g]
であるとする．(他に必要な材料は十分な量があるものとする)
販売価格は，
• ハンバーグ 400 [円/個]
• オムレツ 300 [円/個]
とする．総売上を最大にするには，それぞれハンバーグとオムレツを幾つずつ作れば良いか?
"""

from pulp import LpProblem, LpMaximize, LpVariable, value


m = LpProblem(sense=LpMaximize) # 数理モデル
x = LpVariable('x', lowBound=0) # 変数
y = LpVariable('y', lowBound=0) # 変数

m += 400 * x + 300 * y # 目的関数
m += 60 * x + 40 * y <= 3800 # 制約条件
m += 20 * x + 30 * y <= 2100 # 制約条件
m += 20 * x + 10 * y <= 1200 # 制約条件

m.solve()

total_sales = 400 * x.value() + 300 * y.value()
print("ハンバーグ: {}, オムレツ: {}, 売上: {}".format(
    x.value(),
    y.value(),
    total_sales
))
