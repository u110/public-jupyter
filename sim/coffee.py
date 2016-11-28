# utf-8

import simpy
import numpy as np
import pandas as pd

SIM_SIZE = 10 #8 * 60

class Person(object):
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def is_need_to_go(self):
        if np.random.rand() <= 1/3:
                print("%s\t%2d\tコーヒー飲みたい！" % (self.name, self.env.now))
                return True
        else:
                print("%s\t%2d\t仕事しよ。。" % (self.name, self.env.now))
                return False

    def run(self):
        while True:
            if self.is_need_to_go():
                with coffee_garden.request() as req:
                    yield req
                    yield self.env.timeout(3)
                    print("%s\t%2d\tコーヒー美味しい😊" % (self.name, self.env.now))
            yield self.env.timeout(1)

def monitor(resource,env):
    while True:
        print("Dropping\t%2d\t%2d" % (resource._env.now, resource.count))
        print("Waiting\t%2d\t%2d" % (resource._env.now, len(resource.queue)))
        yield env.timeout(1)


env = simpy.Environment()
coffee_garden = simpy.Resource(env, 2)
env.process(Person("Taro",env).run())
env.process(Person("Jiro",env).run())
env.process(Person("Sabr",env).run())
env.process(Person("Ciro",env).run())
env.process(monitor(coffee_garden, env))

env.run(until=SIM_SIZE)


