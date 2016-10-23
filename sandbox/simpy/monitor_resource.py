from functools import partial, wraps
import simpy

def patch_resource(resource, pre=None, post=None):

	def get_wrapper(func):

		@wraps(func)
		def wrapper(*args, **kwargs):
			if pre:
				pre(resource)

			ret = func(*args, **kwargs)

			if post:
				post(resource)

			return ret
		return wrapper
	

	for name in ['put', 'get', 'request', 'release']:
		if hasattr(resource, name):
			setattr(resource, name, get_wrapper(getattr(resource, name)))


def monitor(data, resource):
	item = (
		resource._env.now,
		resource.count,
		len(resource.queue),
	)
	data.append(item)

def test_process(env, res):
	with res.request() as req:
		yield req
		yield env.timeout(1)

env = simpy.Environment()

res = simpy.Resource(env, capacity=1)
data = []

monitor = partial(monitor, data)
patch_resource(res, post=monitor)

p = env.process(test_process(env, res))
env.run(p)

print(data)
