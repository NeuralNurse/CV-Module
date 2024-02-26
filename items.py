import time


class Item:
	def __init__(self, item, side, limit=30):
		self.item = item
		self.side = side
		self.start = time.perf_counter()
		self.limit = limit
		self.hash = self.item + self.side

	def check_time(self):
		diff = time.perf_counter() - self.start
		return diff > self.limit

n = Item("bob", "t")
