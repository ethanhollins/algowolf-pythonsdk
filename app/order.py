import math
import time
import json
import requests

class Order(dict):

	def __init__(self, broker, order_id, account_id, product, order_type, direction, lotsize, entry_price=None, sl=None, tp=None, open_time=None):
		
		# Hidden Variable
		super().__setattr__('_broker', broker)

		# Dictionary Variables
		self.order_id = order_id
		self.account_id = account_id
		self.product = product
		self.direction = direction
		self.lotsize = lotsize
		self.order_type = order_type
		self.entry_price = entry_price
		self.close_price = None
		self.sl = sl
		self.tp = tp

		if open_time:
			self.open_time = int(open_time)
		else:
			self.open_time = math.floor(time.time())


	@classmethod
	def fromDict(cls, broker, pos):
		# Hidden Variable
		res = cls(
			broker,
			pos['order_id'],
			pos['account_id'],
			pos['product'],
			pos['order_type'],
			pos['direction'],
			pos['lotsize']
		)
		# Dictionary Variables
		for k, v in pos.items():
			res.__setattr__(k, v)

		return res


	def __getattribute__(self, key):
		try:
			return super().__getattribute__(key)
		except AttributeError as e:
			try:
				return self[key]
			except KeyError:
				pass
			raise e


	def __setattr__(self, key, value):
		self[key] = value


	def __str__(self):
		cpy = self.copy()
		cpy['open_time'] = int(cpy['open_time'])
		return json.dumps(cpy, indent=2)


	def cancel(self):

		endpoint = f'/v1/strategy/{self._broker.strategyId}/brokers/{self._broker.brokerId}/orders'
		payload = {
			"items": [{
				"order_id": self.order_id
			}]
		}

		res = self._broker._session.delete(
			self._broker._url + endpoint,
			data=json.dumps(payload)
		)

		result = []
		status_code = res.status_code
		if status_code == 200:
			res = res.json()

			for ref_id, item in res.items():
				if item.get('accepted'):
					func = self._broker._get_trade_handler(item.get('type'))
					wait_result = self._broker._wait(ref_id, func, (ref_id, item))

					result.append(
						EventItem({
							'reference_id': ref_id,
							'accepted': True,
							'type': item.get('type'),
							'item': wait_result
						})
					)

				else:
					result.append(
						EventItem({
							'reference_id': ref_id,
							'accepted': False,
							'type': item.get('type'),
							'message': item.get('message'),
							'item': item.get('item')
						})
					)

		else:
			print(f'{res.status_code}: {res.text}', flush=True)


		return result


	def close(self):
		return self.cancel()


	def modify(self, 
		lotsize=None,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		endpoint = f'/v1/strategy/{self._broker.strategyId}/brokers/{self._broker.brokerId}/orders'

		if not lotsize is None: 
			lotsize = self._broker._convert_lotsize(lotsize)
		payload = {
			"items": [{
				"order_id": self.order_id,
				"entry_range": entry_range,
				"entry_price": entry_price,
				"sl_range": sl_range,
				"tp_range": tp_range,
				"sl_price": sl_price,
				"tp_price": tp_price,
				"lotsize": lotsize
			}]
		}

		res = self._broker._session.put(
			self._broker._url + endpoint,
			data=json.dumps(payload)
		)

		result = []
		status_code = res.status_code
		if status_code == 200:
			res = res.json()

			for ref_id, item in res.items():
				if item.get('accepted'):
					func = self._broker._get_trade_handler(item.get('type'))
					wait_result = self._broker._wait(ref_id, func, (ref_id, item))

					result.append(
						EventItem({
							'reference_id': ref_id,
							'accepted': True,
							'type': item.get('type'),
							'item': wait_result
						})
					)

				else:
					result.append(
						EventItem({
							'reference_id': ref_id,
							'accepted': False,
							'type': item.get('type'),
							'message': item.get('message'),
							'item': item.get('item')
						})
					)
					
		else:
			print(f'{res.status_code}: {res.text}', flush=True)

		return result


	def modifyEntry(self, entry_range=None, entry_price=None):
		return self.modify(entry_range=entry_range, entry_price=entry_price)


	def modifySL(self, sl_range=None, sl_price=None):
		return self.modify(sl_range=sl_range, sl_price=sl_price)


	def modifyTP(self, tp_range=None, tp_price=None):
		return self.modify(tp_range=tp_range, tp_price=tp_price)


	def isBacktest(self):
		return False




class BacktestOrder(Order):


	def cancel(self):
		return self._broker.backtester.deleteOrder(self)


	def modify(self,
		lotsize=None,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		return self._broker.backtester.modifyOrder(
			self, lotsize, entry_range, entry_price,
			sl_range, tp_range, sl_price, tp_price
		)


	def isBacktest(self):
		return True



'''
Imports
'''
from .. import app as tl
from ..app.error import BrokerException
from .broker import EventItem



