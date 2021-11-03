import numpy as np
import pandas as pd
import datetime
import requests
import time
import json
import os
import signal
import shortuuid
import traceback
import math
import ntplib
from copy import copy
from enum import Enum
from .. import app as tl
from .app import App
from io import BytesIO
from redis import Redis

'''
Broker Names
'''

BACKTEST_NAME = 'backtest'
IG_NAME = 'ig'
OANDA_NAME = 'oanda'
FXCM_NAME = 'fxcm'
SPOTWARE_NAME = 'spotware'
FXOPEN_NAME = 'fxopen'
PAPERTRADER_NAME = 'papertrader'

def get_list():
	return [
		OANDA_NAME,
		FXCM_NAME,
		IG_NAME,
		SPOTWARE_NAME,
		FXOPEN_NAME
	]

class State(Enum):
	IDLE = 0
	BACKTEST = 1
	BACKTEST_AND_RUN = 2
	LIVE = 3
	STOPPED = 4


class BacktestMode(Enum):
	RUN = 'run'
	STEP = 'step'


class EventItem(dict):

	def __getattr__(self, key):
		return self[key]

	def __setattr__(self, key, value):
		self[key] = value

	def __str__(self):
		return json.dumps(self, indent=2)


'''
Parent Broker Class
'''
class Broker(object):

	def __init__(self, app, strategy, strategy_id=None, broker_id=None, account_id=None, data_path='data/'):
		self._app = app
		self.strategy = strategy
		self.strategyId = strategy_id
		self.brokerId = broker_id
		self.accountId = account_id

		self.name = None
		self.state = State.IDLE
		self.backtester = None
		self.isUploadBacktest = True
		self.isClearBacktestPositions = False
		self.isClearBacktestOrders = False
		self.isClearBacktestTrades = False
		self._start_from = None
		self._data_path = data_path

		self.redis_client = Redis(host='redis', port=6379, password="dev")

		# Containers
		self.charts = []
		# self.positions = []
		# self.orders = []
		self.ontrade_subs = []
		self.onrejected_subs = []
		self.onsessionstatus_subs = []
		self.handled = {}

		self.backtest_positions = []
		self.backtest_orders = []

		self.time_off = 0
		self._set_time_off()

		self._session = requests.Session()
		self._session.headers.update({
			'Authorization': 'Bearer '+self._app.key,
			'Connection': 'keep-alive',
			'Content-Type': 'application/json'
		})
		self._url = self._app.config.get('API_URL')


	def __getattribute__(self, key):

		if key == 'positions':
			return self.getAllPositions()
		elif key == 'orders':
			return self.getAllOrders()

		return super().__getattribute__(key)


	'''
	Run Utilities
		- Functions that start backtest or live runs
	'''

	def _set_time_off(self):
		try:
			client = ntplib.NTPClient()
			response = client.request('pool.ntp.org')
			self.time_off = response.tx_time - time.time()
		except Exception:
			pass

	# TODO: DO TOKENS INSTEAD and VALIDATION
	def run(self):

		strategy_info = self._initialize_strategy()
		print(strategy_info['brokers'][self.brokerId]['broker'], flush=True)
		self.setName(strategy_info['brokers'][self.brokerId]['broker'])

		self._app.sio.on('connect', handler=self._stream_connect, namespace='/user')
		self._app.sio.on('disconnect', handler=self._stream_disconnect, namespace='/user')
		# self._app.sio.on('ontick', handler=self._stream_ontick, namespace='/user')
		# self._app.sio.on('ontrade', handler=self._stream_ontrade, namespace='/user')
		self._app.sio.on('onsessionstatus', handler=self._stream_onsessionstatus, namespace='/user')

		self._app.connectSio()

		# Subscribe all charts
		self._subscribe_charts(self.charts)

		# If `_start_from` set, run `_backtest_and_run`
		backtest_complete = False
		if self._start_from is not None:
			backtest_complete = self._backtest_and_run(self._start_from, quick_download=True)
			print(f'Backtest Complete: {backtest_complete}', flush=True)

		if not backtest_complete:
			end = datetime.datetime.utcnow()
			# for chart in self.charts:
			#   for period in chart.periods:
			# start = tl.utils.getCountDate(period, 1000, end=end)
			self._collect_data(end, end, download=False, quick_download=True)


		# self.updateAllPositions()
		# self.updateAllOrders()

		self._app.sendScriptRunning()
		print('LIVE', flush=True)
		self._prepare_for_live()
		self.state = State.LIVE
		print(f"LIVE? {self.state}", flush=True)

	def stop(self):
		print('STOPPED', flush=True)
		self.state = State.STOPPED

		# for product in self._chart_subs:
		#   api_chart = self.api.getChart(product)
		#   for period in self._chart_subs[product]:
		#       if period in self._chart_subs[api_chart.product]:
		#           for sub_id in self._chart_subs[api_chart.product][period]:
		#               api_chart.unsubscribe(period, self.brokerId, sub_id)
		try:
			self._app.sendScriptStopped()
			time.sleep(5)
			# self._app.sio.disconnect()
		except:
			pass
		finally:
			PID = os.getpid()
			# os.kill(PID, signal.SIGTERM)
			os.kill(PID, signal.SIGKILL)


	def startFrom(self, dt):
		if tl.utils.isOffsetAware(dt):
			self._start_from = tl.utils.convertTimezone(dt, 'UTC')
		else:
			self._start_from = tl.utils.setTimezone(dt, 'UTC')


	def uploadBacktest(self, is_upload=True):
		self.isUploadBacktest = is_upload


	def setClearBacktestPositions(self, is_clear=True):
		self.isClearBacktestPositions = is_clear


	def setClearBacktestOrders(self, is_clear=True):
		self.isClearBacktestOrders = is_clear


	def setClearBacktestTrades(self, is_clear=True):
		self.isClearBacktestTrades = is_clear


	def _clear_backtest_positions(self):
		for i in range(len(self.positions)-1,-1,-1):
			pos = self.positions[i]
			if pos.isBacktest():
				del self.positions[i]


	def _clear_backtest_orders(self):
		for i in range(len(self.orders)):
			order = self.orders[i]
			if order.isBacktest():
				del self.orders[i]


	def _clear_backtest_trades(self):
		self._clear_backtest_positions()
		self._clear_backtest_orders()


	def _perform_backtest(self, start, end, spread=None, mode=BacktestMode.RUN, download=True, quick_download=False, process_mode=None):
		# Collect relevant data
		# self._collect_data(start, end, download=download, quick_download=quick_download)

		# Run backtest
		return self.backtester.performBacktest(mode.value, start=start, end=end, spread=spread, process_mode=process_mode)


	def _generate_backtest(self, start, end):
		# Generate backtest dict
		reports = self.strategy.reports
		result = {
			'transactions': self.backtester.result,
			'reports': { name:report.to_dict() for name, report in reports.items()},
			'info': self.backtester.info,
			'properties': {
				'broker': self.name,
				'start': tl.convertTimeToTimestamp(start),
				'end': tl.convertTimeToTimestamp(end)
			}
		}

		# Reset reports
		self.strategy.resetReports()

		return result


	def _generate_live_backtest(self):
		result = {
			'transactions': self.backtester.result,
			'info': self.backtester.info
		}

		self.strategy.resetReports()
		self.backtester.result = []
		self.backtester.info = []

		return result


	def upload(self, endpoint, payload):
		# Upload Backtest
		file_name = self.generateReference()
		file = BytesIO(payload)

		blocksize = 1024 * 1024
		chunkindex = 0
		chunkbyteoffest = 0
		totalfilesize = file.getbuffer().nbytes
		totalchunkcount = math.ceil(totalfilesize / blocksize)

		while chunkindex < totalchunkcount:
			headers = {
				'Filename': f'{file_name}.json',
				'Chunkindex': str(chunkindex),
				'Chunkbyteoffset': str(chunkbyteoffest),
				'Totalfilesize': str(totalfilesize),
				'Totalchunkcount': str(totalchunkcount)
			}
			data = file.read(blocksize)

			res = self._session.post(
				self._url + endpoint,
				data=data, headers=headers
			)

			if res.status_code != 200:
				print(f'Failed backtest upload, {res.text}', flush=True)
				return res

			chunkindex += 1
			chunkbyteoffest += blocksize

		print('upload complete', flush=True)
		return res


	def backtest(self, start, end, spread=None, mode=BacktestMode.RUN, upload=False, download=True, quick_download=False, process_mode=None):
		self.state = State.BACKTEST

		self._perform_backtest(start, end, spread=spread, mode=mode, download=download, quick_download=quick_download, process_mode=process_mode)

		# Upload Backtest
		endpoint = f'/v1/strategy/{self.strategyId}/backtest'
		payload = json.dumps(self._generate_backtest(start, end)).encode()
		res = self.upload(endpoint, payload)

		# Retrieve Backtest ID
		backtest_id = res.json().get('backtest_id')
		print(backtest_id, flush=True)
		return backtest_id


	def _backtest_and_run(self, start, quick_download=False):
		self.state = State.BACKTEST_AND_RUN
		# Collect relevant data and connect to live broker
		end = tl.convertTimestampToTime(time.time() + self.time_off)
		print(f'Start: {start}, End: {end}')
		backtest_complete = self._perform_backtest(start, end, spread=0.0, quick_download=quick_download, process_mode=None)

		if backtest_complete:
			if self.isUploadBacktest:
				# Upload Backtest
				endpoint = f'/v1/strategy/{self.strategyId}/{self.brokerId}/{self.accountId}/backtest'
				payload = json.dumps(self._generate_live_backtest()).encode()

				start = time.time()
				print(f'Uploading live backtest... {len(payload)}')
				res = self.upload(endpoint, payload)
				print(f'Live upload done {round(time.time() - start, 2)}s')

		# Clear backtest trades
		self._clear_backtest_trades()

		return backtest_complete


	def isBacktest(self):
		return self.state == State.BACKTEST


	def isBacktestAndRun(self):
		return self.state == State.BACKTEST_AND_RUN


	def isLive(self):
		return self.state == State.LIVE


	def _collect_data(self, start, end, download=True, quick_download=False):
		for chart in self.charts:
			for period in chart.periods:
				if period != tl.period.TICK:
					if quick_download:
						chart.quickDownload(
							period, 
							tl.utils.getCountDate(period, 1000, end=start), end
						)
					else:
						chart.getPrices(
							period, 
							start=tl.utils.getCountDate(period, 1000, end=start), 
							end=end,
							download=download
						)

	'''
	Utilities
	'''
	def generateReference(self):
		return shortuuid.uuid()

	def setName(self, name):
		self.name = name
		if self.name in (OANDA_NAME, FXCM_NAME, SPOTWARE_NAME, PAPERTRADER_NAME, FXOPEN_NAME, 'loadtest'):
			# self.backtester = tl.OandaBacktester(self)
			self.backtester = tl.IGBacktester(self)
		elif self.name in (IG_NAME,):
			self.backtester = tl.IGBacktester(self)


	def _initialize_strategy(self):
		# endpoint = f'/v1/strategy/{self.strategyId}/init/{self.brokerId}'
		# res = self._session.post(
		# 	self._url + endpoint
		# )

		# if res.status_code == 200:
		# 	return res.json()
		# else:
		# 	print(res.json(), flush=True)

		msg_id = self.generateReference()
		self._app.sendRequest({
			"type": "request",
			"timeout": 300,
			"message": {
				"msg_id": msg_id,
				"ept": "init_strategy_by_broker_id_ept",
				"args": [self.strategyId, self.brokerId],
				"Authorization": "Bearer " + self._app.key
			}
		})
		res = self._app.waitResponse(msg_id, timeout=300)
		print(f"[_initialize_strategy] {res}", flush=True)
		if not "error" in res:
			return res["result"]
		else:
			return self._initialize_strategy()


	def _subscribe_charts(self, charts):
		for chart in charts:
			chart.connectAll()


	def _prepare_for_live(self):
		# for i in self.positions + self.orders:
		#   if not any([chart.isChart(self.name, i.product) for chart in self.charts]):
		#       chart = self.getChart(i.product, tl.period.TICK, broker=self.name)
		#       chart.connectAll()
		if self.name == PAPERTRADER_NAME:
			name = FXCM_NAME
		else:
			name = self.name

		for chart in self.charts:
			# Setup tick charts for user broker
			# if not any([chart.isChart(name, chart.product) for chart in self.charts]):
			# 	chart = self.getChart(chart.product, tl.period.TICK, broker=name)
			# 	chart.connectAll()

			chart.prepareLive()
			if chart.broker == name:
				period = chart.getLowestPeriod()
				if period is not None:
					chart.subscribe(period, self._handle_tick_checks)


	def _handle_tick_checks(self, item):
		if item.chart.broker == self.name:
			product = item.chart.product
			timestamp = int(item.timestamp)

			if isinstance(item.ask, list):
				ohlc = np.array([item.ask[3]]*4 + [item.mid[3]]*4 + [item.bid[3]]*4, dtype=np.float64)
			else:
				ohlc = np.array([item.ask]*4 + [item.mid]*4 + [item.bid]*4, dtype=np.float64)

			self.backtester.handleOrders(product, timestamp, ohlc)
			self.backtester.handleStopLoss(product, timestamp, ohlc)
			self.backtester.handleTakeProfit(product, timestamp, ohlc)


	def _create_chart(self, product, *periods, broker='oanda'):
		chart = None
		if isinstance(self._app, App):
			chart = self._app.getChart(broker, product)
			if chart is None:
				chart = tl.Chart(self.strategy, product, broker=broker, data_path=self._data_path)
				self._app.addChart(chart)
		else:
			chart = tl.Chart(self.strategy, product, data_path=self._data_path)

		chart.addPeriods(*periods)
		self.charts.append(chart)
		return chart

	def getAllCharts(self):
		return self.charts

	def getChart(self, product, *periods, broker='oanda'):
		for chart in self.charts:
			if chart.isChart(broker, product):
				chart.addPeriods(*periods)
				return chart

		return self._create_chart(product, *periods, broker=broker)


	def chartExists(self, broker, product):
		for chart in self.charts:
			if chart.isChart(broker, product):
				return True

		return False


	def chartProductExists(self, product):
		for chart in self.charts:
			if chart.product == product:
				return True
		
		return False


	def findChartByProduct(self, product):
		for chart in sorted(self.charts, key=lambda x: x.broker == self.name, reverse=True):
			if chart.product == product:
				return chart


	def getBrokerAsk(self, product):
		chart = self.getApiChart(product)
		return chart.getLatestAsk(tl.period.TICK)


	def getBrokerBid(self, product):
		chart = self.getApiChart(product)
		return chart.getLatestBid(tl.period.TICK)


	def getAsk(self, product):
		chart = self.findChartByProduct(product)
		if not chart is None:
			period = chart.getLowestPeriod()
			if period == tl.period.TICK:
				return chart.asks[period]
			elif period is not None:
				return chart.getLastAskOHLC(period)[3]

		# if self.chartExists(broker, product):
		#   chart = self.getChart(product, broker=broker)
		#   period = chart.getLowestPeriod()
		#   if period is not None:
		#       return chart.getLastAskOHLC(period)[3]
		#   else:
		#       raise tl.error.BrokerException(f'No {product} data found.')
		# else:
		#   raise tl.error.BrokerException(f'Chart {product} doesn\'t exist.')  


	def getBid(self, product):
		chart = self.findChartByProduct(product)
		if not chart is None:
			period = chart.getLowestPeriod()
			if period == tl.period.TICK:
				return chart.bids[period]
			elif period is not None:
				return chart.getLastBidOHLC(period)[3]

		# if self.chartExists(broker, product):
		#   chart = self.getChart(product, broker=broker)
		#   period = chart.getLowestPeriod()
		#   if period is not None:
		#       return chart.getLastBidOHLC(period)[3]
		#   else:
		#       raise tl.error.BrokerException(f'No {product} data found.')
		# else:
		#   raise tl.error.BrokerException(f'Chart {product} doesn\'t exist.')

	def getTimestamp(self, product):
		if self.state == State.LIVE:
			return time.time()
			# if period is None:
			#   period = self.getChart(product, broker=broker).getLowestPeriod()
			# return int(self.getChart(product, broker=broker).getTimestamp(period))

		else:
			return self.findChartByProduct(product)._end_timestamp

	def getBrokerKey(self):
		return self.strategyId + "." + self.brokerId

	def updateAllPositions(self):
		endpoint = f'/v1/strategy/{self.strategyId}/brokers/{self.brokerId}/positions'
		res = self._session.get(
			self._url + endpoint
		)

		status_code = res.status_code
		if status_code == 200:
			data = res.json()
			for account_id in data:
				if account_id == self.accountId:
					self.positions = (
						[pos for pos in self.positions if pos.isBacktest()] +
						[tl.position.Position.fromDict(self, pos) for pos in data[account_id]]
					)
			print(self.positions, flush=True)
		else:
			print(res.json(), flush=True)
			pass

	def getDbPositions(self):
		positions = self.redis_client.hget(self.getBrokerKey(), "positions")
		if positions is None:
			positions = []
		else:
			positions = [
				tl.position.Position.fromDict(self, i) 
				for i in json.loads(positions)
				if i.get('account_id') == self.accountId and self.chartProductExists(i.get("product"))
			]
		return positions

	def getAllPositions(self, account_id=None):
		if self.state == State.LIVE:
			return [
				pos for pos in self.getDbPositions() 
				if not account_id or pos.account_id == account_id
			]
		else:
			return [
				pos for pos in self.backtest_positions
				if not account_id or pos.account_id == account_id
			]

	def getPositionByID(self, order_id):
		for pos in self.getAllPositions():
			if pos.order_id == order_id:
				return pos
		return None

	def updateAllOrders(self):
		endpoint = f'/v1/strategy/{self.strategyId}/brokers/{self.brokerId}/orders'
		res = self._session.get(
			self._url + endpoint
		)

		status_code = res.status_code
		if status_code == 200:
			data = res.json()
			for account_id in data:
				if account_id == self.accountId:
					self.orders = (
						[order for order in self.orders if order.isBacktest()] +
						[tl.order.Order.fromDict(self, order) for order in data[account_id]]
					)
			print(self.orders, flush=True)
		else:
			pass

	def getDbOrders(self):
		orders = self.redis_client.hget(self.getBrokerKey(), "orders")
		if orders is None:
			orders = []
		else:
			orders = [
				tl.order.Order.fromDict(self, i) 
				for i in json.loads(orders)
				if i.get('account_id') == self.accountId and self.chartProductExists(i.get("product"))
			]
		return orders

	def getAllOrders(self, account_id=None):
		if self.state == State.LIVE:
			return [
				order for order in self.getDbOrders() 
				if not account_id or order.account_id == account_id
			]
		else:
			return [
				order for order in self.backtest_orders
				if not account_id or order.account_id == account_id
			]

	def getOrderByID(self, order_id):
		for order in self.orders:
			if order.order_id == order_id:
				return order
		return None


	'''
	Account Utilities
		- All functions access brokerage directly
	'''

	def updateAccountInfo(self):
		info = self.api.getAccountInfo(accounts, override=True)


	def getAccountInfo(self, account_id):
		print(f'GET ACCOUNT INFO: Live -> {self.isLive()}')
		if self.isLive():
			endpoint = f'/v1/strategy/{self.strategyId}/brokers/{self.brokerId}/accounts/{self.accountId}'
			res = self._session.get(
				self._url + endpoint
			)

			status_code = res.status_code
			if status_code == 200:
				return res.json()
			else:
				return None
		else:
			result = {
				account_id: {
					'currency': 'AUD',
					'balance': 10000,
					'pl': 0,
					'margin': 0,
					'available': 1
				}
			}
			return result

	'''
	Dealing Utilities
		- All functions access brokerage directly
	'''

	def _convert_lotsize(self, lotsize):
		# if self.name == SPOTWARE_NAME:
		# 	return int(round(lotsize, 2) * 10000000)
		# elif self.name == OANDA_NAME:
		# 	return int(lotsize * 100000)
		# else:
		return lotsize

	def _convert_incoming_lotsize(self, lotsize):
		# if self.name == SPOTWARE_NAME:
		# 	return round(lotsize / 10000000, 2)
		# elif self.name == OANDA_NAME:
		# 	return round(lotsize / 100000, 5)
		# else:
		return lotsize

	# Broker Functions

	# TODO: ORDERS
	def buy(self,
		product, lotsize, accounts,
		order_type=tl.MARKET_ORDER,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if self.findChartByProduct(product) is None:
			raise Exception('Instrument not loaded.')

		lotsize = self._convert_lotsize(lotsize)

		result = []
		if not self.isLive():
			if order_type == tl.MARKET_ORDER:
				for account_id in accounts:
					result = self.backtester.createPosition(
						product, lotsize, tl.LONG,
						account_id, entry_range, entry_price,
						sl_range, tp_range, sl_price, tp_price
					)

			elif order_type == tl.LIMIT_ORDER or order_type == tl.STOP_ORDER:
				for account_id in accounts:
					result = self.backtester.createOrder(
						product, lotsize, tl.LONG, account_id,
						order_type, entry_range, entry_price,
						sl_range, tp_range, sl_price, tp_price
					)

			else:
				raise tl.error.BrokerException('Unrecognisable order type specified.')

		else:
			endpoint = f'/v1/strategy/{self.strategyId}/brokers/{self.brokerId}/orders'
			payload = {
				'product': product,
				'lotsize': lotsize,
				'accounts': accounts,
				'direction': tl.LONG,
				'order_type': order_type,
				'entry_range': entry_range,
				'entry_price': entry_price,
				'sl_range': sl_range,
				'sl_price': sl_price,
				'tp_range': tp_range,
				'tp_price': tp_price
			}
			res = self._session.post(
				self._url + endpoint,
				data=json.dumps(payload)
			)

			res = res.json()
			for ref_id, item in res.items():
				if item.get('accepted'):
					func = self._get_trade_handler(item.get('type'))
					wait_result = self._wait(ref_id, func, (ref_id, item))
					
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

		return result

	def sell(self,
		product, lotsize, accounts,
		order_type=tl.MARKET_ORDER,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if self.findChartByProduct(product) is None:
			raise Exception('Instrument not loaded.')

		lotsize = self._convert_lotsize(lotsize)

		result = []
		if not self.isLive():
			if order_type == tl.MARKET_ORDER:
				for account_id in accounts:
					result = self.backtester.createPosition(
						product, lotsize, tl.SHORT,
						account_id, entry_range, entry_price,
						sl_range, tp_range, sl_price, tp_price
					)

			elif order_type == tl.LIMIT_ORDER or order_type == tl.STOP_ORDER:
				for account_id in accounts:
					result = self.backtester.createOrder(
						product, lotsize, tl.SHORT, account_id,
						order_type, entry_range, entry_price,
						sl_range, tp_range, sl_price, tp_price
					)

			else:
				raise tl.error.BrokerException('Unrecognisable order type specified.')

		else:
			endpoint = f'/v1/strategy/{self.strategyId}/brokers/{self.brokerId}/orders'
			payload = {
				'product': product,
				'lotsize': lotsize,
				'accounts': accounts,
				'direction': tl.SHORT,
				'order_type': order_type,
				'entry_range': entry_range,
				'entry_price': entry_price,
				'sl_range': sl_range,
				'sl_price': sl_price,
				'tp_range': tp_range,
				'tp_price': tp_price
			}
			res = self._session.post(
				self._url + endpoint,
				data=json.dumps(payload)
			)

			res = res.json()
			for ref_id, item in res.items():
				if item.get('accepted'):
					func = self._get_trade_handler(item.get('type'))
					wait_result = self._wait(ref_id, func, (ref_id, item))
					
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

		return result


	def stopAndReverse(self,
		product, lotsize, accounts,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		positions = self.positions
		if len(positions) > 0:
			direction = positions[-1].direction
			self.closeAllPositions()
		else:
			raise tl.error.OrderException('Must be in position to stop and reverse.')

		if direction == tl.LONG:
			res = self.sell(
				product, lotsize, accounts=accounts,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)
		else:
			res = self.buy(
				product, lotsize, accounts=accounts,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)

		return res

	def marketOrder(self,
		product, lotsize, direction, accounts,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if direction == tl.LONG:
			res = self.buy(
				product, lotsize, accounts=accounts,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)
		else:
			res = self.sell(
				product, lotsize, accounts=accounts,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)

		return res

	def stopOrder(self,
		product, lotsize, direction, accounts,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if direction == tl.LONG:
			res = self.buy(
				product, lotsize, accounts=accounts,
				order_type=tl.STOP_ORDER,
				entry_range=entry_range, entry_price=entry_price,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)
		else:
			res = self.sell(
				product, lotsize, accounts=accounts,
				order_type=tl.STOP_ORDER,
				entry_range=entry_range, entry_price=entry_price,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)

		return res

	def limitOrder(self,
		product, lotsize, direction, accounts,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if direction == tl.LONG:
			res = self.buy(
				product, lotsize, accounts=accounts,
				order_type=tl.LIMIT_ORDER,
				entry_range=entry_range, entry_price=entry_price,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)
		else:
			res = self.sell(
				product, lotsize, accounts=accounts,
				order_type=tl.LIMIT_ORDER,
				entry_range=entry_range, entry_price=entry_price,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)

		return res

	def closeAllPositions(self, positions=None):
		if positions is None:
			positions = self.getAllPositions()
		positions = copy(positions)

		result = []
		for pos in positions:
			result.append(pos.close())
		
		return result


	def cancelAllOrders(orders=None):
		if not orders:
			orders = self.getAllOrders()
		orders = copy(orders)

		result = []
		for order in orders:
			result.append(order.cancel())

		return result

	'''
	Data Utilities
	'''

	def _download_historical_prices(self, broker, product, period, start, end, count):
		print(f'LOADER DOWNLOAD HIST: {broker}, {product}, {period}, {start}, {end}, {count}', flush=True)

		now_time = tl.utils.setTimezone(datetime.datetime.utcnow(), 'UTC')
		if not end is None:
			if tl.utils.isOffsetAware(end):
				end = tl.utils.convertTimezone(end, 'UTC')
			else:
				end = tl.utils.setTimezone(end, 'UTC')
			last_date = end

		if not start is None:
			if tl.utils.isOffsetAware(start):
				start = tl.utils.convertTimezone(start, 'UTC')
			else:
				start = tl.utils.setTimezone(start, 'UTC')
			last_date = start
		
		if start is None and end is None:
			last_date = now_time

		data = pd.DataFrame(
			columns=[
				'timestamps',
				'ask_open', 'ask_high', 'ask_low', 'ask_close',
				'mid_open', 'mid_high', 'mid_low', 'mid_close',
				'bid_open', 'bid_high', 'bid_low', 'bid_close'
			]
		).set_index('timestamps')
		while True:
			endpoint = f'/v1/prices/{broker}/{product}/{period}'

			if not start is None and not end is None:
				res = self._session.get(
					self._url + endpoint,
					params = {
						'from': last_date.strftime('%Y-%m-%dT%H:%M:%SZ'), 
						'to': end.strftime('%Y-%m-%dT%H:%M:%SZ'), 
						'tz': 'UTC'
					}
				)
			else:
				if not start is None:
					print('start', flush=True)
					t_start = last_date
					t_end = tl.utils.getCountDate(period, count, start=last_date)
					last_date = t_end

				elif not end is None:
					print('end', flush=True)
					t_end = last_date
					t_start = tl.utils.getCountDate(period, count, end=last_date)
					last_date = t_start

				else:
					print('none', flush=True)
					t_end = last_date
					t_start = tl.utils.getCountDate(period, count, end=last_date)
					last_date = t_start

				params = {
					'from': t_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
					'to': t_end.strftime('%Y-%m-%dT%H:%M:%SZ')
				}

				res = self._session.get(
					self._url + endpoint,
					params=params
				)

			status_code = res.status_code
			if status_code == 200:
				# Convert result to dataframe
				result = res.json()

				if len(result['ohlc']['timestamps']) > 0:
					result = pd.DataFrame(
						index=result['ohlc']['timestamps'],
						columns=[
							'ask_open', 'ask_high', 'ask_low', 'ask_close',
							'mid_open', 'mid_high', 'mid_low', 'mid_close',
							'bid_open', 'bid_high', 'bid_low', 'bid_close'
						],
						data=np.concatenate((result['ohlc']['mids'], result['ohlc']['mids'], result['ohlc']['mids']), axis=1)
					)
					print(result.shape, flush=True)
				else:
					result = None

				data = pd.concat((data, result))

				if count is None:
					if result is None:
						break
					else:
						new_last_date = tl.utils.convertTimestampToTime(result.index.values[-1])
						if new_last_date == last_date or self._is_last_candle_found(period, new_last_date, end, 1):
							break

						last_date = new_last_date
						print(f'NEXT: {last_date}', flush=True)

				else:
					print(data.shape, flush=True)
					if not start is None:
						if data.shape[0] >= count or self._is_last_candle_found(period, last_date, now_time, 1):
							data.sort_index(inplace=True)
							data = data.iloc[:count]
							break

					elif data.shape[0] >= count:
						data.sort_index(inplace=True)
						data = data.iloc[-count:]
						break

			else:
				break

		data.sort_index(inplace=True)
		data = data[~data.index.duplicated(keep='first')]
		return data

	def _is_last_candle_found(self, period, start_dt, end_dt, count):
		utcnow = tl.utils.setTimezone(datetime.datetime.utcnow(), 'UTC')
		if tl.utils.isWeekend(utcnow):
			utcnow = tl.utils.getWeekendDate(utcnow)
		else:
			utcnow -= datetime.timedelta(seconds=tl.period.getPeriodOffsetSeconds(period))

		if tl.utils.isWeekend(end_dt):
			end_dt = tl.utils.getWeekendDate(end_dt)

		if period == tl.period.ONE_MINUTE:
			new_dt = start_dt + datetime.timedelta(minutes=count)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.TWO_MINUTES:
			new_dt = start_dt + datetime.timedelta(minutes=count*2)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.THREE_MINUTES:
			new_dt = start_dt + datetime.timedelta(minutes=count*3)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.FIVE_MINUTES:
			new_dt = start_dt + datetime.timedelta(minutes=count*5)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.TEN_MINUTES:
			new_dt = start_dt + datetime.timedelta(minutes=count*10)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.FIFTEEN_MINUTES:
			new_dt = start_dt + datetime.timedelta(minutes=count*15)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.THIRTY_MINUTES:
			new_dt = start_dt + datetime.timedelta(minutes=count*30)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.ONE_HOUR:
			new_dt = start_dt + datetime.timedelta(hours=count)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.FOUR_HOURS:
			new_dt = start_dt + datetime.timedelta(hours=count*4)
			return new_dt >= end_dt or new_dt >= utcnow
		elif period == tl.period.DAILY:
			new_dt = start_dt + datetime.timedelta(hours=count*24)
			return new_dt >= end_dt or new_dt >= utcnow
		else:
			raise Exception('Period not found.')

	'''
	Streaming Utilities
	'''

	def _stream_connect(self):
		print('Connected.', flush=True)
		# self._app.sio.emit(
		# 	'subscribe',
		# 	{
		# 		'broker_id': self.brokerId,
		# 		'field': 'ontrade'
		# 	},
		# 	namespace='/user'
		# )
		# if self.state == State.LIVE:
		# 	self._subscribe_charts(self.charts)

	def _stream_disconnect(self):
		print('Disconnected, retrying connection...', flush=True)


	def _stream_ontick(self, item):
		if not self.state == State.STOPPED:
			try:
				if "items" in item:
					chart = self.getChart(item['product'], broker=item['broker'])
					if chart is not None and chart._accept_ticks:
						chart._on_tick(item)
			except Exception as e:
				print(traceback.format_exc(), flush=True)
				self._app.sendScriptStopped()
				self.stop()


	'''
	On Trade Utilities
	'''

	def _wait(self, ref, func=None, res=None, polling=0.1, timeout=30):
		start = time.time()
		while not ref in self.handled:
			if time.time() - start >= timeout: 
				if func and res: return func(*res)
				else: return None
			time.sleep(polling)
		item = self.handled[ref]
		del self.handled[ref]
		return item



	def _stream_ontrade(self, items):
		if self.state == State.LIVE:
			try:
				print(items, flush=True)
				for ref_id, item in items.items():
					if item.get('accepted'):
						result = self.onTradeHandler(ref_id, item)

						print(f'ON TRADE RESULT: {result}', flush=True)
						# Handle result
						if result is not None and len(result) and result.get('account_id') == self.accountId:
							for func in self.ontrade_subs:
								func(
									EventItem({
										'reference_id': ref_id,
										'accepted': True,
										'type': item.get('type'),
										'item': result
									})
								)
						print(f'ON TRADE HANDLED', flush=True)
						print(f"[{self.brokerId}]\nPositions: {self.positions}\nOrders: {self.orders}")

					else:
						for func in self.onrejected_subs:
							func(
								EventItem({
									'reference_id': ref_id,
									'accepted': False,
									'type': item.get('type'),
									'message': item.get('message'),
									'item': item.get('item')
								})
							)

				self.strategy.onUpdateEnd()
				print(f'ON UPDATE END HANDLED', flush=True)


			except Exception as e:
				print(traceback.format_exc(), flush=True)
				self._app.sendScriptStopped()
				self.stop()

	def _stream_onsessionstatus(self, item):
		print(f'On Session Status: {item}', flush=True)
		try:
			for func in self.onsessionstatus_subs:
				func(EventItem(item))

			if item['type'] == 'disconnected':
				self.state = State.STOPPED
				print('DISCONNECTED + STOPPED', flush=True)
				# self.stop()

			elif item['type'] == 'connected':
				if self.state == State.STOPPED:
					return

		except Exception as e:
			print(traceback.format_exc(), flush=True)
			self._app.sendScriptStopped()
			self.stop()


	def _get_trade_handler(self, order_type):
		if order_type == tl.MARKET_ENTRY or order_type == tl.LIMIT_ENTRY or order_type == tl.STOP_ENTRY:
			return self.handlePositionEntry
		elif order_type == tl.LIMIT_ORDER or order_type == tl.STOP_ORDER:
			return self.handleOrderPlacement
		elif order_type == tl.MODIFY:
			self.handleModify
		elif (
			order_type == tl.POSITION_CLOSE
			or order_type == tl.STOP_LOSS
			or order_type == tl.TAKE_PROFIT
		):
			return self.handlePositionClose
		elif order_type == tl.ORDER_CANCEL:
			return self.handleOrderClose


	def subscribeOnTrade(self, func):
		self.ontrade_subs.append(func)

	
	def unsubscribeOnTrade(self, func):
		if func in self.ontrade_subs:
			del self.ontrade_subs[self.ontrade_subs.index(func)]


	def subscribeOnRejected(self, func):
		self.onrejected_subs.append(func)


	def unsubscribeOnRejected(self, func):
		if func in self.onrejected_subs:
			del self.onrejected_subs[self.onrejected_subs.index(func)]


	def subscribeOnSessionStatus(self, func):
		self.onsessionstatus_subs.append(func)


	def unsubscribeOnTrade(self, func):
		if func in self.onsessionstatus_subs:
			del self.onsessionstatus_subs[self.onsessionstatus_subs.index(func)]


	def handlePositionEntry(self, ref_id, item):
		# Handle
		pos = item.get('item')

		if pos.get("account_id") == self.accountId and self.chartProductExists(pos.get("product")):
			existing_pos = self.getPositionByID(pos["order_id"])
			if existing_pos is not None:
				new_pos = existing_pos
			else:
				new_pos = tl.position.Position.fromDict(self, pos)
				new_pos.lotsize = self._convert_incoming_lotsize(new_pos.lotsize)
			
				# Add position
				self.positions.append(new_pos)

			print(f'Position Entry: {ref_id}, {new_pos}')

			# Add to handled
			self.handled[ref_id] = new_pos

			return new_pos
		
		return None


	def handleOrderPlacement(self, ref_id, item):
		# Handle
		order = item.get('item')

		if order.get('account_id') == self.accountId and self.chartProductExists(order.get("product")):
			existing_order = self.getOrderByID(order["order_id"])
			if existing_order is not None:
				new_order = existing_order
			else:
				new_order = tl.order.Order.fromDict(self, order)
				new_order.lotsize = self._convert_incoming_lotsize(new_order.lotsize)

				self.orders.append(new_order)

			# Add to handled
			self.handled[ref_id] = new_order

			return new_order

		return None


	def handleModify(self, ref_id, item):
		# Handle
		order = item.get('item')
		result = None

		if order.get('account_id') == self.accountId:
			if order.get('order_type') == tl.STOP_ORDER or order.get('order_type') == tl.LIMIT_ORDER:
				for match_order in self.getAllOrders():
					if match_order.order_id == order['order_id']:
						match_order.entry_price = order['entry_price']
						match_order.sl = order['sl']
						match_order.tp = order['tp']
						match_order.lotsize = self._convert_incoming_lotsize(order['lotsize'])
						result = match_order
			else:
				for match_pos in self.getAllPositions():
					if match_pos.order_id == order['order_id']:
						match_pos.sl = order['sl']
						match_pos.tp = order['tp']
						result = match_pos

			# Add to handled
			self.handled[ref_id] = result

		return result

	def handlePositionClose(self, ref_id, item):
		# Handle
		positions = copy(self.getAllPositions())
		pos = item.get('item')
		result = None

		if pos.get('account_id') == self.accountId:
			print(f'CLOSING: {item}', flush=True)
			for j in range(len(positions)):
				match_pos = positions[j]
				if match_pos.order_id == pos['order_id']:
					print(f'FOUND', flush=True)
					pos['lotsize'] = self._convert_incoming_lotsize(pos['lotsize'])

					# Handle partial position close
					if pos['lotsize'] < match_pos.lotsize:
						print('PARTIAL CLOSE', flush=True)
						cpy = tl.position.Position.fromDict(self, pos)
						match_pos.lotsize = match_pos.lotsize - cpy.lotsize
						result = cpy
					# Handle full position close
					else:
						print('FULL CLOSE', flush=True)
						match_pos.close_price = pos['close_price']
						match_pos.close_time = pos['close_time']
						result = match_pos
						del self.positions[self.positions.index(match_pos)]

					print(self.positions, flush=True)
					break

			# Add to handled
			self.handled[ref_id] = result

		return result


	def handleOrderClose(self, ref_id, item):
		# Handle
		orders = copy(self.getAllOrders())
		order = item.get('item')
		result = None

		if order.get('account_id') == self.accountId:
			for j in range(len(orders)):
				match_order = orders[j]
				if match_order.order_id == order['order_id']:
					order['lotsize'] = self._convert_incoming_lotsize(order['lotsize'])

					# Handle partial position close
					if match_order.lotsize != order['lotsize']:
						cpy = tl.order.Order.fromDict(self, order)
						match_order.lotsize = match_order.lotsize - cpy.lotsize
						result = cpy
					# Handle full position close
					else:
						match_order.close_price = order['close_price']
						match_order.close_time = order['close_time']
						result = match_order
						del self.orders[self.orders.index(match_order)]

					break

			# Add to handled
			self.handled[ref_id] = result

		return result


	def handleUpdate(self, ref_id, item):
		if item.get('account_id') == self.accountId:
			print(f'HANDLING UPDATE: {item.get("account_id")}', flush=True)
			self.positions = [
				tl.position.Position.fromDict(self, pos) 
				for pos in item.get('positions') 
				if pos.get('account_id') == self.accountId and self.chartProductExists(pos.get("product"))
			]

			self.orders = [
				tl.order.Order.fromDict(self, order) 
				for order in item.get('orders') 
				if order.get('account_id') == self.accountId and self.chartProductExists(order.get("product"))
			]


	def onTradeHandler(self, ref_id, item):
		result = []
		order_type = item.get('type')

		# if self.state == State.LIVE:
		if (order_type == tl.LIMIT_ORDER or order_type == tl.STOP_ORDER
			or (order_type == tl.MODIFY and (item["item"]["order_type"] == tl.LIMIT_ORDER or item["item"]["order_type"] == tl.STOP_ORDER))
			or order_type == tl.ORDER_CANCEL):
			result = tl.order.Order.fromDict(self, item["item"])
		elif (order_type == tl.MARKET_ENTRY or order_type == tl.LIMIT_ENTRY or order_type == tl.STOP_ENTRY
			  or (order_type == tl.MODIFY and not (item["item"]["order_type"] == tl.LIMIT_ORDER or item["item"]["order_type"] == tl.STOP_ORDER))
			  or order_type == tl.POSITION_CLOSE or order_type == tl.STOP_LOSS or order_type == tl.TAKE_PROFIT):
			result = tl.position.Position.fromDict(self, item["item"])
		
		self.handled[ref_id] = result
			
		# else:
		# 	# Position Entry
		# 	if order_type == tl.MARKET_ENTRY or order_type == tl.LIMIT_ENTRY or order_type == tl.STOP_ENTRY:
		# 		result = self.handlePositionEntry(ref_id, item)

		# 	# Order Placement
		# 	elif order_type == tl.LIMIT_ORDER or order_type == tl.STOP_ORDER:
		# 		result = self.handleOrderPlacement(ref_id, item)

		# 	# Trade Modification
		# 	elif order_type == tl.MODIFY:
		# 		result = self.handleModify(ref_id, item)

		# 	# Position Close
		# 	elif (
		# 		order_type == tl.POSITION_CLOSE
		# 		or order_type == tl.STOP_LOSS
		# 		or order_type == tl.TAKE_PROFIT
		# 	):
		# 		print('HANDLE POSITION CLOSE', flush=True)
		# 		result = self.handlePositionClose(ref_id, item)

		# 	# Order Cancel
		# 	elif order_type == tl.ORDER_CANCEL:
		# 		result = self.handleOrderClose(ref_id, item)

		# 	# elif order_type == tl.UPDATE:
		# 	# 	self.handleUpdate(ref_id, item.get('item'))

		return result

