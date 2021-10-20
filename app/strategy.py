import json
import time
import math
import socketio
import requests
import traceback
import pandas as pd
import datetime
import dateutil.parser
from .. import app as tl
from .broker import Broker, BacktestMode, State
from threading import Thread

SAVE_INTERVAL = 60 # Seconds
MAX_GUI = 1000


class Strategy(object):

	def __init__(self, 
		app, module, strategy_id=None, broker_id=None, 
		account_id=None, user_variables={}, data_path='data/'
	):
		# Retrieve broker type
		self.app = app
		self.module = module
		self.strategyId = strategy_id
		self.brokerId = broker_id
		self.accountId = account_id
		self.broker = Broker(
			app, self, strategy_id=self.strategyId, broker_id=self.brokerId, 
			account_id=self.accountId, data_path=data_path
		)

		# GUI Queues
		self.message_queue = []

		self.drawing_queue = []
		self.log_queue = []
		self.info_queue = []
		self.transaction_queue = []
		self.reports = {}
		self.lastSave = time.time()

		self.input_variables = {}
		self.user_variables = user_variables

		self.tick_queue = []
		self.lastTick = None

		if self.app.run_state.value == 'run':
			Thread(target=self._handle_ticks).start()

	def run(self):
		self.broker.run()

	def stop(self):
		self.broker.stop()


	def getAccountCode(self):
		return self.brokerId + '.' + self.accountId


	def __getattribute__(self, key):
		if key == 'positions':
			return self.getAllPositions()
		elif key == 'orders':
			return self.getAllOrders()
		else:
			return super().__getattribute__(key)

	'''
	Broker functions
	'''

	def backtest(self, start, end, spread=None, mode=BacktestMode.RUN, process_mode=None):
		if self.getBroker().state != State.STOPPED:
			if isinstance(mode, str):
				mode = BacktestMode(mode)
			return self.getBroker().backtest(start, end, spread=spread, mode=mode, quick_download=True, process_mode=process_mode)

		else:
			raise tl.error.BrokerlibException('Strategy has been stopped.')


	def startFrom(self, dt):
		return self.getBroker().startFrom(dt)


	def setClearBacktestPositions(self, is_clear=True):
		self.getBroker().setClearBacktestPositions(is_clear)


	def setClearBacktestOrders(self, is_clear=True):
		self.getBroker().setClearBacktestOrders(is_clear)


	def setClearBacktestTrades(self, is_clear=True):
		self.getBroker().setClearBacktestTrades(is_clear)


	def clearBacktestPositions(self):
		self.getBroker()._clear_backtest_positions()


	def clearBacktestOrders(self):
		self.getBroker()._clear_backtest_orders()


	def clearBacktestTrades(self):
		self.getBroker()._clear_backtest_trades()


	def getBrokerName(self):
		return self.getBroker().name


	# Chart functions
	def getChart(self, broker, product, *periods):
		if self.getBroker().state != State.STOPPED:
			return self.getBroker().getChart(product, *periods, broker=broker)

		else:
			raise tl.error.BrokerlibException('Strategy has been stopped.')

	# Account functions
	def getCurrency(self):
		return self.getBroker().getAccountInfo(self.accountId)[self.accountId]['currency']

	def getBalance(self):
		balance = self.getBroker().getAccountInfo(self.accountId)[self.accountId]['balance']
		return balance

	def getProfitLoss(self):
		return self.getBroker().getAccountInfo(self.accountId)[self.accountId]['pl']

	def getEquity(self):
		info = self.getBroker().getAccountInfo(self.accountId)[self.accountId]
		return info['balance'] + info['pl']

	def getMargin(self):
		return self.getBroker().getAccountInfo(self.accountId)[self.accountId]['margin']


	# Order functions
	def getAllPositions(self):
		result = self.getBroker().getAllPositions(account_id=self.accountId)
		return result


	def getAllOrders(self):
		result = self.getBroker().getAllOrders(account_id=self.accountId)
		return result


	def buy(self,
		product, lotsize,
		order_type=tl.MARKET_ORDER,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if self.getBroker().state != State.STOPPED:
			return self.getBroker().buy(
				product, lotsize, [self.accountId],
				order_type=order_type,
				entry_range=entry_range, entry_price=entry_price,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)

		else:
			raise tl.error.BrokerlibException('Strategy has been stopped.')


	def sell(self,
		product, lotsize,
		order_type=tl.MARKET_ORDER,
		entry_range=None, entry_price=None,
		sl_range=None, tp_range=None,
		sl_price=None, tp_price=None
	):
		if self.getBroker().state != State.STOPPED:
			return self.getBroker().sell(
				product, lotsize, [self.accountId],
				order_type=order_type,
				entry_range=entry_range, entry_price=entry_price,
				sl_range=sl_range, tp_range=tp_range,
				sl_price=sl_price, tp_price=tp_price
			)

		else:
			raise tl.error.BrokerlibException('Strategy has been stopped.')


	def closeAllPositions(self, positions=None):
		return self.getBroker().closeAllPositions(positions)


	'''
	GUI Functions
	'''

	def draw(self, draw_type, layer, product, value, timestamp, 
				section_name='instrument', section_props={},
				color='#000000', scale=1.0, rotation=0, props={}):
		if self.getBroker().state == State.LIVE:
			real_timestamp = self.lastTick.timestamp + tl.period.getPeriodOffsetSeconds(self.lastTick.period)
		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			real_timestamp = self.lastTick.chart._end_timestamp

		drawing = {
			'id': self.broker.generateReference(),
			'product': product,
			'layer': layer,
			'type': draw_type,
			'timestamps': [int(timestamp)],
			'prices': [value],
			'section_name': section_name,
			'section_props': section_props,
			'properties': {
				**{
					'colors': [color],
					'scale': scale,
					'rotation': rotation
				},
				**props
			}
		}

		self._create_drawing(real_timestamp, layer, drawing)


	def text(self, text, layer, product, value, timestamp, 
				section_name='instrument', section_props={},
				color='#000000', font_size=10, rotation=0):
		if self.getBroker().state == State.LIVE:
			real_timestamp = self.lastTick.timestamp + tl.period.getPeriodOffsetSeconds(self.lastTick.period)
		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			real_timestamp = self.lastTick.chart._end_timestamp

		drawing = {
			'id': self.broker.generateReference(),
			'product': product,
			'layer': layer,
			'type': 'text',
			'timestamps': [int(timestamp)],
			'prices': [value],
			'section_name': section_name,
			'section_props': section_props,
			'properties': {
				'text': text,
				'colors': [color],
				'text': text,
				'font_size': font_size,
				'rotation': rotation
			}
		}

		self._create_drawing(real_timestamp, layer, drawing)


	def _create_drawing(self, timestamp, layer, drawing):
		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			item = {
				'timestamp': timestamp,
				'type': tl.CREATE_DRAWING,
				'account_id': self.getAccountCode(),
				'item': drawing
			}

			# Append to message queue
			self.message_queue.append(item)
			# Save to drawing queue
			if self.strategyId == "Q529J5" or self.strategyId == "34MULF":
				self.transaction_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			# Handle drawings through backtester
			self.getBroker().backtester.createDrawing(timestamp, layer, drawing)


	def clearDrawingLayer(self, layer):

		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			timestamp = self.lastTick.timestamp + tl.period.getPeriodOffsetSeconds(self.lastTick.period)

			item = {
				'id': self.broker.generateReference(),
				'timestamp': timestamp,
				'type': tl.CLEAR_DRAWING_LAYER,
				'account_id': self.getAccountCode(),
				'item': layer
			}

			# Append to message queue
			self.message_queue.append(item)
			# Handle to drawing queue
			if self.strategyId == "Q529J5" or self.strategyId == "34MULF":
				self.transaction_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			timestamp = self.lastTick.chart._end_timestamp

			# Handle drawings through backtester
			self.getBroker().backtester.clearDrawingLayer(timestamp, layer)


	def clearAllDrawings(self):

		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			timestamp = self.lastTick.timestamp + tl.period.getPeriodOffsetSeconds(self.lastTick.period)

			item = {
				'id': self.broker.generateReference(),
				'timestamp': timestamp,
				'type': tl.CLEAR_ALL_DRAWINGS,
				'account_id': self.getAccountCode(),
				'item': None
			}

			# Append to message queue
			self.message_queue.append(item)
			# Handle to drawing queue
			if self.strategyId == "Q529J5" or self.strategyId == "34MULF":
				self.transaction_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			timestamp = self.lastTick.chart._end_timestamp

			# Handle drawings through backtester
			self.getBroker().backtester.deleteAllDrawings(timestamp)


	def log(self, *objects, sep=' ', end='\n', file=None, flush=None):
		# print(*objects, sep=sep, end=end, file=file, flush=True)
		msg = sep.join(map(str, objects)) + end
		timestamp = math.floor(time.time())

		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			if self.lastTick is not None:
				timestamp = self.lastTick.timestamp + tl.period.getPeriodOffsetSeconds(self.lastTick.period)

			item = {
				'timestamp': timestamp,
				'type': tl.CREATE_LOG,
				'account_id': self.getAccountCode(),
				'item': msg
			}

			# Append to message queue
			self.message_queue.append(item)
			# Save to log queue
			if self.strategyId == "Q529J5" or self.strategyId == "34MULF":
				self.transaction_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			if self.lastTick is not None:
				timestamp = self.lastTick.chart._end_timestamp

			# Handle logs through backtester
			self.getBroker().backtester.createLogItem(timestamp, msg)


	def clearLogs(self):
		return


	def info(self, product, period, name, value='', type='text', color=None):
		timestamp = self.lastTick.chart.timestamps[period][0]

		# Check if value is json serializable
		json.dumps(value)

		item = {
			'name': str(name),
			'type': type,
			'value': value,
			'color': color
		}

		if self.getBroker().state == State.LIVE:
			item = {
				'product': product,
				'period': period,
				'timestamp': timestamp,
				'type': tl.CREATE_INFO,
				'account_id': self.getAccountCode(),
				'item': item
			}

			# Append to message queue
			self.message_queue.append(item)
			# Handle to info queue
			if self.strategyId == "Q529J5" or self.strategyId == "34MULF":
				self.info_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			item = {
				'product': product,
				'period': period,
				'timestamp': timestamp,
				'type': tl.CREATE_INFO,
				'account_id': self.getAccountCode(),
				'item': item
			}

			# Handle info through backtester
			self.getBroker().backtester.createInfoItem(item)


	def banner(self, html):
		item = {
			'type': tl.BANNER,
			'item': html
		}

		if self.getBroker().state == State.LIVE:
			# Append to message queue
			self.message_queue.append(item)


	def createReport(self, name, columns):
		self.reports[name] = pd.DataFrame(columns=columns)


	def resetReports(self):
		for name in self.reports:
			self.createReport(name, self.reports[name].columns)


	def report(self, name, *data):
		if name in self.reports:
			self.reports[name].loc[self.reports[name].shape[0]] = list(map(str, data))


	'''
	Setters
	'''

	def resetGuiQueues(self):
		self.transaction_queue = []
		self.info_queue = []
		# self.drawing_queue = []
		# self.log_queue = []
		self.lastSave = time.time()

		# Reset report data
		for name in self.reports:
			report = self.reports[name]
			self.createReport(name, report.columns)


	def handleMessageQueue(self):
		if len(self.message_queue) > 0:
			try:
				self.app.sio.emit(
					'ongui', 
					{
						'strategy_id': self.strategyId, 
						'item': {
							'type': 'message_queue', 
							'item': self.message_queue
						}
					}, 
					namespace='/admin'
				)
			except Exception:
				print(traceback.format_exc(), flush=True)

			self.message_queue = []


	def handleDrawingsSave(self, gui):
		if len(gui) == 0:
			gui = self.getGui()

		if 'drawings' not in gui or not isinstance(gui['drawings'], dict):
			gui['drawings'] = {}

		for i in self.drawing_queue:
			if i['type'] == tl.CREATE_DRAWING:
				if i['item']['layer'] not in gui['drawings']:
					gui['drawings'][i['item']['layer']] = []
				gui['drawings'][i['item']['layer']].append(i['item'])

			elif i['type'] == tl.CLEAR_DRAWING_LAYER:
				if i['item'] in gui['drawings']:
					gui['drawings'][i['item']] = []

			elif i['type'] == tl.CLEAR_ALL_DRAWINGS:
				for layer in gui['drawings']:
					gui['drawings'][layer] = []

		for layer in gui['drawings']:
			gui['drawings'][layer] = gui['drawings'][layer][-MAX_GUI:]

		return gui

	def handleLogsSave(self, gui):
		if len(gui) == 0:
			gui = self.getGui()

		if 'logs' not in gui or not isinstance(gui['logs'], list):
			gui['logs'] = []


		gui['logs'] += self.log_queue
		gui['logs'] = gui['logs'][-MAX_GUI:]

		return gui

	def handleInfoSave(self, gui):
		if len(gui) == 0:
			gui = self.getGui()

		if 'info' not in gui or not isinstance(gui['info'], dict):
			gui['info'] = {}

		for i in self.info_queue:
			if i['product'] not in gui['info']:
				gui['info'][i['product']] = {}
			if i['period'] not in gui['info'][i['product']]:
				gui['info'][i['product']][i['period']] = {}
			if str(int(i['timestamp'])) not in gui['info'][i['product']][i['period']]:
				gui['info'][i['product']][i['period']][str(int(i['timestamp']))] = []

			gui['info'][i['product']][i['period']][str(int(i['timestamp']))].append(i['item'])
		
		gui['info'][i['product']][i['period']] = dict(sorted(
			gui['info'][i['product']][i['period']].items(), key=lambda x: float(x[0])
		)[-MAX_GUI:])
		
		return gui


	def handleReportsSave(self):
		result = {}
		for name in self.reports:
			report = self.reports[name]
			# Check if report has new data
			if report.shape[0] > 0:
				# Retrieve saved report and concatenate
				old_report = self.getReport(name)
				if old_report is not None:
					report = pd.concat((old_report, report))
				result[name] = report.to_dict()

		return result
		

	def getGui(self):
		endpoint = f'/v1/strategy/{self.strategyId}/gui/{self.brokerId}/{self.accountId}'
		res = self.getBroker()._session.get(
			self.getBroker()._url + endpoint
		)

		if res.status_code == 200:
			return res.json()
		else:
			return {}


	def getReport(self, name):
		endpoint = f'/v1/strategy/{self.strategyId}/gui/{self.brokerId}/{self.accountId}/reports/{name}'
		res = self.getBroker()._session.get(
			self.getBroker()._url + endpoint
		)

		if res.status_code == 200:
			return pd.DataFrame(data=res.json())
		else:
			return None


	def saveGui(self):
		print('SAVE GUI', flush=True)
		update = {}

		# if len(self.drawing_queue) > 0:
		# 	update['drawings'] = self.drawing_queue

		# if len(self.log_queue) > 0:
		# 	update['logs'] = self.log_queue

		if len(self.info_queue) > 0:
			update['info'] = self.info_queue

		if len(self.transaction_queue) > 0:
			update['transactions'] = self.transaction_queue

		reports = {}
		for name in self.reports:
			report = self.reports[name]
			if report.size > 0:
				reports[name] = report.to_dict()
		print(f'SAVE GUI REPORTS: {reports}', flush=True)
		if len(reports) > 0:
			update['reports'] = reports

		if len(update) > 0:
			endpoint = f'/v1/strategy/{self.strategyId}/gui/{self.brokerId}/{self.accountId}'
			payload = json.dumps(update).encode()
			res = self.getBroker().upload(endpoint, payload)

			if res.status_code == 200:
				self.resetGuiQueues()


	def on_user_input(self, item):
		for name in item:
			if name in self.user_variables:
				self.user_variables[name]['value'] = item[name]['value']

		# Initialize strategy
		if 'onUserInput' in dir(self.module.main) and callable(self.module.main.onUserInput):
			self.module.main.onUserInput(item)


	def setInputVariable(self, name, input_type, scope=tl.GLOBAL, default=None, properties={}):
		if input_type == int:
			assert default is None or isinstance(default, int)
			input_type = tl.INTEGER

		elif input_type == float:
			assert default is None or isinstance(default, float)
			input_type = tl.DECIMAL

		elif input_type == str:
			assert default is None or isinstance(default, str)
			input_type = tl.TEXT

		elif input_type == tl.HEADER:
			assert default is None or isinstance(default, str)

		elif input_type == tl.PERCENTAGE:
			assert default is None or isinstance(default, float) or isinstance(default, int)

		elif input_type == tl.TIME:
			assert default is None or isinstance(default, datetime.time) or isinstance(default, str)
			if isinstance(default, datetime.time):
				default = default.isoformat()
			elif isinstance(default, str):
				dateutil.parser.parse(default)

		elif input_type == tl.DATE:
			assert default is None or isinstance(default, datetime.datetime) or isinstance(default, str)
			if isinstance(default, datetime.datetime):
				default = default.isoformat()
			elif isinstance(default, str):
				dateutil.parser.parse(default)

		elif input_type == tl.CASH:
			assert default is None or isinstance(default, float) or isinstance(default, int)

		elif not input_type in (tl.BALANCE, tl.CUSTOM):
			raise Exception(f'Unknown input variable type: {input_type}')

		self.input_variables[name] = {
			'default': default,
			'type': input_type,
			'scope': scope,
			'value': default,
			'index': len(self.input_variables),
			'properties': properties
		}

		return self.getInputVariable(name)


	def getInputVariable(self, name):
		input_type = None
		value = None
		if name in self.input_variables:
			if name in self.user_variables and self.user_variables[name]['type'] == self.input_variables[name]['type']:
				value = self.user_variables[name]['value']
				input_type = self.user_variables[name]['type']

				if 'properties' in self.user_variables[name]:
					if self.user_variables[name]['properties'].get('enabled') == False:
						value = None

			else:
				value = self.input_variables[name]['value']
				input_type = self.input_variables[name]['type']

				if 'properties' in self.input_variables[name]:
					if self.input_variables[name]['properties'].get('enabled') == False:
						value = None

			if value is not None:
				if input_type == tl.INTEGER:
					value = int(value)

				elif input_type == tl.DECIMAL:
					value = float(value)

				elif input_type == tl.TEXT:
					value = str(value)

				elif input_type == tl.PERCENTAGE:
					value = float(value)

				elif input_type == tl.TIME:
					value = dateutil.parser.parse(value)
					value = datetime.time(value.hour, value.minute, value.second, value.microsecond)

				elif input_type == tl.DATE:
					value = dateutil.parser.parse(value)

				elif input_type == tl.CASH:
					value = float(value)

			return value


	def _handle_ticks(self):
		try:
			while self.getBroker().state != State.STOPPED:
				if (
					self.getBroker().state == State.LIVE and
					len(self.tick_queue)
				):
					handled = False
					for chart in self.app.charts:
						if (
							self.tick_queue[0]['broker'] == chart.broker and
							self.tick_queue[0]['product'] == chart.product
						):
							handled = True
							item = self.tick_queue[0]
							chart.handleTick(item)
							del self.tick_queue[0]
							break

					if not handled:
						del self.tick_queue[0]

				time.sleep(0.001)

		except Exception:
			print(traceback.format_exc(), flush=True)
			print('---STOPPED---', flush=True)
			self.strategy.getBroker().stop()


	def setTick(self, tick):
		self.lastTick = tick


	def onTickEnd(self):
		self.handleMessageQueue()
		# Save GUI
		if self.getBroker().state == State.LIVE:
			if time.time() - self.lastSave > SAVE_INTERVAL:
				self.lastSave = time.time()
				Thread(target=self.saveGui).start()


	def onUpdateEnd(self):
		self.handleMessageQueue()
		
	'''
	Getters
	'''

	def getBroker(self):
		return self.broker

