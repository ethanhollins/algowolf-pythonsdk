import json
import time
import math
import socketio
import requests
from .. import app as tl
from .broker import Broker, BacktestMode, State
from threading import Thread

SAVE_INTERVAL = 60 # Seconds
MAX_GUI = 200


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
		self.drawing_queue = []
		self.log_queue = []
		self.info_queue = []
		self.lastSave = time.time()

		self.input_variables = {}
		self.user_variables = user_variables

		self.tick_queue = []
		self.lastTick = None

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

	def backtest(self, start, end, mode=BacktestMode.RUN):
		if self.getBroker().state != State.STOPPED:
			if isinstance(mode, str):
				mode = BacktestMode(mode)
			return self.getBroker().backtest(start, end, mode=mode, quick_download=True)

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
	def getChart(self, product, *periods):
		if self.getBroker().state != State.STOPPED:
			return self.getBroker().getChart(product, *periods)

		else:
			raise tl.error.BrokerlibException('Strategy has been stopped.')

	# Account functions
	def getCurrency(self):
		return self.getBroker().getAccountInfo([self.accountId])[self.accountId]['currency']

	def getBalance(self):
		return self.getBroker().getAccountInfo([self.accountId])[self.accountId]['balance']

	def getProfitLoss(self):
		return self.getBroker().getAccountInfo([self.accountId])[self.accountId]['pl']

	def getEquity(self):
		info = self.getBroker().getAccountInfo([self.accountId])[self.accountId]
		return info['balance'] + info['pl']

	def getMargin(self):
		return self.getBroker().getAccountInfo([self.accountId])[self.accountId]['margin']


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

	def draw(self, draw_type, layer, product, price, timestamp, 
				color='#000000', scale=1.0, rotation=0):
		timestamp = self.lastTick.timestamp
		drawing = {
			'id': self.broker.generateReference(),
			'product': product,
			'layer': layer,
			'type': draw_type,
			'timestamps': [int(timestamp)],
			'prices': [price],
			'properties': {
				'colors': [color],
				'scale': scale,
				'rotation': rotation
			}
		}
		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			item = {
				'timestamp': timestamp,
				'type': tl.CREATE_DRAWING,
				'account_id': self.getAccountCode(),
				'item': drawing
			}

			# Send Gui Socket Message
			self.app.sio.emit(
				'ongui', 
				{'strategy_id': self.strategyId, 'item': item}, 
				namespace='/admin'
			)

			# Save to drawing queue
			self.drawing_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			# Handle drawings through backtester
			self.getBroker().backtester.createDrawing(timestamp, layer, drawing)


	def clearDrawingLayer(self, layer):
		timestamp = self.lastTick.timestamp

		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			item = {
				'id': self.broker.generateReference(),
				'timestamp': timestamp,
				'type': tl.CLEAR_DRAWING_LAYER,
				'account_id': self.getAccountCode(),
				'item': layer
			}

			# Send Gui Socket Message
			self.app.sio.emit(
				'ongui', 
				{'strategy_id': self.strategyId, 'item': item}, 
				namespace='/admin'
			)

			# Handle to drawing queue
			self.drawing_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			# Handle drawings through backtester
			self.getBroker().backtester.clearDrawingLayer(timestamp, layer)


	def clearAllDrawings(self):
		timestamp = self.lastTick.timestamp

		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			item = {
				'id': self.broker.generateReference(),
				'timestamp': timestamp,
				'type': tl.CLEAR_ALL_DRAWINGS,
				'account_id': self.getAccountCode(),
				'item': None
			}

			# Send Gui Socket Message
			self.app.sio.emit(
				'ongui', 
				{'strategy_id': self.strategyId, 'item': item}, 
				namespace='/admin'
			)

			# Handle to drawing queue
			self.drawing_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			# Handle drawings through backtester
			self.getBroker().backtester.deleteAllDrawings(timestamp)


	def log(self, *objects, sep=' ', end='\n', file=None, flush=None):
		# print(*objects, sep=sep, end=end, file=file, flush=flush)
		msg = sep.join(map(str, objects)) + end
		if self.lastTick is not None:
			timestamp = self.lastTick.timestamp
		else:
			timestamp = math.floor(time.time())

		if self.getBroker().state == State.LIVE or self.getBroker().state == State.IDLE:
			item = {
				'timestamp': timestamp,
				'type': tl.CREATE_LOG,
				'account_id': self.getAccountCode(),
				'item': msg
			}

			# Send Gui Socket Message
			self.app.sio.emit(
				'ongui', 
				{'strategy_id': self.strategyId, 'item': item}, 
				namespace='/admin'
			)

			# Save to log queue
			self.log_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			# Handle logs through backtester
			self.getBroker().backtester.createLogItem(timestamp, msg)


	def clearLogs(self):
		return


	def info(self, name, value):
		timestamp = self.lastTick.timestamp

		# Check if value is json serializable
		json.dumps(value)

		item = {
			'name': str(name),
			'value': value
		}

		if self.getBroker().state == State.LIVE:
			item = {
				'timestamp': timestamp,
				'type': tl.CREATE_INFO,
				'account_id': self.getAccountCode(),
				'item': item
			}

			# Send Gui Socket Message
			self.app.sio.emit(
				'ongui', 
				{'strategy_id': self.strategyId, 'item': item}, 
				namespace='/admin'
			)

			# Handle to info queue
			self.info_queue.append(item)

		elif self.getBroker().state.value <= State.BACKTEST_AND_RUN.value:
			# Handle info through backtester
			self.getBroker().backtester.createInfoItem(timestamp, item)

	'''
	Setters
	'''

	def resetGuiQueues(self):
		self.drawing_queue = []
		self.log_queue = []
		self.info_queue = []
		self.lastSave = time.time()


	def handleDrawingsSave(self, gui):
		if gui is None:
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
		if gui is None:
			gui = self.getGui()

		if 'logs' not in gui or not isinstance(gui['logs'], list):
			gui['logs'] = []


		gui['logs'] += self.log_queue
		gui['logs'] = gui['logs'][-MAX_GUI:]

		return gui

	def handleInfoSave(self, gui):
		if gui is None:
			gui = self.getGui()

		if 'info' not in gui or not isinstance(gui['info'], dict):
			gui['info'] = {}

		for i in self.info_queue:
			if i['timestamp'] not in gui['info']:
				gui['info'][i['timestamp']] = []

			gui['info'][i['timestamp']].append(i['item'])
		
		gui['info'] = dict(sorted(
			gui['logs'].items(), key=lambda x: x[0]
		)[-MAX_GUI:])
		
		return gui


	def getGui(self):
		endpoint = f'/v1/strategy/{self.strategyId}/gui/{self.brokerId}/{self.accountId}'
		res = self.getBroker()._session.get(
			self.getBroker()._url + endpoint
		)

		if res.status_code == 200:
			return res.json()
		else:
			return {}


	def saveGui(self):
		if time.time() - self.lastSave > SAVE_INTERVAL:
			gui = None

			if len(self.drawing_queue) > 0:
				gui = self.handleDrawingsSave(gui)

			if len(self.log_queue) > 0:
				gui = self.handleLogsSave(gui)

			if len(self.info_queue) > 0:
				gui = self.handleInfoSave(gui)

			if gui is not None:
				endpoint = f'/v1/strategy/{self.strategyId}/gui/{self.brokerId}/{self.accountId}'
				res = self.getBroker()._session.put(
					self.getBroker()._url + endpoint,
					data=json.dumps(gui)
				)

				if res.status_code == 200:
					self.resetGuiQueues()


	def on_user_input(self, item):
		for name in item:
			if name in self.user_variables:
				self.user_variables[name]['value'] = item[name]['value']

		# Initialize strategy
		if 'onUserInput' in dir(module) and callable(module.onUserInput):
			module.onUserInput(item)


	def setInputVariable(self, name, input_type, scope=tl.GLOBAL, default=None, properties={}):
		if input_type == int:
			input_type = tl.INTEGER
		elif input_type == float:
			input_type = tl.DECIMAL
		elif input_type == str:
			input_type = tl.TEXT
		elif input_type == None:
			raise Exception('')

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
		if name in self.input_variables:
			if name in self.user_variables:
				if self.user_variables[name]['type'] == self.input_variables[name]['type']:
					return self.user_variables[name]['value']

			return self.input_variables[name]['value']


	def setTick(self, tick):
		self.lastTick = tick

		# # Save GUI
		if self.getBroker().state == State.LIVE:
			self.saveGui()

	'''
	Getters
	'''

	def getBroker(self):
		return self.broker

