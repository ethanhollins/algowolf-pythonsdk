import importlib
import sys
import requests
import json
import time
import traceback
import os
import socketio
from enum import Enum
from datetime import datetime, timedelta
from threading import Thread


class RunState(Enum):
	RUN = 'run'
	BACKTEST = 'backtest'
	COMPILE = 'compile'


class App(object):

	def __init__(self, config, package, strategy_id, account_code):
		if package.endswith('.py'):
			package = package.replace('.py', '')
			
		self.config = config
		self.scriptId, self.package = package.split('.')
		self.strategyId = strategy_id
		self.run_state = None

		if account_code is not None:
			self.brokerId, self.accountId = self._convert_account_code(account_code)
		else:
			self.brokerId = 'BACKTESTER'
			self.accountId = 'ACCOUNT_1'

		self.sio = socketio.Client()

		# Containers
		self.strategy = None
		self.module = None
		self.indicators = []
		self.charts = []

		sys.path.append(os.path.join(config.get('SCRIPTS_PATH'), self.scriptId))


	def _convert_account_code(self, account_code):
		return account_code.split('.')


	def _retrieve_session_token(self, key):
		headers = {
			'Authorization': 'Bearer ' + key
		}

		endpoint = f'/v1/session'
		res = requests.get(
			self.config.get('API_URL') + endpoint,
			headers=headers
		)

		if res.status_code == 200:
			self.key = res.json().get('token')
		else:
			raise Exception('Failed to get session token.')


	def connectSio(self):
		headers = {
			'Authorization': 'Bearer '+self.key
		}

		self.sio.connect(
			self.config.get('STREAM_URL'), 
			namespaces=['/admin', '/user'],
			headers=headers
		)


	# TODO: Add accounts parameter
	def run(self, auth_key, input_variables):
		self.run_state = RunState.RUN

		try:
			# Start strategy for each account
			self.startStrategy(auth_key, input_variables)
			if self.strategy.getBroker().state.value <= 2:
				# Run strategy
				self.strategy.run()

			# Call strategy onStart
			if 'onStart' in dir(self.module) and callable(self.module.onStart):
				self.module.onStart()

			self.sendScriptRunning()

		except Exception as e:
			print(traceback.format_exc(), flush=True)
			self.stop()
			raise e


	def stop(self):
		if self.strategy is not None:
			self.sendScriptStopped()
			self.strategy.get('strategy').stop()
			# self.strategy = None
			# self.module = None


	def backtest(self, auth_key, broker, _from, to, mode, input_variables, spread):
		self.run_state = RunState.BACKTEST

		e = None
		try:
			_from = tl.utils.convertTimestampToTime(float(_from))
			to = tl.utils.convertTimestampToTime(float(to))
			# if isinstance(_from, str):
			# 	_from = datetime.strptime(_from, '%Y-%m-%dT%H:%M:%SZ')
			# if isinstance(to, str):
			# 	to = datetime.strptime(to, '%Y-%m-%dT%H:%M:%SZ')
			self.startStrategy(auth_key, input_variables)
			self.strategy.getBroker().setName(broker)

			if spread is not None:
				spread = float(spread)

			backtest_id = self.strategy.backtest(_from, to, spread=spread, mode=mode)
		except Exception as err:
			print(traceback.format_exc(), flush=True)
			e = err
		finally:
			self.sio.disconnect()
			if e is not None:
				raise e

		return backtest_id


	def compile(self, auth_key):
		self.run_state = RunState.COMPILE

		account_id = 'ACCOUNT_1'
		properties = {}
		e = None
		try:
			self.startStrategy(None, {})

		except Exception as err:
			print(traceback.format_exc(), flush=True)
			e = err
		finally:
			self.sio.disconnect()

			if e is not None:
				raise e

			properties['input_variables'] = self.strategy.input_variables
			endpoint = f'/v1/scripts/{self.scriptId}'
			payload = {
				'properties': properties
			}
			self.strategy.getBroker()._session.post(
				self.strategy.getBroker()._url + endpoint,
				data=json.dumps(payload)
			)

		return properties


	def getPackageModule(self, package):
		spec = importlib.util.find_spec(package)
		module = importlib.util.module_from_spec(spec)
		# sys.modules[spec.name] = module
		spec.loader.exec_module(module)

		# if '__version__' in dir(module):
		# 	return self.getPackageModule(package + '.' + module.__version__)

		return module

	def startStrategy(self, auth_key, input_variables):
		if self.strategy is None:
			self.module = self.getPackageModule(f'{self.package}')

			if auth_key is not None:
				self._retrieve_session_token(auth_key)
			else:
				self.key = ''

			self.strategy = Strategy(self, self.module, strategy_id=self.strategyId, broker_id=self.brokerId, account_id=self.accountId, user_variables=input_variables)

			# Set global variables
			self.module.print = self.strategy.log

			self.module.strategy = self.strategy
			self.module.utils = tl.utils
			self.module.product = tl.product
			self.module.period = tl.period
			self.module.indicator = tl.indicator
			for i in dir(tl.constants):
				vars(self.module)[i] = vars(tl.constants)[i]

			# Initialize strategy
			if 'init' in dir(self.module) and callable(self.module.init):
				self.module.init()

			# Search for convertional function names
			if 'onTrade' in dir(self.module) and callable(self.module.onTrade):
				self.strategy.getBroker().subscribeOnTrade(self.module.onTrade)

			if 'onRejected' in dir(self.module) and callable(self.module.onRejected):
				self.strategy.getBroker().subscribeOnRejected(self.module.onRejected)	

			if 'onTick' in dir(self.module) and callable(self.module.onTick):
				for chart in self.strategy.getBroker().getAllCharts():
					for period in chart.periods:
						chart.subscribe(period, self.module.onTick)

			if 'onSessionStatus' in dir(self.module) and callable(self.module.onSessionStatus):
				self.strategy.getBroker().subscribeOnSessionStatus(self.module.onSessionStatus)


	def sendScriptRunning(self):
		try:
			self.sio.emit(
				'ongui', 
				{
					'strategy_id': self.strategyId, 
					'item': {
						'account_code': '.'.join((self.brokerId, self.accountId)),
						'type': 'script_running',
					}
				}, 
				namespace='/admin'
			)
		except Exception:
			print(traceback.format_exc(), flush=True)


	def sendScriptStopped(self):
		try:
			self.sio.emit(
				'ongui', 
				{
					'strategy_id': self.strategyId, 
					'item': {
						'account_code': '.'.join((self.brokerId, self.accountId)),
						'type': 'script_stopped',
					}
				}, 
				namespace='/admin'
			)
		except Exception:
			print(traceback.format_exc(), flush=True)


	def sendLiveBacktestUploaded(self):
		try:
			self.sio.emit(
				'ongui', 
				{
					'strategy_id': self.strategyId, 
					'item': {
						'account_code': '.'.join((self.brokerId, self.accountId)),
						'type': 'live_backtest_uploaded',
					}
				}, 
				namespace='/admin'
			)
		except Exception:
			print(traceback.format_exc(), flush=True)


	def getStrategy(self):
		return self.strategy


	def getModule(self):
		return self.module


	def addChart(self, chart):
		self.charts.append(chart)


	def getChart(self, broker, product):
		for i in self.charts:
			if i.isChart(broker, product):
				return i
		return None


'''
Imports
'''


from .strategy import Strategy
from .error import BrokerlibException
from .. import app as tl
