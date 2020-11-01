import importlib
import sys
import requests
import json
import time
import traceback
import os
import socketio
from datetime import datetime, timedelta
from threading import Thread

class App(object):

	def __init__(self, config, package, strategy_id, broker_id, account_id):
		if package.endswith('.py'):
			package = package.replace('.py', '')
			
		self.config = config
		self.package = package
		self.strategyId = strategy_id
		self.brokerId = broker_id
		self.account_id = account_id

		self.sio = self.setupSio()

		# Containers
		self.strategy = None
		self.module = None
		self.indicators = []
		self.charts = []


	def setupSio(self):
		self.sio.connect(self.config.get('STREAM_URL'), namespaces=['/admin'])


	# TODO: Add accounts parameter
	def run(self, input_variables):
		# Start strategy for each account
		self.startStrategy(input_variables)
		if self.strategy.getBroker().state.value <= 2:
			# Run strategy
			self.strategy.run()

		# Call strategy onStart
		if 'onStart' in dir(self.module) and callable(self.module.onStart):
			self.module.onStart()


	def stop(self):
		if self.strategy is not None:
			self.strategy.get('strategy').stop()
			self.strategy = None
			self.module = None


	def backtest(self, _from, to, mode, input_variables):
		account_id = 'ACCOUNT_1'

		e = None
		try:
			if isinstance(_from, str):
				_from = datetime.strptime(_from, '%Y-%m-%dT%H:%M:%SZ')
			if isinstance(to, str):
				to = datetime.strptime(to, '%Y-%m-%dT%H:%M:%SZ')
			self.startStrategy(account_id, input_variables)
			self.strategies[account_id].get('strategy').getBroker().setName(self.api.name)

			backtest_id = self.strategies[account_id].get('strategy').backtest(_from, to, mode)
		except Exception as err:
			print(traceback.format_exc())
			e = err
		finally:
			if account_id in self.strategies:
				del self.strategies[account_id]

			if e is not None:
				raise TradelibException(str(e))

		return backtest_id


	def compile(self):
		account_id = 'ACCOUNT_1'
		properties = {}
		e = None
		try:
			self.startStrategy(account_id, {})

		except Exception as err:
			print(traceback.format_exc())
			e = err
		finally:
			if account_id in self.strategies:
				strategy = self.strategies[account_id].get('strategy')
				properties['input_variables'] = strategy.input_variables
				del self.strategies[account_id]

			if e is not None:
				raise TradelibException(str(e))

			return properties


	def getPackageModule(self, package):
		spec = importlib.util.find_spec(package)
		module = importlib.util.module_from_spec(spec)
		# sys.modules[spec.name] = module
		spec.loader.exec_module(module)

		if '__version__' in dir(module):
			return self.getPackageModule(package + '.' + module.__version__)

		return module

	def startStrategy(self, account_id, input_variables):
		if self.strategy is None:
			self.module = self.getPackageModule(f'{self.package}')

			self.strategy = Strategy(module, strategy_id=self.strategyId, broker_id=self.brokerId, account_id=account_id, user_variables=input_variables)
			self.strategy.setApp(self)

			# Set global variables
			self.module.print = strategy.log

			self.module.strategy = strategy
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

			if 'onTick' in dir(self.module) and callable(self.module.onTick):
				for chart in self.strategy.getBroker().getAllCharts():
					for period in chart.periods:
						chart.subscribe(period, self.module.onTick)


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
from .error import BrokerlibException, TradelibException
import app as tl
