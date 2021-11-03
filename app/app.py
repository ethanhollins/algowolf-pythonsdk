import importlib
import sys
import os
import signal
import requests
import json
import time
import traceback
import os
import socketio
import zmq
from copy import copy
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
		self.send_queue = []
		self._msg_queue = {}
		
		self._setup_zmq_connections()

		self._script_path = os.path.join(config.get('SCRIPTS_PATH'), self.scriptId)
		sys.path.append(self._script_path)


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

	
	def _setup_zmq_connections(self):
		print("[_setup_zmq_connections] START", flush=True)
		self.zmq_context = zmq.Context()

		# self.zmq_sub_socket = self.zmq_context.socket(zmq.SUB)
		# self.zmq_sub_socket.connect("tcp://zmq_broker:5556")
		# self.zmq_sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

		# self.zmq_dealer_socket = self.zmq_context.socket(zmq.DEALER)
		# self.zmq_dealer_socket.connect("tcp://zmq_broker:5557")

		# self.zmq_poller = zmq.Poller()
		# self.zmq_poller.register(self.zmq_sub_socket, zmq.POLLIN)
		# self.zmq_poller.register(self.zmq_dealer_socket, zmq.POLLIN)

		Thread(target=self.zmq_send_loop).start()
		Thread(target=self.zmq_message_loop).start()


	def onCommand(self, message):
		if self.strategy is not None:
			if message.get("type") == "ontick":
				self.strategy.getBroker()._stream_ontick(message["message"])

			elif message.get("type") == "ontrade":
				if message.get("broker_id") == self.strategy.brokerId:
					self.strategy.getBroker()._stream_ontrade(message["message"])
			
			elif message.get("type") == "response":
				self._msg_queue[message["message"]["msg_id"]] = message["message"]


	def waitResponse(self, msg_id, timeout=60):
		start = time.time()

		while time.time() - start < timeout:
			if msg_id in copy(list(self._msg_queue.keys())):
				res = self._msg_queue[msg_id]
				del self._msg_queue[msg_id]
				print('WAIT RECV', flush=True)
				return res
			time.sleep(0.01)

		return {
			'error': 'No response.'
		}

	
	def sendRequest(self, message):
		self.send_queue.append(message)


	def zmq_send_loop(self):
		self.zmq_dealer_socket = self.zmq_context.socket(zmq.DEALER)
		self.zmq_dealer_socket.connect("tcp://zmq_broker:5557")

		while True:
			try:
				if len(self.send_queue):
					item = self.send_queue[0]
					del self.send_queue[0]
					print(f"[Send] Send {item}, {len(self.send_queue)}", flush=True)

					self.zmq_dealer_socket.send_json(item, zmq.NOBLOCK)

			except Exception:
				print(traceback.format_exc())

			time.sleep(0.001)


	def zmq_message_loop(self):
		self.zmq_sub_socket = self.zmq_context.socket(zmq.SUB)
		self.zmq_sub_socket.connect("tcp://zmq_broker:5556")
		self.zmq_sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

		self.zmq_poller = zmq.Poller()
		self.zmq_poller.register(self.zmq_sub_socket, zmq.POLLIN)

		while True:
			try:
				socks = dict(self.zmq_poller.poll())

				if self.zmq_sub_socket in socks:
					message = self.zmq_sub_socket.recv_json()
					self.onCommand(message)

			except Exception:
				print(traceback.format_exc(), flush=True)


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
			if 'onStart' in dir(self.module.main) and callable(self.module.main.onStart):
				print(f"[onStart]: {self.strategy.getBroker().state}")
				self.module.main.onStart()

		except Exception as e:
			print(traceback.format_exc(), flush=True)
			self.stop()
			raise e


	def stop(self):
		
		if self.strategy is not None:
			self.strategy.stop()
			# self.strategy = None
			# self.module = None
		else:
			self.sendScriptStopped()
			time.sleep(5)

			try:
				self.sio.disconnect()
			except:
				pass
			finally:
				PID = os.getpid()
				# os.kill(PID, signal.SIGTERM)
				os.kill(PID, signal.SIGKILL)


	def backtest(self, auth_key, broker, _from, to, mode, input_variables, spread, process_mode):
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

			backtest_id = self.strategy.backtest(_from, to, spread=spread, mode=mode, process_mode=process_mode)
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


	def recursivelyImportScript(self, name, path):
		print(f'recursivelyImportScript: {os.listdir(path)}', flush=True)
		# for i in os.listdir(path):
		# 	c_path = os.path.join(path, i)
		# 	print(f'{c_path}: {c_path.endswith(".py")}', flush=True)
		# 	if c_path.endswith('.py'):
		# 		spec = importlib.util.find_spec(name + '.' + i.split('.')[0])
		# 		print(spec.name, flush=True)
		# 		module = importlib.util.module_from_spec(spec)
		# 		print(module, flush=True)		

				# sys.modules[name] = module
				# spec.loader.exec_module(module)

			# elif os.path.isdir(c_path):

		for i in os.listdir(path):
			c_path = os.path.join(path, i)
			if os.path.isdir(c_path) and i != 'venv':
				print(f'APPEND: {c_path}', flush=True)
				sys.path.append(c_path)
				self.recursivelyImportScript(name, c_path)

			elif c_path.endswith('.py') or c_path.endswith('.pyx'):
				module_name = name + '.' + i.split('.')[0]
				print(module_name, flush=True)
				spec = importlib.util.spec_from_file_location(module_name, c_path)
				print(c_path, flush=True)
				print(spec, flush=True)




	def getPackageModule(self, package):

		print(package, flush=True)
		spec = importlib.util.find_spec(package)
		module = importlib.util.module_from_spec(spec)
		print(spec.name, flush=True)
		print(dir(module), flush=True)

		print(os.listdir(self._script_path), flush=True)
		self.recursivelyImportScript(spec.name, self._script_path)

		# sys.modules[spec.name] = module
		spec.loader.exec_module(module)

		# if '__version__' in dir(module):
		# 	return self.getPackageModule(package + '.' + module.__version__)

		return module

	def startStrategy(self, auth_key, input_variables):
		if self.strategy is None:
			self.module = self.getPackageModule(f'{self.package}')

			if auth_key is not None:
				# self._retrieve_session_token(auth_key)
				self.key = auth_key
			else:
				self.key = ''

			self.strategy = Strategy(self, self.module, strategy_id=self.strategyId, broker_id=self.brokerId, account_id=self.accountId, user_variables=input_variables)

			# Set global variables
			self.module.main.print = self.strategy.log

			self.module.main.strategy = self.strategy
			self.module.main.utils = tl.utils
			self.module.main.product = tl.product
			self.module.main.period = tl.period
			self.module.main.indicator = tl.indicator
			for i in dir(tl.constants):
				vars(self.module.main)[i] = vars(tl.constants)[i]

			# Initialize strategy
			if 'init' in dir(self.module.main) and callable(self.module.main.init):
				self.module.main.init()

			# Search for convertional function names
			if 'onTrade' in dir(self.module.main) and callable(self.module.main.onTrade):
				self.strategy.getBroker().subscribeOnTrade(self.module.main.onTrade)

			if 'onRejected' in dir(self.module.main) and callable(self.module.main.onRejected):
				self.strategy.getBroker().subscribeOnRejected(self.module.main.onRejected)	

			if 'onTick' in dir(self.module.main) and callable(self.module.main.onTick):
				for chart in self.strategy.getBroker().getAllCharts():
					for period in chart.periods:
						chart.subscribe(period, self.module.main.onTick)

			if 'onSessionStatus' in dir(self.module.main) and callable(self.module.main.onSessionStatus):
				self.strategy.getBroker().subscribeOnSessionStatus(self.module.main.onSessionStatus)


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
