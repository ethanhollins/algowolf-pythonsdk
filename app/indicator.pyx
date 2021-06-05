import sys
import numpy as np
import functools


class RoundedArray(np.ndarray):

	def __new__(cls, input_array, *args, **kwargs):
		return np.asarray(input_array).view(cls)

	def __init__(self, input_array, precision):
		self._precision = precision

	def __getitem__(self, idx):
		item = super(RoundedArray, self).__getitem__(idx)
		try:
			return np.around(item, decimals=self._precision)
		except:
			return item


class Indicator(object):

	def __init__(self, name, properties, storage, period=None, precision=5):
		self.name = name
		self.properties = properties
		self.storage = storage
		self.period = period

		self.idx = 0
		self.asks = None
		self.mids = None
		self.bids = None
		self._asks = None
		self._mids = None
		self._bids = None

		self.precision = precision


	def _preprocessing(self, data):
		timestamps = data.index.values
		asks = data.values[:,:4]
		mids = data.values[:,4:8]
		bids = data.values[:,8:]
		
		return timestamps, asks, mids, bids

	def _perform_calculation(self, int price_type, double[:,:] ohlc, int idx):
		return


	def _set_idx(self, idx):
		self.idx = idx
		# self.asks = np.around(self._asks[:self.idx+1][::-1], decimals=5)
		# self.mids = np.around(self._mids[:self.idx+1][::-1], decimals=5)
		# self.bids = np.around(self._bids[:self.idx+1][::-1], decimals=5)
		self.asks = RoundedArray(self._asks[:self.idx+1][::-1], self.precision)
		self.mids = RoundedArray(self._mids[:self.idx+1][::-1], self.precision)
		self.bids = RoundedArray(self._bids[:self.idx+1][::-1], self.precision)


	def limit(self):
		self._asks = self._asks[-constants.BUFFER_COUNT:]
		self._mids = self._mids[-constants.BUFFER_COUNT:]
		self._bids = self._bids[-constants.BUFFER_COUNT:]
		self.idx = self._asks.shape[0]-1

		self.asks = RoundedArray(self._asks[:self.idx+1][::-1], self.precision)
		self.mids = RoundedArray(self._mids[:self.idx+1][::-1], self.precision)
		self.bids = RoundedArray(self._bids[:self.idx+1][::-1], self.precision)
		# print(self.asks[:5], flush=True)

		# self.asks = np.around(self._asks[:self.idx+1][::-1], decimals=5)
		# self.mids = np.around(self._mids[:self.idx+1][::-1], decimals=5)
		# self.bids = np.around(self._bids[:self.idx+1][::-1], decimals=5)


	def isIndicator(self, name, props):
		return (
			name.lower() == self.name and
			props == self.properties
		)

	def calculate(self, data, idx):		
		timestamps, asks, mids, bids = self._preprocessing(data)

		# Calculate ask prices
		for i in range(idx, timestamps.shape[0]):
			new_ask = [self._perform_calculation(0, asks, i)]
			if isinstance(self._asks, type(None)):
				self._asks = np.array(new_ask, dtype=np.float64)
			elif i < self._asks.shape[0]:
				self._asks[i] = new_ask[0]
			else:
				self._asks = np.concatenate((self._asks, new_ask))

		# Calculate mid prices
		for i in range(idx, timestamps.shape[0]):
			new_mid = [self._perform_calculation(1, mids, i)]

			if isinstance(self._mids, type(None)):
				self._mids = np.array(new_mid, dtype=np.float64)
			elif i < self._mids.shape[0]:
				self._mids[i] = new_mid[0]
			else:
				self._mids = np.concatenate((self._mids, new_mid))

		# Calculate bid prices
		for i in range(idx, timestamps.shape[0]):
			new_bid = [self._perform_calculation(2, bids, i)]

			if isinstance(self._bids, type(None)):
				self._bids = np.array(new_bid, dtype=np.float64)
			elif i < self._bids.shape[0]:
				self._bids[i] = new_bid[0]
			else:
				self._bids = np.concatenate((self._bids, new_bid))


	def getCurrentAsk(self):
		return self._asks[self.idx]

	def getCurrentBid(self):
		return self._bids[self.idx]

	def getAsk(self, off, amount):
		return self._asks[max(self.idx+1-off-amount,0):self.idx-off]

	def getBid(self, off, amount):
		return self._bids[max(self.idx+1-off-amount,0):self.idx+1-off]

	def setPeriod(self, period):
		self.period = period

'''
Overlays
'''

# Bollinger Bands
class BOLL(Indicator):

	def __init__(self, period, std_dev, precision=5):
		super().__init__('boll', [period, std_dev], None, precision=precision)

	def _perform_calculation(self, int price_type, double[:,:] ohlc, int idx):
		# Properties:
		cdef int period = self.properties[0]
		cdef double std_dev = self.properties[1]
		cdef double mean
		cdef double d_sum
		cdef double sd

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]

		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]*2

		# Perform calculation
		mean = np.sum(ohlc[:,3]) / ohlc.shape[0]
		d_sum = np.sum(np.subtract(ohlc[:,3], mean) ** 2)
		sd = np.sqrt(d_sum/period)

		return [
			mean + sd * std_dev,
			mean - sd * std_dev
		]


# Donchian Channel
class DONCH(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('donch', [period], None, precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]

		# Get relevant OHLC
		ohlc = ohlc[max((idx)-period, 0):idx]
		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]*2

		high_low = [0,0]
		for i in range(ohlc.shape[0]):
			if high_low[0] == 0 or ohlc[i,1] > high_low[0]:
				high_low[0] = ohlc[i][1]
			if high_low[1] == 0 or ohlc[i,2] < high_low[1]:
				high_low[1] = ohlc[i,2]
		return high_low


# Exponential Moving Average
class EMA(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('ema', [period], [0, 0], precision=precision)

	def _perform_calculation(self, int price_type, double[:,:] ohlc, int idx):
		# Properties:
		cdef double period = self.properties[0]
		cdef double prev_ema
		cdef double multi
		cdef double ma
		cdef double ema

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-<int> period, 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]

		# Perform calculation
		if idx > period:
			if price_type == 0:
				prev_ema = self._asks[idx-1, 0]
			elif price_type == 1:
				prev_ema = self._mids[idx-1, 0]
			else:
				prev_ema = self._bids[idx-1, 0]

			multi = 2.0 / (period + 1.0)
			ema = (ohlc[-1, 3] - prev_ema) * multi + prev_ema

		else:
			ma = 0
			for i in range(ohlc.shape[0]):
				ma += ohlc[i,3]

			ema = ma / period

		return [ema]


# Moving Average Envelope
class MAE(Indicator):

	def __init__(self, period, percent, type='ema', precision=5):
		super().__init__('mae', [period, percent], None, precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]
		percent = self.properties[1]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]*3

		# Perform calculation

		if idx > period:
			if price_type == 'ask':
				prev_ema = self._asks[idx-1, 0]
			elif price_type == 'mid':
				prev_ema = self._mids[idx-1, 0]
			else:
				prev_ema = self._bids[idx-1, 0]

			multi = 2 / (period + 1)
			ema = (ohlc[-1, 3] - prev_ema) * multi + prev_ema

		else:
			ma = 0
			for i in range(ohlc.shape[0]):
				ma += ohlc[i,3]

			ema = ma / period

		off = ema * (percent/100)
		return [ema, ema + off, ema - off]


# Simple Moving Average
class SMA(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('sma', [period], None, precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]

		# Perform calculation
		ma = 0
		for i in range(ohlc.shape[0]):
			ma += ohlc[i,3]
		return [np.around(ma / period, decimals=5)]


'''
Studies
'''

# Average True Range
class ATR(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('atr', [period], None, precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]

		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]

		# Perform calculation
		prev_close = ohlc[-2, 3]
		high = ohlc[-1, 1]
		low = ohlc[-1, 2]

		if idx > period:
			if price_type == 'ask':
				prev_atr = self._asks[idx-1, 0]
			elif price_type == 'mid':
				prev_atr = self._mids[idx-1, 0]
			else:
				prev_atr = self._bids[idx-1, 0]

			if prev_close > high:
				tr = prev_close - low
			elif prev_close < low:
				tr = high - prev_close
			else:
				tr = high - low

			atr = (prev_atr * (period-1) + tr) / period

		else:
			tr_sum = 0
			for i in range(ohlc.shape[0]):
				if i == 0:
					tr_sum += (ohlc[i,1] - ohlc[i,2])
				else:
					if prev_close > high:
						tr_sum += prev_close - low
					elif prev_close < low:
						tr_sum += high - prev_close
					else:
						tr_sum += high - low

			atr = tr_sum / period


		return [atr]

# Modified Average True Range
class TR(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('tr', [period], None, precision=precision)

	def _perform_calculation(self, int price_type, double[:,:] ohlc, int idx):
		# Properties:
		cdef double period = self.properties[0]
		cdef double prev
		cdef double prev_close
		cdef double high
		cdef double low
		cdef double tr
		cdef double alpha
		cdef double atr
		cdef double tr_sum

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-(<int> period+1), 0):idx+1]

		# Check min period met
		if ohlc.shape[0] < period+1:
			return [np.nan]

		# Perform calculation
		if idx > period+1:
			if price_type == 0:
				prev = self._asks[idx-1, 0]
			elif price_type == 1:
				prev = self._mids[idx-1, 0]
			else:
				prev = self._bids[idx-1, 0]

			prev_close = ohlc[-2, 3]
			high = ohlc[-1, 1]
			low = ohlc[-1, 2]

			if prev_close > high:
				tr = prev_close - low
			elif prev_close < low:
				tr = high - prev_close
			else:
				tr = high - low

			alpha = 1 / period
			atr = alpha * tr + (1 - alpha) * prev

		else:
			tr_sum = 0
			for i in range(1, ohlc.shape[0]):
				prev_close = ohlc[i-1, 3]
				high = ohlc[i, 1]
				low = ohlc[i, 2]

				if prev_close > high:
					tr_sum += prev_close - low
				elif prev_close < low:
					tr_sum += high - prev_close
				else:
					tr_sum += high - low

			atr = tr_sum / period

		return [atr]


# Commodity Channel Index
class CCI(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('cci', [period], None, precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]

		# Perform calculation

		# Calculate Typical price SMA
		c_typ = (ohlc[-1,1] + ohlc[-1,2] + ohlc[-1,3])/3.0
		typ_sma = 0.0
		for i in range(ohlc.shape[0]):
			typ_sma += (ohlc[i,1] + ohlc[i,2] + ohlc[i,3])/3.0

		typ_sma /= period
		
		# Calculate Mean Deviation
		mean_dev = 0.0
		for i in range(ohlc.shape[0]):
			mean_dev += np.absolute(
				((ohlc[i,1] + ohlc[i,2] + ohlc[i,3])/3.0) - typ_sma
			)

		mean_dev /= period
		const = .015

		if mean_dev == 0:
			return 0

		return [np.around((c_typ - typ_sma) / (const * mean_dev), decimals=5)]


# Relative Strength Index
class RSI(Indicator):

	def __init__(self, period, precision=5):
		super().__init__('rsi', [period], [0, 0, 0, 0, 0, 0], precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]
		if price_type == 'ask':
			prev_gain = self.storage[0]
			prev_loss = self.storage[1]
		elif price_type == 'mid':
			prev_gain = self.storage[2]
			prev_loss = self.storage[3]
		else:
			prev_gain = self.storage[4]
			prev_loss = self.storage[5]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-(period+1), 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < period+1:
			return [np.nan]

		# Perform calculation
		gain_sum = 0.0
		loss_sum = 0.0
			
		if prev_gain and prev_loss:
			chng = ohlc[-1,3] - ohlc[-2,3]
			if chng >= 0:
				gain_sum += chng
			else:
				loss_sum += np.absolute(chng)

			gain_avg = (prev_gain * (period-1) + gain_sum)/period
			loss_avg = (prev_loss * (period-1) + loss_sum)/period

		else:
			for i in range(1, ohlc.shape[0]):
				chng = ohlc[i,3] - ohlc[i-1,3]

				if chng >= 0:
					gain_sum += chng
				else:
					loss_sum += np.absolute(chng)

			gain_avg = gain_sum / period
			loss_avg = loss_sum / period

		if price_type == 0:
			if not isinstance(self._asks, type(None)) and idx > self._asks.shape[0]-1:
				self.storage[0] = gain_avg
				self.storage[1] = loss_avg
		elif price_type == 1:
			if not isinstance(self._mids, type(None)) and idx > self._mids.shape[0]-1:
				self.storage[2] = gain_avg
				self.storage[3] = loss_avg
		else:
			if not isinstance(self._bids, type(None)) and idx > self._bids.shape[0]-1:
				self.storage[4] = gain_avg
				self.storage[5] = loss_avg

		if loss_avg == 0.0:
			return [100.0]
		else:
			return [100.0 - (100.0 / (1.0 + gain_avg/loss_avg))]


# Relative Strength Index
class MACD(Indicator):

	def __init__(self, fast, slow, signal, precision=5):
		super().__init__('macd', [fast, slow, signal], [0, 0, 0, 0, 0, 0, 0, 0, 0], precision=precision)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		fast_period = self.properties[0]
		slow_period = self.properties[1]
		signal_period = self.properties[2]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-(slow_period+1), 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < slow_period+1:
			return [np.nan, np.nan, np.nan]

		# Perform calculation
		# Fast EMA
		if idx > fast_period:
			if price_type == 0:
				prev_fast_ema = self.storage[0]
			elif price_type == 1:
				prev_fast_ema = self.storage[1]
			else:
				prev_fast_ema = self.storage[2]

			multi = 2.0 / (fast_period + 1.0)
			fast_ema = (ohlc[-1, 3] - prev_fast_ema) * multi + prev_fast_ema

		else:
			fast_ma = 0
			for i in range(fast_period):
				fast_ma += ohlc[i,3]

			fast_ema = fast_ma / fast_period

		if price_type == 0:
			self.storage[0] = fast_ema
		elif price_type == 1:
			self.storage[1] = fast_ema
		else:
			self.storage[2] = fast_ema

		# Slow Ema
		if idx > slow_period:
			if price_type == 0:
				prev_slow_ema = self.storage[3]
			elif price_type == 1:
				prev_slow_ema = self.storage[4]
			else:
				prev_slow_ema = self.storage[5]

			multi = 2.0 / (slow_period + 1.0)
			slow_ema = (ohlc[-1, 3] - prev_slow_ema) * multi + prev_slow_ema

		else:
			# Slow EMA
			slow_ma = 0
			for i in range(slow_period):
				slow_ma += ohlc[i,3]

			slow_ema = slow_ma / slow_period

		if price_type == 0:
			self.storage[3] = slow_ema
		elif price_type == 1:
			self.storage[4] = slow_ema
		else:
			self.storage[5] = slow_ema

		# MACD
		macd = fast_ema - slow_ema

		if idx > signal_period:
			# Signal Ema
			if price_type == 0:
				prev_signal_ema = self.storage[6]
			elif price_type == 1:
				prev_signal_ema = self.storage[7]
			else:
				prev_signal_ema = self.storage[8]

			multi = 2.0 / (signal_period + 1.0)
			signal_ema = (macd - prev_signal_ema) * multi + prev_signal_ema

		else:
			# Slow EMA
			signal_ema = macd / signal_period

		if price_type == 0:
			self.storage[6] = signal_ema
		elif price_type == 1:
			self.storage[7] = signal_ema
		else:
			self.storage[8] = signal_ema

		hist = macd - signal_ema

		return [macd, signal_ema, hist]

'''
Imports
'''

from . import constants
