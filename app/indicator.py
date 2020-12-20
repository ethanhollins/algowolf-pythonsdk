import numpy as np
import functools


class Indicator(object):

	def __init__(self, name, properties, storage, period=None):
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

	def _preprocessing(self, data):
		timestamps = data.index.values
		asks = data.values[:,:4]
		mids = data.values[:,4:8]
		bids = data.values[:,8:]
		
		return timestamps, asks, mids, bids

	def _perform_calculation(self, price_type, ohlc, idx):
		return

	def _set_idx(self, idx):
		self.idx = idx
		self.asks = np.around(self._asks[:self.idx+1][::-1], decimals=5)
		self.mids = np.around(self._mids[:self.idx+1][::-1], decimals=5)
		self.bids = np.around(self._bids[:self.idx+1][::-1], decimals=5)


	def limit(self):
		self._asks = self._asks[-1000:]
		self._mids = self._mids[-1000:]
		self._bids = self._bids[-1000:]
		self.idx = self._asks.shape[0]-1

		self.asks = np.around(self._asks[:self.idx+1][::-1], decimals=5)
		self.mids = np.around(self._mids[:self.idx+1][::-1], decimals=5)
		self.bids = np.around(self._bids[:self.idx+1][::-1], decimals=5)


	def isIndicator(self, name, props):
		return (
			name.lower() == self.name and
			props == self.properties
		)

	def calculate(self, data, idx):		
		timestamps, asks, mids, bids = self._preprocessing(data)

		# Calculate ask prices
		for i in range(idx, timestamps.shape[0]):
			new_ask = [self._perform_calculation('ask', asks, i)]
			if isinstance(self._asks, type(None)):
				self._asks = np.array(new_ask, dtype=float)
			elif i < self._asks.shape[0]:
				self._asks[i] = new_ask[0]
			else:
				self._asks = np.concatenate((self._asks, new_ask))

		# Calculate mid prices
		for i in range(idx, timestamps.shape[0]):
			new_mid = [self._perform_calculation('mid', mids, i)]

			if isinstance(self._mids, type(None)):
				self._mids = np.array(new_mid, dtype=float)
			elif i < self._mids.shape[0]:
				self._mids[i] = new_mid[0]
			else:
				self._mids = np.concatenate((self._mids, new_mid))

		# Calculate bid prices
		for i in range(idx, timestamps.shape[0]):
			new_bid = [self._perform_calculation('bid', bids, i)]

			if isinstance(self._bids, type(None)):
				self._bids = np.array(new_bid, dtype=float)
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

	def __init__(self, period, std_dev):
		super().__init__('boll', [period, std_dev], None)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]
		std_dev = self.properties[1]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]

		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]*2

		# print(ohlc)

		# Perform calculation
		mean = np.sum(ohlc[:,3]) / ohlc.shape[0]
		d_sum = np.sum((ohlc[:,3] - mean) ** 2)
		sd = np.sqrt(d_sum/period)

		return [
			mean + sd * std_dev,
			mean - sd * std_dev
		]


# Donchian Channel
class DONCH(Indicator):

	def __init__(self, period):
		super().__init__('donch', [period], None)

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

	def __init__(self, period):
		super().__init__('ema', [period], [0, 0])

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-period, 0):idx+1]
		# Check min period met
		if ohlc.shape[0] < period:
			return [np.nan]

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

		return [ema]


# Moving Average Envelope
class MAE(Indicator):

	def __init__(self, period, percent, type='ema'):
		super().__init__('mae', [period, percent], None)

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

	def __init__(self, period):
		super().__init__('sma', [period], None)

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

	def __init__(self, period):
		super().__init__('atr', [period], None)

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

	def __init__(self, period):
		super().__init__('tr', [period], None)

	def _perform_calculation(self, price_type, ohlc, idx):
		# Properties:
		period = self.properties[0]

		# Get relevant OHLC
		ohlc = ohlc[max((idx+1)-(period+1), 0):idx+1]

		# Check min period met
		if ohlc.shape[0] < period+1:
			return [np.nan]

		# Perform calculation
		if idx > period+1:
			if price_type == 'ask':
				prev = self._asks[idx-1, 0]
			elif price_type == 'mid':
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

	def __init__(self, period):
		super().__init__('cci', [period], None)

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

	def __init__(self, period):
		super().__init__('rsi', [period], [0, 0, 0, 0])

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

		if price_type == 'ask':
			if idx > len(self.asks)-1:
				self.storage[0] = gain_avg
				self.storage[1] = loss_avg
		elif price_type == 'mid':
			if idx > len(self.mids)-1:
				self.storage[2] = gain_avg
				self.storage[3] = loss_avg
		else:
			if idx > len(self.bids)-1:
				self.storage[4] = gain_avg
				self.storage[5] = loss_avg

		if loss_avg == 0.0:
			return [100.0]
		else:
			return [100.0 - (100.0 / (1.0 + gain_avg/loss_avg))]


'''
Imports
'''

from .. import app as tl


