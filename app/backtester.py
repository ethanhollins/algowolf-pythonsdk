import time
import math
import numpy as np
import pandas as pd
from copy import copy
from datetime import datetime


class Backtester(object):

	def __init__(self, broker):
		self.broker = broker
		self.result = []
		self.info = {}

		self._idx = 0


	def _create_empty_transaction_df(self):
		df = pd.DataFrame(columns=[
			'reference_id', 'timestamp', 'type', 'accepted',
			'order_id', 'account_id', 'product', 'order_type',
			'direction', 'lotsize', 'entry_price', 'close_price', 'sl', 'tp',
			'open_time', 'close_time'
		])
		return df.set_index('reference_id')


	def _order_validation(self, order, min_dist=0):

		if order.direction == tl.LONG:
			price = self.broker.getAsk(order.product)
		else:
			price = self.broker.getBid(order.product)

		# Entry validation
		if order.get('order_type') == tl.STOP_ORDER or order.get('order_type') == tl.LIMIT_ORDER:
			if order.entry_price == None:
				raise tl.error.OrderException('Order must contain entry price.')
			elif order.order_type == tl.LIMIT_ORDER:
				if order.direction == tl.LONG:
					if order.entry_price > price - tl.utils.convertToPrice(min_dist):
						raise tl.error.OrderException('Long limit order entry must be lesser than current price.')
				else:
					if order.entry_price < price + tl.utils.convertToPrice(min_dist):
						raise tl.error.OrderException('Short limit order entry must be greater than current price.')
			elif order.order_type == tl.STOP_ORDER:
				if order.direction == tl.LONG:
					if order.entry_price < price + tl.utils.convertToPrice(min_dist):
						raise tl.error.OrderException('Long stop order entry must be greater than current price.')
				else:
					if order.entry_price > price - tl.utils.convertToPrice(min_dist):
						raise tl.error.OrderException('Short stop order entry must be lesser than current price.')

		# SL/TP validation
		if order.direction == tl.LONG:
			if order.sl and order.sl > order.entry_price - tl.utils.convertToPrice(min_dist):
				raise tl.error.OrderException('Stop loss price must be lesser than entry price.')
			if order.tp and order.tp < order.entry_price + tl.utils.convertToPrice(min_dist):
				raise tl.error.OrderException('Take profit price must be greater than entry price.')
		else:
			if order.sl and order.sl < order.entry_price + tl.utils.convertToPrice(min_dist):
				raise tl.error.OrderException('Stop loss price must be greater than entry price.')
			if order.tp and order.tp > order.entry_price - tl.utils.convertToPrice(min_dist):
				raise tl.error.OrderException('Take profit price must be lesser than entry price.')


	def createPosition(self,
		product, lotsize, direction,
		account_id, entry_range, entry_price,
		sl_range, tp_range, sl_price, tp_price
	):
		if direction == tl.LONG:
			price = self.broker.getAsk(product)
		else:
			price = self.broker.getBid(product)

		# Set Stoploss
		sl = None
		if sl_price:
			sl = np.around(sl_price, 5)
		elif sl_range:
			if direction == tl.LONG:
				sl = np.around(price - tl.utils.convertToPrice(sl_range), 5)
			else:
				sl = np.around(price + tl.utils.convertToPrice(sl_range), 5)

		# Set Takeprofit
		tp = None
		if tp_price:
			tp = np.around(tp_price, 5)
		elif tp_range:
			if direction == tl.LONG:
				tp = np.around(price + tl.utils.convertToPrice(tp_range), 5)
			else:
				tp = np.around(price - tl.utils.convertToPrice(tp_range), 5)

		# Set Entry Price
		entry =None
		if entry_price:
			entry = np.around(entry_price, 5)
		else:
			entry = np.around(price, 5)

		open_time = int(self.broker.getTimestamp(product))

		order_id = self.broker.generateReference()
		order = tl.BacktestPosition(self.broker,
			order_id, account_id, product, tl.MARKET_ENTRY, 
			direction, lotsize, entry_price=entry, sl=sl, tp=tp,
			open_time=open_time
		)

		# Validate order
		# self._order_validation(order)

		self.broker.positions.append(order)

		return order


	def createOrder(self,
		product, lotsize, direction, account_id,
		order_type, entry_range, entry_price,
		sl_range, tp_range, sl_price, tp_price
	):
		if direction == tl.LONG:
			price = self.broker.getAsk(product)
		else:
			price = -self.broker.getBid(product)

		# Calc entry
		entry = None
		if entry_price:
			entry = np.around(entry_price, 5)
		elif entry_range:
			if order_type == tl.LIMIT_ORDER:
				entry = np.around(abs(tl.convertToPrice(entry_range) - price), 5)
			elif order_type == tl.STOP_ORDER:
				entry = np.around(abs(tl.convertToPrice(entry_range) + price), 5)
			
		# Calc stop loss
		sl = None
		if sl_price:
			sl = np.around(sl_price, 5)
		elif sl_range:
			if direction == tl.LONG:
				sl = np.around(entry - tl.convertToPrice(sl_range), 5)
			else:
				sl = np.around(entry + tl.convertToPrice(sl_range), 5)

		# Calc take profit
		tp = None
		if tp_price:
			tp = np.around(tp_price, 5)
		elif tp_range:
			if direction == tl.LONG:
				tp = np.around(entry + tl.convertToPrice(tp_range), 5)
			else:
				tp = np.around(entry - tl.convertToPrice(tp_range), 5)


		open_time = int(self.broker.getTimestamp(product))
		

		order_id = self.broker.generateReference()
		order = tl.BacktestOrder(self.broker,
			order_id, account_id, product, 
			order_type, direction, lotsize,
			entry_price=entry, sl=sl, tp=tp,
			open_time=open_time
		)

		# Validate order
		# self._order_validation(order)

		self.broker.orders.append(order)

		return order


	def modifyPosition(self, pos, sl_range, tp_range, sl_price, tp_price):
		if sl_range is not None:
			if pos.direction == tl.LONG:
				pos.sl = round(pos.entry_price - tl.utils.convertToPrice(sl_range), 5)
			else:
				pos.sl = round(pos.entry_price + tl.utils.convertToPrice(sl_range), 5)
		elif sl_price is not None:
			pos.sl = sl_price

		if tp_range is not None:
			if pos.direction == tl.LONG:
				pos.tp = round(pos.entry_price + tl.utils.convertToPrice(tp_range), 5)
			else:
				pos.tp = round(pos.entry_price - tl.utils.convertToPrice(tp_range), 5)
		elif tp_price is not None:
			pos.tp = tp_price

		return pos


	def deletePosition(self, pos, lotsize):
		if lotsize >= pos.lotsize:
			if pos.direction == tl.LONG:
				pos.close_price = self.broker.getBid(pos.product)
			else:
				pos.close_price = self.broker.getAsk(pos.product)
			pos.close_time = self.broker.getTimestamp(pos.product)

			result = pos

			# Delete position from broker positions
			del self.broker.positions[self.broker.positions.index(pos)]

		elif lotsize <= 0:
			# Raise error
			raise tl.error.OrderException('Position close size must be greater than 0.')

		else:
			cpy = tl.BacktestPosition.fromDict(self, self)
			cpy.lotsize = lotsize
			if pos.direction == tl.LONG:
				cpy.close_price = self.broker.getBid(pos.product)
			else:
				cpy.close_price = self.broker.getAsk(pos.product)
			cpy.close_time =  self.broker.getTimestamp(pos.product)
			
			result = cpy

			pos.lotsize -= lotsize

		return result


	def modifyOrder(self, order, lotsize, entry_range, entry_price, sl_range, tp_range, sl_price, tp_price):
		if lotsize is not None:
			order.lotsize = lotsize

		# Convert to price
		if entry_range is not None:
			if direction == tl.LONG:
				order.entry_price = round(order.entry_price + tl.utils.convertToPrice(entry_range), 5)
			else:
				order.entry_price = round(order.entry_price - tl.utils.convertToPrice(entry_range), 5)
		elif entry_price is not None:
			order.entry_price = entry_price

		if sl_range is not None:
			if direction == tl.LONG:
				order.sl = round(self.entry - tl.utils.convertToPrice(sl_range), 5)
			else:
				order.sl = round(self.entry + tl.utils.convertToPrice(sl_range), 5)
		elif sl_price is not None:
			order.sl = sl_price

		if tp_range is not None:
			if direction == tl.LONG:
				order.tp = round(self.entry + tl.utils.convertToPrice(tp_range), 5)
			else:
				order.tp = round(self.entry - tl.utils.convertToPrice(tp_range), 5)
		elif tp_price is not None:
			order.tp = tp_price

		return order


	def deleteOrder(self, order):
		order.close_time = self.broker.getTimestamp(order.product)
		del self.broker.orders[self.broker.orders.index(order)]

		return order


	def handleTransaction(self, res):
		for k, v in res.items():
			for func in self.broker.ontrade_subs:
				func(
					BrokerItem({
						'reference_id': k,
						'type': v.get('type'),
						'item': v.get('item')
					})
				)


	def createTransactionItem(self, ref_id, timestamp, order_type, prev_item, new_item):
		item = {
			'id': ref_id,
			'timestamp': timestamp,
			'type': order_type,
			'item': {
				'prev': prev_item,
				'new': new_item
			}
		}

		self.result.append(item)


	def createDrawing(self, timestamp, layer, drawing):
		item = {
			'timestamp': timestamp,
			'type': tl.CREATE_DRAWING,
			'item': drawing
		}

		self.result.append(item)


	def clearDrawingLayer(self, timestamp, layer):
		item = {
			'timestamp': timestamp,
			'type': tl.CLEAR_DRAWING_LAYER,
			'item': layer
		}

		self.result.append(item)


	def deleteAllDrawings(self, timestamp):
		item = {
			'timestamp': timestamp,
			'type': tl.CLEAR_ALL_DRAWINGS,
			'item': None
		}

		self.result.append(item)


	def createInfoItem(self, product, period, timestamp, item):
		timestamp = int(timestamp)

		if product not in self.info:
			self.info[product] = {}
		if period not in self.info[product]:
			self.info[product][period] = {}
		if not timestamp in self.info[product][period]:
			self.info[product][period][timestamp] = []
		self.info[product][period][timestamp].append(item)


	def createLogItem(self, timestamp, item):
		item = {
			'timestamp': timestamp,
			'type': tl.CREATE_LOG,
			'item': item
		}

		self.result.append(item)


	def createReport(self, name, columns):
		self.reports[name] = pd.DataFrame(columns=columns)


	def report(self, name, *data):
		reports = self.broker.strategy.reports
		if name in reports:
			reports[name].loc[reports[name].shape[0]] = set(map(str, data))


	def createOrderPosition(self, order):
		if order.order_type == tl.LIMIT_ORDER:
			order_type = tl.LIMIT_ENTRY
		elif order.order_type == tl.STOP_ORDER:
			order_type = tl.STOP_ENTRY
		else:
			order_type = tl.MARKET_ENTRY

		pos = tl.BacktestPosition.fromOrder(self.broker, order)
		pos.open_time = self.broker.getTimestamp(pos.product)
		pos.order_type = order_type
		self.broker.positions.append(pos)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': pos.open_time,
				'type': pos.order_type,
				'accepted': True,
				'item': pos
			}
		}

		return [pos], res


	def handleOrders(self, product, timestamp, ohlc):
		ask = ohlc[:4]
		bid = ohlc[8:]

		for order in self.broker.getAllOrders():
			if order.product != product or not order.isBacktest():
				continue

			if order.order_type == tl.LIMIT_ORDER:
				if order.direction == tl.LONG:
					if ask[2] <= order.entry_price:
						# Enter Order Position LONG
						result, res = self.createOrderPosition(order)

						# Close Order
						order.close_price = order.entry_price
						order.close_time = timestamp

						# Delete Order
						for i in range(len(self.broker.orders)-1,-1,-1):
							if self.broker.orders[i].order_id == order.order_id:
								del self.broker.orders[i]

						# On Trade
						self.handleTransaction(res)
						for i in res:
							item = res[i]
							self.createTransactionItem(i, timestamp, item['item']['order_type'], copy(order), copy(item['item']))

				else:
					if bid[1] >= order.entry_price:
						# Enter Order Position LONG
						result, res = self.createOrderPosition(order)

						# Close Order
						order.close_price = order.entry_price
						order.close_time = timestamp
						
						# Delete Order
						for i in range(len(self.broker.orders)-1,-1,-1):
							if self.broker.orders[i].order_id == order.order_id:
								del self.broker.orders[i]

						# On Trade
						self.handleTransaction(res)
						for i in res:
							item = res[i]
							self.createTransactionItem(i, timestamp, item['item']['order_type'], copy(order), copy(item['item']))

			elif order.order_type == tl.STOP_ORDER:
				if order.direction == tl.LONG:
					if ask[1] >= order.entry_price:
						# Enter Order Position LONG
						result, res = self.createOrderPosition(order)

						# Close Order
						order.close_price = order.entry_price
						order.close_time = timestamp

						# Delete Order
						for i in range(len(self.broker.orders)-1,-1,-1):
							if self.broker.orders[i].order_id == order.order_id:
								del self.broker.orders[i]

						# On Trade
						self.handleTransaction(res)
						for i in res:
							item = res[i]
							self.createTransactionItem(i, timestamp, item['item']['order_type'], copy(order), copy(item['item']))

				else:
					if bid[2] <= order.entry_price:
						# Enter Order Position LONG
						result, res = self.createOrderPosition(order)

						# Close Order
						order.close_price = order.entry_price
						order.close_time = timestamp

						# Delete Order
						for i in range(len(self.broker.orders)-1,-1,-1):
							if self.broker.orders[i].order_id == order.order_id:
								del self.broker.orders[i]

						# On Trade
						self.handleTransaction(res)
						for i in res:
							item = res[i]
							self.createTransactionItem(i, timestamp, item['item']['order_type'], copy(order), copy(item['item']))


	def handleStopLoss(self, product, timestamp, ohlc):
		ask = ohlc[:4]
		bid = ohlc[8:]
		for pos in self.broker.getAllPositions():
			if pos.product != product or not pos.sl or not pos.isBacktest():
				continue

			if ((pos.direction == tl.LONG and bid[2] <= pos.sl) or
				(pos.direction == tl.SHORT and ask[1] >= pos.sl)):
				
				prev_item = copy(pos)
				
				# Close Position
				pos.close_price = pos.sl
				pos.close_time = timestamp
				# Delete Position
				del self.broker.positions[self.broker.positions.index(pos)]

				ref_id = self.broker.generateReference()
				res = {
					ref_id: {
						'timestamp': timestamp,
						'type': tl.STOP_LOSS,
						'accepted': True,
						'item': pos
					}
				}

				# On Trade
				self.handleTransaction(res)
				self.createTransactionItem(ref_id, timestamp, tl.STOP_LOSS, prev_item, copy(pos))


	def handleTakeProfit(self, product, timestamp, ohlc):
		ask = ohlc[:4]
		bid = ohlc[8:]

		for pos in self.broker.positions:
			if (pos.product != product or not pos.tp or not pos.isBacktest()):
				continue

			if ((pos.direction == tl.LONG and bid[1] >= pos.tp) or
				(pos.direction == tl.SHORT and ask[2] <= pos.tp)):
				
				prev_item = copy(pos)

				# Close Position
				pos.close_price = pos.tp
				pos.close_time = timestamp

				# Delete Position
				del self.broker.positions[self.broker.positions.index(pos)]

				ref_id = self.broker.generateReference()
				res = {
					ref_id: {
						'timestamp': timestamp,
						'type': tl.TAKE_PROFIT,
						'accepted': True,
						'item': pos
					}
				}

				# On Trade
				self.handleTransaction(res)
				self.createTransactionItem(ref_id, timestamp, tl.TAKE_PROFIT, prev_item, copy(pos))


	def _process_chart_data(self, charts, start, end, spread=None):
		periods = []
		dataframes = []
		indicator_dataframes = []
		all_ts = []
		tick_df = None

		start_time = time.time()
		start_ts = tl.convertTimeToTimestamp(start)

		for i in range(len(charts)):
			periods.append([])
			dataframes.append([])
			indicator_dataframes.append([])

			chart = charts[i]
			sorted_periods = sorted(chart.periods, key=lambda x: tl.period.getPeriodOffsetSeconds(x))

			data = chart.quickDownload(
				tl.period.ONE_MINUTE, 
				start, end, set_data=False
			)
			tick_df = data.copy()

			# Set Tick Artificial Spread
			if isinstance(spread, float):
				half_spread = tl.utils.convertToPrice(spread / 2)
				# Ask
				tick_df.values[:, :4] = tick_df.values[:, 4:8] + half_spread
				# Bid
				tick_df.values[:, 8:] = tick_df.values[:, 4:8] - half_spread

			if data.size > 0:
				first_data_ts = datetime.utcfromtimestamp(data.index.values[0]).replace(
					hour=0, minute=0, second=0, microsecond=0
				).timestamp()
				df_off = 0
				for j in range(len(sorted_periods)):
					period = sorted_periods[j]
					df = data.copy()

					first_ts = df.index.values[0] - ((df.index.values[0] - first_data_ts) % tl.period.getPeriodOffsetSeconds(period))
					next_ts = tl.utils.getNextTimestamp(period, first_ts, now=df.index.values[0])
					period_ohlc = np.array(data.values[0], dtype=float)
					bar_ts = []
					bar_end_ts = []

					indicators = [ind for ind in chart.indicators.values() if ind.period == period]

					# Process Tick Data
					for x in range(1, data.shape[0]):
						last_ts = df.index.values[x-1]
						ts = df.index.values[x]
						prev = df.iloc[x-1]
						curr = df.iloc[x]

						# New Bar
						if ts >= next_ts:
							if len(bar_ts) == 0 and j == 0:
								df_off = x

							bar_end_ts.append(last_ts)

							bar_ts.append(next_ts - tl.period.getPeriodOffsetSeconds(period))
							next_ts = tl.utils.getNextTimestamp(period, next_ts, now=ts)
							

						# Intra Bar
						else:
							if prev[1] > curr[1]:
								curr[1] = prev[1]
							if prev[5] > curr[5]:
								curr[5] = prev[5]
							if prev[9] > curr[9]:
								curr[9] = prev[9]
							if prev[2] < curr[2]:
								curr[2] = prev[2]
							if prev[6] < curr[6]:
								curr[6] = prev[6]
							if prev[10] < curr[10]:
								curr[10] = prev[10]

							curr[0] = prev[0]
							curr[4] = prev[4]
							curr[8] = prev[8]

						df.iloc[x] = curr

					bar_end_ts.append(ts)
					bar_ts.append(next_ts - tl.period.getPeriodOffsetSeconds(period))

					prev_bars_df = chart.quickDownload(
						period, end=start, count=1000, set_data=False
					)

					# Get Completed Bars DataFrame
					bars_df = df.loc[df.index.intersection(bar_end_ts)]
					bars_df.index = bar_ts

					bars_df = pd.concat((prev_bars_df, bars_df))

					# Set Artificial Spread
					if isinstance(spread, float):
						half_spread = tl.utils.convertToPrice(spread / 2)
						# Ask
						df.values[:, :4] = df.values[:, 4:8] + half_spread
						# Bid
						df.values[:, 8:] = df.values[:, 4:8] - half_spread

						# Ask
						bars_df.values[:, :4] = bars_df.values[:, 4:8] + half_spread
						# Bid
						bars_df.values[:, 8:] = bars_df.values[:, 4:8] - half_spread
					
					bars_start_idx = prev_bars_df.shape[0]

					# Process Indicators
					period_indicator_arrays = []
					indicator_dataframes[i].append([])
					for y in range(len(indicators)):
						ind = indicators[y]

						# Calculate initial bars
						ind.calculate(bars_df.iloc[:prev_bars_df.shape[0]], 0)

						indicator_array = None
						ind_asks = None
						ind_mids = None
						ind_bids = None
						bars_idx = 0

						for x in range(df.shape[0]):
							c_df = bars_df.iloc[:bars_start_idx+bars_idx+1].copy()
							c_df.iloc[-1] = df.iloc[x]
						
							# Calc indicator
							asks = ind._perform_calculation('ask', c_df.values[:, :4], bars_start_idx+bars_idx)
							mids = ind._perform_calculation('mid', c_df.values[:, 4:8], bars_start_idx+bars_idx)
							bids = ind._perform_calculation('bid', c_df.values[:, 8:], bars_start_idx+bars_idx)

							if ind_asks is None and ind_bids is None:
								ind_asks = np.zeros((bars_df.shape[0], len(asks)), dtype=float)
								ind_asks[:prev_bars_df.shape[0]] = ind._asks
								ind_mids = np.zeros((bars_df.shape[0], len(mids)), dtype=float)
								ind_mids[:prev_bars_df.shape[0]] = ind._mids
								ind_bids = np.zeros((bars_df.shape[0], len(bids)), dtype=float)
								ind_bids[:prev_bars_df.shape[0]] = ind._bids

							if bars_start_idx+bars_idx < bars_df.shape[0]:
								ind_asks[bars_start_idx+bars_idx] = asks
								ind_mids[bars_start_idx+bars_idx] = mids
								ind_bids[bars_start_idx+bars_idx] = bids

								ind._asks = ind_asks[:bars_start_idx+bars_idx+1]
								ind._mids = ind_mids[:bars_start_idx+bars_idx+1]
								ind._bids = ind_bids[:bars_start_idx+bars_idx+1]

								if indicator_array is None:
									indicator_array = np.zeros((df.shape[0], len(asks)*3), dtype=float)

								indicator_array[x] = np.concatenate((asks, mids, bids))

							if bars_idx < len(bar_end_ts) and df.index.values[x] == bar_end_ts[bars_idx]:
								bars_idx += 1
							
						indicator_dataframes[i][j].append(pd.DataFrame(index=df.index.copy(), data=indicator_array))

					all_ts = np.concatenate((all_ts, df.index.values))
					chart._data[period] = bars_df
					chart._set_idx(period, bars_start_idx-1)

					periods[i].append(period)
					# dataframes[i].append(df.iloc[start_idx:])
					dataframes[i].append(df)

		all_ts = np.unique(np.sort(all_ts))
		return all_ts, periods, dataframes, indicator_dataframes, tick_df


	def _event_loop(self, charts, all_ts, periods, dataframes, indicator_dataframes, tick_df):
		start_time = time.time()

		# For Each Timestamp
		for x in range(all_ts.size):
			# For Each Chart
			for y in range(len(charts)):
				# For Each Period
				for z in range(len(periods[y])):
					# Set index of current chart period
					timestamp = all_ts[x]
					tick_timestamp = timestamp
					chart = charts[y]
					period = periods[y][z]
					idx = chart._idx[period]
					bar_end = False

					period_data = chart._data[period]
					if (
						idx+1 < period_data.shape[0] and
						x+1 < all_ts.shape[0] and
						all_ts[x+1] >= period_data.index.values[idx+1]
					):
						bar_end = True
						tick_timestamp = period_data.index.values[idx]

					df = dataframes[y][z]
					period_data.iloc[idx] = df.iloc[x]
					chart.timestamps[period] = period_data.index.values[:idx+1][::-1]

					c_data = period_data.values[:idx+1]

					chart.asks[period] = c_data[:, :4][::-1]
					chart.mids[period] = c_data[:, 4:8][::-1]
					chart.bids[period] = c_data[:, 8:][::-1]
					# Add offset for real time because ONE MINUTE bars are being used for tick data
					chart._end_timestamp = timestamp + tl.period.getPeriodOffsetSeconds(tl.period.ONE_MINUTE)

					ohlc = chart.getLastOHLC(period)
					indicators = [ind for ind in chart.indicators.values() if ind.period == period]
					for i in range(len(indicators)):
						ind = indicators[i]
						ind_df = indicator_dataframes[y][z][i]

						result_size = int(ind_df.shape[1]/3)
						ind._asks[idx] = ind_df.values[x, :result_size]
						ind._mids[idx] = ind_df.values[x, result_size:result_size*2]
						ind._bids[idx] = ind_df.values[x, result_size*2:]
						ind._set_idx(idx)

					# If lowest period, do position/order check
					if z == 0:
						c_tick = tick_df.values[x]
						self.handleOrders(chart.product, chart._end_timestamp, c_tick)
						self.handleStopLoss(chart.product, chart._end_timestamp, c_tick)
						self.handleTakeProfit(chart.product, chart._end_timestamp, c_tick)

					# Call threaded ontick functions
					if period in chart._subscriptions:
						for func in chart._subscriptions[period]:
							tick = BrokerItem({
								'chart': chart, 
								'timestamp': tick_timestamp,
								'period': period, 
								'ask': ohlc[:4],
								'mid': ohlc[4:8],
								'bid': ohlc[8:],
								'bar_end': bar_end
							})

							self.broker.strategy.setTick(tick)
							func(tick)

					if bar_end:
						chart._idx[period] = idx + 1

		print('Event Loop Complete {:.2f}s'.format(time.time() - start_time), flush=True)

		for i in charts:
			for period in chart._data:
				print(f'{period}: {chart._data[period].index.values[-1]}\n{chart._data[period].values[-1]}', flush=True)


	def performBacktest(self, mode, start=None, end=None, spread=None):
		# Get timestamps
		start_ts = tl.convertTimeToTimestamp(start)
		end_ts = tl.convertTimeToTimestamp(end)

		# Process chart data
		charts = copy(self.broker.charts)
		all_ts, periods, dataframes, indicator_dataframes, tick_df = self._process_chart_data(charts, start, end, spread=spread)

		# Run event loop
		self._event_loop(charts, all_ts, periods, dataframes, indicator_dataframes, tick_df)


class IGBacktester(Backtester):

	def __init__(self, broker, sort_reversed=False):
		super(IGBacktester, self).__init__(broker)


	def createPosition(self,
		product, lotsize, direction,
		account_id, entry_range, entry_price,
		sl_range, tp_range, sl_price, tp_price
	):
		result = super(IGBacktester, self).createPosition(
			product, lotsize, direction,
			account_id, entry_range, entry_price,
			sl_range, tp_range, sl_price, tp_price
		)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.open_time,
				'type': tl.MARKET_ENTRY,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.open_time, tl.MARKET_ENTRY, None, copy(result))

		return result


	def createOrder(self,
		product, lotsize, direction, account_id,
		order_type, entry_range, entry_price,
		sl_range, tp_range, sl_price, tp_price
	):
		result = super(IGBacktester, self).createOrder(
			product, lotsize, direction, account_id,
			order_type, entry_range, entry_price,
			sl_range, tp_range, sl_price, tp_price
		)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.open_time,
				'type': result.order_type,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.open_time, result.order_type, None, copy(result))

		return result


	def modifyPosition(self, pos, sl_range, tp_range, sl_price, tp_price):
		prev_item = copy(pos)

		result = super(IGBacktester, self).modifyPosition(
			pos, sl_range, tp_range, sl_price, tp_price
		)

		ref_id = self.broker.generateReference()
		timestamp = self.broker.getTimestamp(result.product)
		res = {
			ref_id: {
				'timestamp': timestamp,
				'type': tl.MODIFY,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, timestamp, tl.MODIFY, prev_item, copy(result))

		return result


	def deletePosition(self, pos, lotsize):
		prev_item = copy(pos)

		result = super(IGBacktester, self).deletePosition(pos, lotsize)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.close_time,
				'type': tl.POSITION_CLOSE,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.close_time, tl.POSITION_CLOSE, prev_item, copy(result))

		return result


	def modifyOrder(self, order, lotsize, entry_range, entry_price, sl_range, tp_range, sl_price, tp_price):
		prev_item = copy(pos)

		result = super(IGBacktester, self).modifyOrder(
			order, lotsize, entry_range, entry_price, sl_range, tp_range, sl_price, tp_price
		)

		ref_id = self.broker.generateReference()
		timestamp = self.broker.getTimestamp(result.product)
		res = {
			ref_id: {
				'timestamp': timestamp,
				'type': tl.MODIFY,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, timestamp, tl.MODIFY, prev_item, copy(result))

		return result


	def deleteOrder(self, order):
		prev_item = copy(pos)

		result = super(IGBacktester, self).deleteOrder(order)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.close_time,
				'type': tl.ORDER_CANCEL,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.close_time, tl.ORDER_CANCEL, prev_item, copy(result))

		return result



class OandaBacktester(Backtester):

	def __init__(self, broker, sort_reversed=False):
		super(OandaBacktester, self).__init__(broker)
		self._sort_reversed = sort_reversed


	def _net_off(self, account_id, direction, lotsize):
		positions = sorted(
			[
				pos for pos in self.broker.getAllPositions(account_id=account_id) 
				if pos.direction != direction
				if pos.isBacktest()
			],
			key=lambda x: x.open_time,
			reverse=self._sort_reversed
		)

		remaining = lotsize
		result = []
		for pos in positions:
			delete_size = min(pos.lotsize, remaining)
			result.append(pos.close(lotsize=delete_size))
			remaining -= delete_size
			if remaining <= 0:
				return remaining, result

		return remaining, result


	def createPosition(self,
		product, lotsize, direction,
		account_id, entry_range, entry_price,
		sl_range, tp_range, sl_price, tp_price
	):
		# Handle closing any opposite positions first
		remaining, _ = self._net_off(account_id, direction, lotsize)
		result = None
		if remaining > 0:
			# Create Position
			result = super(OandaBacktester, self).createPosition(
				product, remaining, direction, 
				account_id, entry_range, entry_price,
				None, None, None, None
			)

			res = {}
			ref_id = self.broker.generateReference()
			entry_res = {
				ref_id: {
					'timestamp': result.open_time,
					'type': tl.MARKET_ENTRY,
					'accepted': True,
					'item': result
				}
			}
			res.update(entry_res)

			# Add SL
			if sl_range or sl_price:
				result = super(OandaBacktester, self).modifyPosition(
					result, sl_range, None, sl_price, None
				)
				sl_res = {
					self.broker.generateReference(): {
						'timestamp': self.broker.getTimestamp(result.product),
						'type': tl.MODIFY,
						'accepted': True,
						'item': result
					}
				}
				res.update(sl_res)

			# Add TP
			if tp_range or tp_price:
				result = super(OandaBacktester, self).modifyPosition(
					result, None, tp_range, None, tp_price
				)
				tp_res = {
					self.broker.generateReference(): {
						'timestamp': self.broker.getTimestamp(result.product),
						'type': tl.MODIFY,
						'accepted': True,
						'item': result
					}
				}
				res.update(tp_res)

			self.handleTransaction(res)
			self.createTransactionItem(ref_id, result.open_time, tl.MARKET_ENTRY, None, copy(result))

		return result


	def createOrder(self,
		product, lotsize, direction, account_id,
		order_type, entry_range, entry_price,
		sl_range, tp_range, sl_price, tp_price
	):
		result = super(OandaBacktester, self).createOrder(
			product, lotsize, direction, account_id,
			order_type, entry_range, entry_price,
			sl_range, tp_range, sl_price, tp_price
		)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.open_time,
				'type': result.order_type,
				'accepted': True,
				'item': result
			}
		}
		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.open_time, result.order_type, None, copy(result))

		return result


	def modifyPosition(self, pos, sl_range, tp_range, sl_price, tp_price):
		prev_item = copy(pos)
		res = {}
		result = pos
		if pos.sl != sl_price:
			result = super(OandaBacktester, self).modifyPosition(
				pos, None, None, sl_price, None
			)
			sl_res = {
				self.broker.generateReference(): {
					'timestamp': self.broker.getTimestamp(pos.product),
					'type': tl.MODIFY,
					'accepted': True,
					'item': result
				}
			}
			res.update(sl_res)


		if pos.tp != tp_price:
			result = super(OandaBacktester, self).modifyPosition(
				pos, None, None, None, tp_price
			)
			tp_res = {
				self.broker.generateReference(): {
					'timestamp': self.broker.getTimestamp(pos.product),
					'type': tl.MODIFY,
					'accepted': True,
					'item': result
				}
			}
			res.update(tp_res)

		self.handleTransaction(res)
		self.createTransactionItem(
			self.broker.generateReference(), 
			self.broker.getTimestamp(pos.product), 
			tl.MODIFY, prev_item, copy(result)
		)

		return result


	def deletePosition(self, pos, lotsize):
		prev_item = copy(pos)

		result = super(OandaBacktester, self).deletePosition(pos, lotsize)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.close_time,
				'type': tl.POSITION_CLOSE,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.close_time, tl.POSITION_CLOSE, prev_item, copy(result))

		return result


	def modifyOrder(self, order, lotsize, entry_range, entry_price, sl_range, tp_range, sl_price, tp_price):
		prev_item = copy(order)

		new_order = tl.BacktestOrder.fromDict(self.broker, order)
		new_order.order_id = self.broker.generateReference()

		order.close_time = self.broker.getTimestamp(order.product)
		del self.broker.orders[self.broker.orders.index(order)]

		res = {}

		cancel_ref_id = self.broker.generateReference()
		cancel_res = {
			cancel_ref_id: {
				'timestamp': order.close_time,
				'type': tl.ORDER_CANCEL,
				'accepted': True,
				'item': order
			}
		}
		res.update(cancel_res)

		result = super(OandaBacktester, self).modifyOrder(
			new_order, lotsize, entry_range, entry_price, sl_range, tp_range, sl_price, tp_price
		)

		self.broker.orders.append(result)

		create_ref_id = self.broker.generateReference()
		timestamp = self.broker.getTimestamp(order.product)
		modify_res = {
			create_ref_id: {
				'timestamp': timestamp,
				'type': result.order_type,
				'accepted': True,
				'item': result
			}
		}
		res.update(modify_res)
		
		self.handleTransaction(res)
		self.createTransactionItem(cancel_ref_id, timestamp, tl.ORDER_CANCEL, prev_item, order)
		self.createTransactionItem(create_ref_id, timestamp, result.order_type, None, copy(result))

		return result


	def deleteOrder(self, order):
		prev_item = copy(order)

		result = super(OandaBacktester, self).deleteOrder(order)

		ref_id = self.broker.generateReference()
		res = {
			ref_id: {
				'timestamp': result.close_time,
				'type': tl.ORDER_CANCEL,
				'accepted': True,
				'item': result
			}
		}

		self.handleTransaction(res)
		self.createTransactionItem(ref_id, result.close_time, tl.ORDER_CANCEL, prev_item, copy(result))

		return result


	def createOrderPosition(self, order):
		if order.order_type == tl.LIMIT_ORDER:
			order_type = tl.LIMIT_ENTRY
		elif order.order_type == tl.STOP_ORDER:
			order_type = tl.STOP_ENTRY
		else:
			order_type = tl.MARKET_ENTRY

		remaining, _ = self._net_off(order.account_id, order.direction, order.lotsize)
		result = []
		res = {}
		if remaining > 0:
			pos = tl.BacktestPosition.fromOrder(self.broker, order)
			pos.order_id = self.broker.generateReference()
			pos.order_type = order_type
			pos.lotsize = remaining
			pos.open_time = self.broker.getTimestamp(order.product)
			self.broker.positions.append(pos)
			result.append(pos)

			res.update({
				self.broker.generateReference(): {
					'timestamp': pos.open_time,
					'type': pos.order_type,
					'accepted': True,
					'item': pos
				}
			})

		return result, res


'''
Imports
'''
from .. import app as tl
from .broker import BrokerItem
