import pandas as pd
import numpy as np
import time
import sys
from copy import copy
from datetime import datetime


def _process_chart_data(charts, start, end, spread=None):
	cdef list periods = []
	cdef list dataframes = []
	cdef list indicator_dataframes = []
	cdef list offsets = []
	cdef list sorted_periods
	cdef list period_indicator_arrays
	cdef double[:] all_ts
	tick_df = None

	cdef list bars_df_list = []
	cdef list ind_bars_df_list = []

	start_time = time.time()
	start_ts = utils.convertTimeToTimestamp(start)

	cdef int i
	cdef int i_size = len(charts)
	cdef int j
	cdef int j_size
	cdef int x
	cdef int x_size

	cdef double ts
	cdef double last_ts
	cdef double next_ts
	cdef double[:] prev
	cdef double[:] curr
	# cdef list asks
	# cdef list mids
	# cdef list bids
	# cdef double[:,:] indicator_array
	# cdef double[:,:] ind_asks
	# cdef double[:,:] ind_mids
	# cdef double[:,:] ind_bids

	cdef int bars_idx
	cdef int bars_start_idx


	for i in range(i_size):
		periods.append([])
		dataframes.append([])
		indicator_dataframes.append([])
		offsets.append([])

		bars_df_list.append([])
		ind_bars_df_list.append([])

		chart = charts[i]
		# sorted_periods = sorted(chart.periods, key=lambda x: tl_period.getPeriodOffsetSeconds(x))
		sorted_periods = copy(chart._priority)

		if tl_period.TICK in sorted_periods:
			del sorted_periods[sorted_periods.index(tl_period.TICK)]

		j_size = len(sorted_periods)
		
		data = chart.quickDownload(
			tl_period.ONE_MINUTE, 
			start, end, set_data=False
		)

		all_ts = data.index.values
		tick_df = data.copy()

		if data.size > 0:
			# Set Tick Artificial Spread
			if isinstance(spread, float):
				half_spread = utils.convertToPrice(spread / 2)
				# Ask
				tick_df.values[:, :4] = tick_df.values[:, 4:8] + half_spread
				# Bid
				tick_df.values[:, 8:] = tick_df.values[:, 4:8] - half_spread

			first_data_ts = datetime.utcfromtimestamp(data.index.values[0]).replace(
				hour=0, minute=0, second=0, microsecond=0
			).timestamp()
			df_off = 0
			for j in range(j_size):
				period = sorted_periods[j]
				df = data.copy()

				first_ts = df.index.values[0] - ((df.index.values[0] - first_data_ts) % tl_period.getPeriodOffsetSeconds(period))
				next_ts = utils.getNextTimestamp(period, first_ts, now=df.index.values[0])
				period_ohlc = np.array(data.values[0], dtype=float)
				bar_ts = []
				bar_end_ts = []

				indicators = [ind for ind in chart.indicators.values() if ind.period == period]

				# Process Tick Data
				for x in range(1, data.shape[0]):
					last_ts = df.index.values[x-1]
					ts = df.index.values[x]
					prev = df.values[x-1]
					curr = df.values[x]

					# New Bar
					if ts >= next_ts:
						if len(bar_ts) == 0 and j == 0:
							df_off = x

						bar_end_ts.append(last_ts)

						bar_ts.append(next_ts - tl_period.getPeriodOffsetSeconds(period))
						next_ts = utils.getNextTimestamp(period, next_ts, now=ts)
						

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

					df.values[x] = curr

				# if ts + tl_period.getPeriodOffsetSeconds(tl_period.ONE_MINUTE) >= next_ts:
				bar_end_ts.append(ts)
				bar_ts.append(next_ts - tl_period.getPeriodOffsetSeconds(period))

				df = df.iloc[df_off:]
				# test_bars_df = self._construct_bars(period, df)

				prev_bars_df = chart.quickDownload(
					period, end=utils.convertTimestampToTime(df.index.values[0]), 
					count=2000, set_data=False
				)

				# # Get Completed Bars DataFrame
				bars_df = df.loc[df.index.intersection(bar_end_ts)]
				bar_end_ts = bar_end_ts[-bars_df.shape[0]:]
				bar_ts = bar_ts[-bars_df.shape[0]:]
				bars_df.index = bar_ts
				prev_bars_df = prev_bars_df.loc[prev_bars_df.index < bars_df.index.values[0]]
				bars_df = pd.concat((prev_bars_df, bars_df))

				# Set Artificial Spread
				if isinstance(spread, float):
					half_spread = utils.convertToPrice(spread / 2)
					# Ask
					df.values[:, :4] = df.values[:, 4:8] + half_spread
					# Bid
					df.values[:, 8:] = df.values[:, 4:8] - half_spread

					# Ask
					bars_df.values[:, :4] = bars_df.values[:, 4:8] + half_spread
					# Bid
					bars_df.values[:, 8:] = bars_df.values[:, 4:8] - half_spread
				
				bars_start_idx = prev_bars_df.shape[0]

				df = df.round(decimals=5)
				bars_df = bars_df.round(decimals=5)

				# Process Indicators
				period_indicator_arrays = []
				indicator_dataframes[i].append([])
				ind_bars_df_list[i].append([])
				for y in range(len(indicators)):
					ind = indicators[y]

					# Calculate initial bars
					ind.calculate(bars_df.iloc[:prev_bars_df.shape[0]], 0)

					indicator_array = None
					ind_asks = None
					ind_mids = None
					ind_bids = None
					bars_idx = 0

					x_size = df.shape[0]
					for x in range(x_size):
						c_df = bars_df.values[:bars_start_idx+bars_idx+1].copy()
						c_df[-1] = df.values[x]
					
						# Calc indicator
						asks = ind._perform_calculation(0, c_df[:, :4], bars_start_idx+bars_idx)
						mids = ind._perform_calculation(1, c_df[:, 4:8], bars_start_idx+bars_idx)
						bids = ind._perform_calculation(2, c_df[:, 8:], bars_start_idx+bars_idx)

						if ind_asks is None and ind_bids is None:
							ind_asks = np.zeros((bars_df.shape[0], len(asks)), dtype=np.float64)
							ind_asks[:prev_bars_df.shape[0]] = ind._asks
							ind_mids = np.zeros((bars_df.shape[0], len(mids)), dtype=np.float64)
							ind_mids[:prev_bars_df.shape[0]] = ind._mids
							ind_bids = np.zeros((bars_df.shape[0], len(bids)), dtype=np.float64)
							ind_bids[:prev_bars_df.shape[0]] = ind._bids

						if bars_start_idx+bars_idx < bars_df.shape[0]:
							ind_asks[bars_start_idx+bars_idx] = asks
							ind_mids[bars_start_idx+bars_idx] = mids
							ind_bids[bars_start_idx+bars_idx] = bids

							ind._asks = ind_asks[:bars_start_idx+bars_idx+1]
							ind._mids = ind_mids[:bars_start_idx+bars_idx+1]
							ind._bids = ind_bids[:bars_start_idx+bars_idx+1]

							if indicator_array is None:
								indicator_array = np.zeros((df.shape[0], len(asks)*3), dtype=np.float64)

							indicator_array[x] = np.concatenate((asks, mids, bids))
						#bars_idx+1 < len(bar_end_ts) and 
						if df.index.values[x] == bar_end_ts[bars_idx]:
							bars_idx += 1

					ind_bars_df_list[i][j].append(np.concatenate((ind._asks, ind._mids, ind._bids), axis=1))
					indicator_dataframes[i][j].append(pd.DataFrame(index=df.index.copy(), data=indicator_array))
					
				chart._data[period] = bars_df
				chart._set_idx(period, bars_start_idx)
				
				chart.timestamps[period] = chart._data[period].index.values[:bars_start_idx][::-1]
				c_data = chart._data[period].values[:bars_start_idx]
				chart.asks[period] = c_data[:, :4][::-1]
				chart.mids[period] = c_data[:, 4:8][::-1]
				chart.bids[period] = c_data[:, 8:][::-1]

				periods[i].append(period)
				dataframes[i].append(df)
				offsets[i].append(df_off)

				bars_df_list[i].append(bars_df)

	all_ts = np.unique(np.sort(all_ts))
	print('Processing finished {:.2f}s'.format(time.time() - start_time))
	sys.stdout.flush()
	return all_ts, periods, dataframes, indicator_dataframes, tick_df, offsets, bars_df_list, ind_bars_df_list, bars_start_idx


def _event_loop(self, charts, double[:] all_ts, list periods, list dataframes, list indicator_dataframes, tick_df, list offsets):
	print('Start Event Loop.')
	start_time = time.time()

	# Iterator Vars
	cdef int x
	cdef int y
	cdef int z

	cdef int x_size = all_ts.size
	cdef int y_size = len(charts)
	cdef int z_size = len(periods[0])

	# Body Vars
	# cdef np.ndarray[float, ndim=1] c_tick
	cdef int idx
	cdef int off
	cdef double timestamp
	cdef double trade_timestamp

	cdef str period
	cdef double[:,:] tick_values = tick_df.values

	chart = charts[0]
	cdef double tick_timestamp = all_ts[0]
	cdef double[:] ohlc_test = chart.getLastOHLC(periods[0][0])

	# For Each Timestamp
	for x in range(x_size):
		# For Each Chart
		for y in range(y_size):
			# For Each Period
			for z in range(z_size):
				timestamp = all_ts[x]
				chart = charts[y]

				# If lowest period, do position/order check
				if z == 0:
					c_tick = tick_values[x]
					trade_timestamp = timestamp + 60
					self.handleOrders(chart.product, trade_timestamp, c_tick)
					self.handleStopLoss(chart.product, trade_timestamp, c_tick)
					self.handleTakeProfit(chart.product, trade_timestamp, c_tick)

				off = offsets[y][z]
				if x >= off:
					tick_timestamp = timestamp
					period = periods[y][z]
					idx = chart._idx[period]
					bar_end = False

					period_data = chart._data[period]

					if (
						x+1 < all_ts.shape[0] and 
						idx+1 < period_data.shape[0] and 
						all_ts[x+1] >= period_data.index.values[idx+1]
					):
						bar_end = True
						tick_timestamp = period_data.index.values[idx]

					df = dataframes[y][z]

					period_data.values[idx] = df.values[x-off]

					chart.timestamps[period] = period_data.index.values[:idx+1][::-1]

					c_data = period_data.values[:idx+1]

					chart.asks[period] = c_data[:, :4][::-1]
					chart.mids[period] = c_data[:, 4:8][::-1]
					chart.bids[period] = c_data[:, 8:][::-1]
					# Add offset for real time because ONE MINUTE bars are being used for tick data
					chart._end_timestamp = timestamp + 60

					ohlc = chart.getLastOHLC(period)
					indicators = [ind for ind in chart.indicators.values() if ind.period == period]
					for i in range(len(indicators)):
						ind = indicators[i]
						ind_df = indicator_dataframes[y][z][i]

						result_size = int(ind_df.shape[1]/3)
						ind._asks[idx] = ind_df.values[x-off, :result_size]
						ind._mids[idx] = ind_df.values[x-off, result_size:result_size*2]
						ind._bids[idx] = ind_df.values[x-off, result_size*2:]
						ind._set_idx(idx)

					# Call threaded ontick functions
					# if period in chart._subscriptions:

					for func in chart._subscriptions[period]:
						tick = EventItem({
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

	print('Event Loop Complete {:.2f}s'.format(time.time() - start_time))
	sys.stdout.flush()


from . import utils, period as tl_period
from .broker import EventItem