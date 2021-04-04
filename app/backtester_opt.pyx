import time


def _event_loop(self, charts, all_ts, periods, dataframes, indicator_dataframes, tick_df, offsets):
	print('Start Event Loop.')
	start_time = time.time()

	# Iterator Vars
	cdef int x
	cdef int y
	cdef int z

	# Body Vars
	# cdef np.ndarray[float, ndim=1] c_tick
	cdef int idx
	cdef int off
	cdef float timestamp
	cdef float trade_timestamp

	# For Each Timestamp
	for x in range(all_ts.size):
		# For Each Chart
		for y in range(len(charts)):
			# For Each Period
			for z in range(len(periods[y])):
				timestamp = all_ts[x]
				chart = charts[y]

				# If lowest period, do position/order check
				# if z == 0:
				# 	c_tick = tick_df.values[x]
				# 	trade_timestamp = timestamp + 60
				# 	self.handleOrders(chart.product, trade_timestamp, c_tick)
				# 	self.handleStopLoss(chart.product, trade_timestamp, c_tick)
				# 	self.handleTakeProfit(chart.product, trade_timestamp, c_tick)

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

					period_data.iloc[idx] = df.iloc[x-off]

					chart.timestamps[period] = period_data.index.values[:idx+1][::-1]

					c_data = period_data.values[:idx+1]

					chart.asks[period] = c_data[:, :4][::-1]
					chart.mids[period] = c_data[:, 4:8][::-1]
					chart.bids[period] = c_data[:, 8:][::-1]
					# Add offset for real time because ONE MINUTE bars are being used for tick data
					chart._end_timestamp = timestamp + 60

					ohlc = chart.getLastOHLC(period)
					# indicators = [ind for ind in chart.indicators.values() if ind.period == period]
					# for i in range(len(indicators)):
					# 	ind = indicators[i]
					# 	ind_df = indicator_dataframes[y][z][i]

					# 	result_size = int(ind_df.shape[1]/3)
					# 	ind._asks[idx] = ind_df.values[x-off, :result_size]
					# 	ind._mids[idx] = ind_df.values[x-off, result_size:result_size*2]
					# 	ind._bids[idx] = ind_df.values[x-off, result_size*2:]
					# 	ind._set_idx(idx)

					# Call threaded ontick functions
					if period in chart._subscriptions:
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


from .broker import EventItem