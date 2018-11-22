import pandas as pd
import math
import matplotlib.pyplot as plt

class paths:
	def Paths(df,dates,column,Range_start, Range_end, Plot_date):

		"""
		Generate price paths from historical data.

		df: pandas DataFrame, dates: list with dates as datetime ([parser.parse(x) for x in dates] can be used to change str to datetime), column: column with data in df, Range_start: numbers of days before the date,  Range_end: numbers of days after the date, Plot_date: plot line with latest data, type date as str
		"""

		indexes = [] 

		# whole operation for latest date

		if Plot_date != '':
			plot_date = parser.parse(Plot_date)
			# plot_date = plot_date.date()

			if df[df['Date']==plot_date]['Date'].empty:
				print('ERROR! Plot_date is not in the df.')

			index_plot_date = []
			index_plot_datex = df[df['Date']==plot_date].index
			index_plot_date.extend(index_plot_datex)


			final_plot_date = pd.DataFrame()

			for x in range(Range_start,Range_end):
				try:
					final_plot_date.loc[x,0] = df.loc[index_plot_date[0]+x][column]
				except:
					continue

			for i in range(len(final_plot_date)):
				if i == abs(Range_start):
					continue
				try:
					final_plot_date.iloc[i,0] = (final_plot_date.iloc[i,0]/final_plot_date.iloc[abs(Range_start),0]-1)*100
				except:
					continue

			for i in range(0,1):
				try:
					final_plot_date.iloc[abs(Range_start),0] = 0      
				except:
					continue

		# compare dates  

		for i in dates:
			if df[df['Date']==i]['Date'].empty:
				print('ERROR! {} is not in the df.'.format(i))
			index = df[df['Date']==i].index
			indexes.extend(index)

		# generate slices of data for selected dates        

		final = pd.DataFrame()

		for i in range(0,len(indexes)):
			for x in range(Range_start,Range_end):
				final.loc[x,i] = df.loc[indexes[i]+x][column]

		# rebase

		for date in final:
			for i in range(len(final)):
				if i == abs(Range_start):
					continue
				final.iloc[i,date] = (final.iloc[i,date]/final.iloc[abs(Range_start),date]-1)*100
			final.iloc[abs(Range_start),date] = 0    

		# change column names to the dates

		final = final.rename(columns=(lambda x:dates[x].date()))

		# create df with min -> max values

		path = pd.DataFrame()

		for x in range(len(final.columns)):
			for i in range(len(final)):
				a = final.iloc[i,:]
				b = a.sort_values()[x]
				path.loc[i,x] = b
				a = None
				b = None 

		path.index = final.index

		# chart                

		fig, ax = plt.subplots(figsize=(15,12))

		ax.set_xlabel('Days')
		ax.set_ylabel('[%]')
		ax.set_xlim(Range_start,Range_end)

		if len(final.columns)%2 == 0: 

			for i in range(len(path.columns)):
				path[i].plot(ax=ax, color='Red', linewidth=0)



			for i in range(0,math.trunc(len(final.columns)/2)):
				ax.fill_between(path.index,path[i],path[2*math.trunc(len(final.columns)/2)-i-1],alpha=0.3,color='Pink')

		else:

			for i in [x for x in range(len(path.columns)) if x != math.trunc(len(path.columns)/2)]:
				path[i].plot(ax=ax, color='Red', linewidth=0)

			path[math.trunc(len(path.columns)/2)].plot(ax=ax, color='Red', linewidth=0.1)    # mid line


			for i in [x for x in range(len(path.columns)) if x != math.trunc(len(path.columns)/2)]:
				ax.fill_between(path.index,path[i],path[2*math.trunc(len(final.columns)/2)-i],alpha=0.3,color='Pink')

		if Plot_date != '':
			final_plot_date[0].plot(ax=ax, color='Black', linewidth=2)

		plt.grid(linestyle='-.',linewidth=0.3)  

		fig2, ax2 = plt.subplots(figsize=(15,12))

		ax2.set_xlabel('Days')
		ax2.set_ylabel('[%]')
		ax2.set_xlim(Range_start,Range_end)
		final.plot(ax=ax2)


		ax2.yaxis.grid(True, which='major')
		ax2.yaxis.grid(True, which='minor')

		if Plot_date != '':
				final_plot_date[0].plot(ax=ax2, color='Black', linewidth=3)

		plt.grid(linestyle='-.',linewidth=0.3)

class checker:

	def FindExtremes(df, column, first_rolling, second_rolling, final_format):

		dfx = df.copy()

		dfx.reset_index(inplace=True)
		dfx.rename(columns={'index':'Date'}, inplace=True)

		#self.dfx = dfx
		#self.column = column

		import matplotlib.dates as mdates

		dfx['mean'] = dfx[column].rolling(window=first_rolling).mean()
		dfx['mean_2'] = dfx[column].rolling(window=second_rolling).mean()

		if dfx.loc[first_rolling-1,'mean'] > dfx.loc[0,column]:
			mov_direction = 0
			mov_direction_fixed = 0 
		else:
			mov_direction = 1
			mov_direction_fixed = 1    

		dates = []

		for i in range(first_rolling-1,len(dfx)):
			if (mov_direction == 1) & (dfx.loc[i,'mean'] > dfx.loc[i,'mean_2']):
				dates.append(i-1)
				mov_direction = 0
			if (mov_direction == 0) & (dfx.loc[i,'mean'] < dfx.loc[i,'mean_2']):
				dates.append(i-1)
				mov_direction = 1  

		mov_h = []
		mov_l = []
		if mov_direction_fixed == 1:
			for i in dates[::2]:
				mov_h.append(i)
			mov_l.append(0)    
			for i in dates[1::2]:
				mov_l.append(i)    
		else:
			for i in dates[::2]:
				mov_l.append(i)
			mov_h.append(0) 
			for i in dates[1::2]:
				mov_h.append(i)  	

		highs = []
		lows = []

		if mov_direction_fixed == 1:
			for i in range(len(mov_h)):
				highs.append(dfx[mov_l[i]:mov_h[i]][column].idxmax())

			for i in range(len(mov_l)):
				try:
					lows.append(dfx[mov_h[i]:mov_l[i+1]][column].idxmin())
				except:
					continue		
		else:
			for i in range(len(mov_l)):
				highs.append(dfx[mov_h[i]:mov_l[i]][column].idxmin())

			for i in range(len(mov_h)):
				try:
					lows.append(dfx[mov_l[i]:mov_l[h+1]][column].idxmax())
				except:
					continue			


		if mov_direction_fixed == 0:
			del lows[0]
		else:
			del highs[0]

		extremes = []
		extremes = lows + highs		


		if final_format == 'df':
			low = pd.DataFrame()
			high = pd.DataFrame()
			
			for i in highs:
				high.loc[i,'High'] = dfx.loc[i,'Date']
			high.reset_index(inplace=True,drop=True)

			for i in lows:    
				low.loc[i,'Low'] = dfx.loc[i,'Date']
			low.reset_index(inplace=True,drop=True)   

			high.reset_index(inplace=True,drop=True)
			low.reset_index(inplace=True,drop=True)    

			final = pd.concat([high,low],axis=1)
			
		elif final_format == 'list':
			
			final = []
		
			for i in extremes:
				final.append(dfx.loc[i,'Date'])
				
		elif final_format == 'highs':
			
			final = []
		
			for i in highs:
				final.append(dfx.loc[i,'Date'])
				
		elif final_format == 'lows':
			
			final = []
		
			for i in lows:
				final.append(dfx.loc[i,'Date'])        

		dfx.set_index('Date',inplace=True)

		myFmt = mdates.DateFormatter('%Y-%m')
		years = mdates.YearLocator()   
		months = mdates.MonthLocator()

		ax = dfx[column].plot(figsize=(15,12), markevery=extremes,style='s-')

		ax.xaxis.set_major_locator(years)
		ax.xaxis.set_major_formatter(myFmt)
		ax.xaxis.set_minor_locator(months)

		ax2 = dfx[[column,'mean','mean_2']].plot(figsize=(15,12))

		ax2.xaxis.set_major_locator(years)
		ax2.xaxis.set_major_formatter(myFmt)
		ax2.xaxis.set_minor_locator(months)


		return(final)
	
class downloader:
	def stooq(Symbol, Interval, Part = False, Date_from = '2000-01-01', Date_to = '2100-01-01', Open=False, High=False, Low=False, Close=True, Volume=False):
		"""
		Download data from stooq.pl
		"""
		
		import datetime
		
		Date_f = Date_from.replace('-','')
		Date_t = Date_to.replace('-','')
		
		if Part == False:
			url = 'http://stooq.com/q/d/l/?s={}&i={}'.format(Symbol,Interval)
		else:
			url = 'http://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d'.format(Symbol,Date_f,Date_t)

		data = pd.read_csv(url)
		
		data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

		if Open == False:
			del data['Open']
		if High == False:
			del data['High']
		if Low == False:
			del data['Low']
		if Close == False:
			del data['Close']
		if (('Volume' in data.columns) & (Volume == False)) :
			del data['Volume']
			
		data.rename(columns={'Close':Symbol}, inplace=True)
		
		return data

class indicators:
	
	def OECD_growing_countries():
	
		'''
		Number of OECD countries with growing CLI (Leading indicator) in 12m period.
		'''
		
		
		# idea from https://10-procent-rocznie.blogspot.com/2016/06/wykres-ktory-wstrzasna-inwestorami.html
		
		from dateutil import parser 
		import requests as rq 
		import re
		
		# --- solution from https://stackoverflow.com/questions/40565871/read-data-from-oecd-api-into-python-and-pandas ---
		
		OECD_ROOT_URL = "http://stats.oecd.org/SDMX-JSON/data"
		
		def make_OECD_request(dsname, dimensions, params = None, root_dir = OECD_ROOT_URL):
			# Make URL for the OECD API and return a response
			# 4 dimensions: location, subject, measure, frequency
			# OECD API: https://data.oecd.org/api/sdmx-json-documentation/#d.en.330346

			if not params:
				params = {}

			dim_args = ['+'.join(d) for d in dimensions]
			dim_str = '.'.join(dim_args)

			url = root_dir + '/' + dsname + '/' + dim_str + '/all'

			# print('Requesting URL ' + url)
			return rq.get(url = url, params = params)
			
		def create_DataFrame_from_OECD(country = 'CZE', subject = [], measure = [], frequency = 'M',  startDate = None, endDate = None):     
			# Request data from OECD API and return pandas DataFrame

			# country: country code (max 1)
			# subject: list of subjects, empty list for all
			# measure: list of measures, empty list for all
			# frequency: 'M' for monthly and 'Q' for quarterly time series
			# startDate: date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
			# endDate: date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations

			# Data download

			response = make_OECD_request('MEI'
										 , [[country], subject, measure, [frequency]]
										 , {'startTime': startDate, 'endTime': endDate, 'dimensionAtObservation': 'AllDimensions'})

			# Data transformation

			if (response.status_code == 200):

				responseJson = response.json()

				obsList = responseJson.get('dataSets')[0].get('observations')

				if (len(obsList) > 0):

					# print('Data downloaded from %s' % response.url)

					timeList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'TIME_PERIOD'][0]['values']
					subjectList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'SUBJECT'][0]['values']
					measureList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'MEASURE'][0]['values']

					obs = pd.DataFrame(obsList).transpose()
					obs.rename(columns = {0: 'series'}, inplace = True)
					obs['id'] = obs.index
					obs = obs[['id', 'series']]
					obs['dimensions'] = obs.apply(lambda x: re.findall('\d+', x['id']), axis = 1)
					obs['subject'] = obs.apply(lambda x: subjectList[int(x['dimensions'][1])]['id'], axis = 1)
					obs['measure'] = obs.apply(lambda x: measureList[int(x['dimensions'][2])]['id'], axis = 1)
					obs['time'] = obs.apply(lambda x: timeList[int(x['dimensions'][4])]['id'], axis = 1)
					obs['names'] = obs['subject'] + '_' + obs['measure']

					data = obs.pivot_table(index = 'time', columns = ['names'], values = 'series')

					return(data)

				else:

					print('Error: No available records, please change parameters')

			else:

				print('Error: %s' % response.status_code)
				
		# --- end of the solution ---		
		
		codes = ['AUS', 'AUT', 'BEL', 'BRA', 'CHL', 'CHN', 'CZE', 'DNK', 'EST',
		'FIN', 'FRA', 'GRC', 'ESP', 'NLD', 'IND', 'IDN', 'IRL', 'ISR',
		'JPN', 'CAN', 'KOR', 'MEX', 'DEU', 'NOR', 'NZL', 'POL', 'PRT',
		'RUS', 'ZAF', 'SVK', 'SVN', 'CHE', 'SWE', 'TUR', 'USA', 'HUN',
		'GBR', 'ITA']

		countries = ['Australia', 'Austria', 'Belgia', 'Brazylia', 'Chile', 'Chiny',
		'Czechy', 'Dania', 'Estonia', 'Finlandia', 'Francja', 'Grecja',
		'Hiszpania', 'Holandia', 'Indie', 'Indonezja', 'Irlandia',
		'Izrael', 'Japonia', 'Kanada', 'Korea Południowa', 'Meksyk',
		'Niemcy', 'Norwegia', 'Nowa Zelandia', 'Polska', 'Portugalia',
		'Rosja', 'RPA', 'Słowacja', 'Słowenia', 'Szwajcaria', 'Szwecja',
		'Turcja', 'USA', 'Węgry', 'Wielka Brytania', 'Włochy']
		
		df = pd.DataFrame()
		df = create_DataFrame_from_OECD(country = codes[0], subject = ['LOLITOAA'])
		df.columns = [countries[0]]

		for i in range(1,len(codes)):
			df1 = create_DataFrame_from_OECD(country = codes[i], subject = ['LOLITOAA'])
			df1.columns = [countries[i]]
			if i == 1:
				final = pd.concat([df,df1],axis=1, sort=True)
			else:
				final = pd.concat([final,df1],axis=1, sort=True)
			   
		pct = final.loc['1994-01':].pct_change(periods=12)
		pct = pct.loc['1995-01':]
		
		growing_countries = pd.DataFrame()
		growing_countries['Growing countries'] = pct.select_dtypes(include='float64').gt(0).sum(axis=1)
		growing_countries.reset_index(inplace=True)
		growing_countries['index'] = [parser.parse(x) for x in growing_countries['index']]
		growing_countries.set_index('index', drop=True,inplace=True)
		
		return(growing_countries)
		