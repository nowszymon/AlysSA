import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil import parser 

class paths:
	
	def MargaRET(df, Column, Cut_off_date):

		"""
		MargaRET predicts future values based on correlation of actual quotes with historical ones.
		
		df: pandas DataFrame - 'Date' as separate column, Column: Column with data, Cut_off_date: cut-off date.
		"""

		cut_off_index = df.index[df['Date'] == Cut_off_date][0]
		
		cut_data = pd.DataFrame()
		for i in range(cut_off_index,len(df)):
			cut_data.loc[i,Column] = df.loc[i,Column]
			
		cut_data.reset_index(inplace=True,drop=True)	
		df = df[:cut_off_index]
		
		slices = pd.DataFrame()
		for i in range(0,(len(df)-len(cut_data))):
			for x in range(0,len(cut_data)):
				slices.loc[x,i] = df.loc[i+x, Column]
				
		correlation_table = pd.DataFrame()		
		for i in slices:
			correlation_table.loc[i,'Correlation'] = cut_data[Column].corr(slices[i])
			
		max_corr = correlation_table['Correlation'].idxmax()

		final = pd.DataFrame()
		final['Projection'] = df.loc[max_corr:(max_corr+2*len(cut_data)),Column]
		final.reset_index(inplace=True,drop=True)
		final[Column] = cut_data
		
		dates = df.loc[max_corr:(max_corr+2*len(cut_data)),'Date']
		final.set_index(dates,inplace=True,drop=True)
		
		fig, ax = plt.subplots(figsize=(15,10))
		ax.plot(final['Projection'],color='red',linestyle='--', alpha=0.5)
		ax2 = ax.twinx()
		ax2.plot(final[Column])

		ax.set_title('{} projection (r = {}%)'.format(Column,"%.2f" % (correlation_table['Correlation'].max()*100)),fontsize=15)
		ax.set_ylabel('Projection')
		ax2.set_ylabel(Column)
		
		ax.legend(loc=2)
		ax2.legend()
		
		# anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
		# ax.add_artist(anchored_text)
	
	def Paths(df,dates,column,Range_start, Range_end, Plot_date, fix_dates, x_label='Days'):

		"""
		Generate price paths from historical data.

		df: pandas DataFrame, dates: list with dates as datetime ([parser.parse(x) for x in dates] can be used to change str to datetime), column: column with data in df, Range_start: numbers of days before the date,  Range_end: numbers of days after the date, Plot_date: plot line with latest data, type date as str, fix_dates: for non existing dates in df (i.e. weekends) fix add days until it finds session day
		"""

		indexes = [] 
		dates_c = dates.copy()

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
		
		non_dates = []
		
		for i in dates_c:
			if df[df['Date']==i]['Date'].empty:
				if fix_dates == True:
					while df[df['Date']==i]['Date'].empty:
						i = i + timedelta(days=1)
				else:
					print('ERROR! {} is not in the df.'.format(i))
					non_dates.append(i)
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

		dates_for_columns = [x for x in dates_c if x not in non_dates]    
		final = final.rename(columns=(lambda x:dates_for_columns[x].date()))
		
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
		
		# mean
		
		mean = pd.DataFrame()
		
		mean['mean'] = final.mean(axis=1)

		# chart                

		fig, ax = plt.subplots(figsize=(15,12))

		ax.set_xlabel(x_label)
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

		ax2.set_xlabel(x_label)
		ax2.set_ylabel('[%]')
		ax2.set_xlim(Range_start,Range_end)
		final.plot(ax=ax2)
		plt.legend(loc='upper left')

		ax2.yaxis.grid(True, which='major')
		ax2.yaxis.grid(True, which='minor')

		if Plot_date != '':
				final_plot_date[0].plot(ax=ax2, color='Black', linewidth=3)

		plt.grid(linestyle='-.',linewidth=0.3)

		fig3, ax3 = plt.subplots(figsize=(15,12))
		
		ax3.set_xlabel(x_label)
		ax3.set_ylabel('[%]')
		ax3.set_xlim(Range_start,Range_end)
		mean['mean'].plot(ax=ax3)
		
		ax2.yaxis.grid(True, which='major')
		ax2.yaxis.grid(True, which='minor')
		
		if Plot_date != '':
			final_plot_date[0].plot(ax=ax3, color='Black', linewidth=3)
		
		plt.grid(linestyle='-.',linewidth=0.3)		
		
class checker:

	def FindExtremes(df, column, first_rolling, second_rolling, final_format):
	
		"""
		Find extremes of time series. 
		
		df: pandas DataFrame with dates as index, column: column with data, first_rolling: first MA to find extremes (higher than second_rolling), second rolling, final_format: df, highs or lows (list format)
		"""

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
		
	def crossing(df,column,level,direction,stop_level):
	
		"""
		Find dates for points when time series cross selected level. Stop_level can be used to choose only valid points.

		df: pandas DataFrame, column: column with data, level: level for crossing, direction: up or down, stop_level: level that must be crossed after giving a signal to show another one
		"""
	
		df = df.copy()
		
		indexes = []
		stop = False
		
		if direction == 'down':
			for i in range(1,len(df)):
				if stop == True:
					if ((df.loc[i,column]>=stop_level) & (df.loc[i,column] > df.loc[i-1,column])):
						stop = False
						continue
				if ((df.loc[i,column]<=level) & (df.loc[i,column] < df.loc[i-1,column]) & (stop == False)):
					indexes.append(i)
					stop = True
					
		elif direction == 'up':
			for i in range(1,len(df)):
				if stop == True:
					if ((df.loc[i,column]<=stop_level) & (df.loc[i,column] > df.loc[i-1,column])):
						stop = False
						continue
				if ((df.loc[i,column]>=level) & (df.loc[i,column] > df.loc[i-1,column]) & (stop == False)):
					indexes.append(i)
					stop = True
					
		dates = []
		
		for i in indexes:
			dates.append(df.loc[i,'Date'])       
			
			
		df.set_index('Date',inplace=True)

		ax = df[column].plot(x='Date',y=column,figsize=(15,12), markevery=indexes,style='s-')
		
		for dat in dates:
			plt.axvline(x=dat, color='k', linestyle='--')       
				
		return(dates)	
	
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
	
	def IndexRollingMA(df,column,rolling,avg_2):
		
		df = df.copy()
		df['avg'] = df[column].rolling(window=rolling).mean()
		df.dropna(inplace=True)
		df.reset_index(inplace=True,drop=True)
		df['avg_2'] = df['avg'].pct_change(periods=avg_2)
		df.dropna(inplace=True)
		df.reset_index(inplace=True,drop=True)
		
		return(df)
	
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
		
	def WIGIndicators(index='WIG'):

		import numpy as np
		import datetime

		if index == 'WIG':

			wig = downloader.stooq('WIG','d')
			wigPE = downloader.stooq('WIG_PE','d')
			wigPB = downloader.stooq('WIG_PB','d')
			wigDY = downloader.stooq('WIG_DY','d')
			wigMV = downloader.stooq('WIG_MV','d')

			wig.set_index('Date',inplace=True,drop=True)
			wigPE.set_index('Date',inplace=True,drop=True)
			wigPB.set_index('Date',inplace=True,drop=True)
			wigDY.set_index('Date',inplace=True,drop=True)
			wigMV.set_index('Date',inplace=True,drop=True)

			wig_mean = [np.mean(wig)]*len(wig)
			wigPE_mean = [np.mean(wigPE)]*len(wigPE)
			wigPB_mean = [np.mean(wigPB)]*len(wigPB)
			wigDY_mean = [np.mean(wigDY)]*len(wigDY)
			wigMV_mean = [np.mean(wigMV)]*len(wigMV)

			fig = plt.figure(figsize=(15,15))

			ax1 = plt.subplot2grid((10,2), (0,0),rowspan=4,colspan=2)
			ax2 = plt.subplot2grid((10,2), (4,0),rowspan=3,colspan=1)
			ax3 = plt.subplot2grid((10,2), (4,1),rowspan=3,colspan=1)
			ax4 = plt.subplot2grid((10,2), (7,0),rowspan=3,colspan=1)
			ax5 = plt.subplot2grid((10,2), (7,1),rowspan=3,colspan=1)

			ax1.set_title('WIG',fontsize=15)
			ax2.set_title('P/E')
			ax3.set_title('P/BV')
			ax4.set_title('Dividend Yield')
			ax5.set_title('Market Value')

			ax1.plot(wig)
			ax2.plot(wigPE)
			ax3.plot(wigPB)
			ax4.plot(wigDY)
			ax5.plot(wigMV)

			ax1.plot(wig.index.values,wig_mean, linestyle='--', label='average')
			ax2.plot(wigPE.index.values,wigPE_mean, linestyle='--', label='average')
			ax3.plot(wigPB.index.values,wigPB_mean, linestyle='--', label='average')
			ax4.plot(wigDY.index.values,wigDY_mean, linestyle='--', label='average')
			ax5.plot(wigMV.index.values,wigMV_mean, linestyle='--', label='average')

			bbox_props = dict(boxstyle='larrow')
			bbox_props_2 = dict(boxstyle='larrow', color='orange')
			ax1.annotate(str(wig['WIG'][-1]), (wig.index[-1], wig['WIG'][-1]), xytext = (wig.index[-1]+ datetime.timedelta(weeks=90), wig['WIG'][-1]),bbox=bbox_props,color='white')
			ax2.annotate(str(round(wigPE['WIG_PE'][-1],2)), (wigPE.index[-1], wigPE['WIG_PE'][-1]), xytext = (wigPE.index[-1]+ datetime.timedelta(weeks=35), wigPE['WIG_PE'][-1]),bbox=bbox_props,color='white')
			ax3.annotate(str(round(wigPB['WIG_PB'][-1],2)), (wigPB.index[-1], wigPB['WIG_PB'][-1]), xytext = (wigPB.index[-1]+ datetime.timedelta(weeks=43), wigPB['WIG_PB'][-1]),bbox=bbox_props,color='white')
			ax4.annotate(str(round(wigDY['WIG_DY'][-1],2)), (wigDY.index[-1], wigDY['WIG_DY'][-1]), xytext = (wigDY.index[-1]+ datetime.timedelta(weeks=35), wigDY['WIG_DY'][-1]),bbox=bbox_props,color='white')
			ax5.annotate(str(round(wigMV['WIG_MV'][-1],2)), (wigMV.index[-1], wigMV['WIG_MV'][-1]), xytext = (wigMV.index[-1]+ datetime.timedelta(weeks=43), wigMV['WIG_MV'][-1]),bbox=bbox_props,color='white')

			ax1.legend()
			ax2.legend()
			ax3.legend()
			ax4.legend()
			ax5.legend()

			plt.tight_layout()
			plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
			plt.show()
			
		elif index == 'mWIG40':

			mwig40 = downloader.stooq('MWIG40','d')
			mwig40TR = downloader.stooq('MWIG40TR','d')
			mwig40PE = downloader.stooq('MWIG40_PE','d')
			mwig40PB = downloader.stooq('MWIG40_PB','d')
			mwig40DY = downloader.stooq('MWIG40_DY','d')
			mwig40MV = downloader.stooq('MWIG40_MV','d')

			mwig40.set_index('Date',inplace=True,drop=True)
			mwig40TR.set_index('Date',inplace=True,drop=True)
			mwig40PE.set_index('Date',inplace=True,drop=True)
			mwig40PB.set_index('Date',inplace=True,drop=True)
			mwig40DY.set_index('Date',inplace=True,drop=True)
			mwig40MV.set_index('Date',inplace=True,drop=True)

			mwig40_mean = [np.mean(mwig40)]*len(mwig40)
			mwig40TR_mean = [np.mean(mwig40TR)]*len(mwig40TR)
			mwig40PE_mean = [np.mean(mwig40PE)]*len(mwig40PE)
			mwig40PB_mean = [np.mean(mwig40PB)]*len(mwig40PB)
			mwig40DY_mean = [np.mean(mwig40DY)]*len(mwig40DY)
			mwig40MV_mean = [np.mean(mwig40MV)]*len(mwig40MV)

			fig = plt.figure(figsize=(15,15))

			ax1 = plt.subplot2grid((10,2), (0,0),rowspan=4,colspan=2)
			ax2 = plt.subplot2grid((10,2), (4,0),rowspan=3,colspan=1)
			ax3 = plt.subplot2grid((10,2), (4,1),rowspan=3,colspan=1)
			ax4 = plt.subplot2grid((10,2), (7,0),rowspan=3,colspan=1)
			ax5 = plt.subplot2grid((10,2), (7,1),rowspan=3,colspan=1)

			ax1.set_title('mWIG40',fontsize=15)
			ax2.set_title('P/E')
			ax3.set_title('P/BV')
			ax4.set_title('Dividend Yield')
			ax5.set_title('Market Value')

			ax1.plot(mwig40,label='mWIG40')
			ax1.plot(mwig40TR, label='mWIG40 TR')
			ax2.plot(mwig40PE)
			ax4.plot(mwig40DY)
			ax3.plot(mwig40PB)
			ax5.plot(mwig40MV)

			ax1.plot(mwig40.index.values,mwig40_mean, linestyle='--', label='average')
			ax1.plot(mwig40TR.index.values,mwig40TR_mean, linestyle='--', label='TR average')
			ax2.plot(mwig40PE.index.values,mwig40PE_mean, linestyle='--', label='average')
			ax4.plot(mwig40DY.index.values,mwig40DY_mean, linestyle='--', label='average')
			ax3.plot(mwig40PB.index.values,mwig40PB_mean, linestyle='--', label='average')
			ax5.plot(mwig40MV.index.values,mwig40MV_mean, linestyle='--', label='average')

			bbox_props = dict(boxstyle='larrow')
			bbox_props_2 = dict(boxstyle='larrow', color='orange')
			ax1.annotate(str(round(mwig40['MWIG40'][-1],2)), (mwig40.index[-1], mwig40['MWIG40'][-1]), xytext = (mwig40.index[-1]+ datetime.timedelta(weeks=65), mwig40['MWIG40'][-1]),bbox=bbox_props,color='white')
			ax1.annotate(str(round(mwig40TR['MWIG40TR'][-1],2)), (mwig40TR.index[-1], mwig40TR['MWIG40TR'][-1]), xytext = (mwig40TR.index[-1]+ datetime.timedelta(weeks=65), mwig40TR['MWIG40TR'][-1]),bbox=bbox_props_2,color='black')
			ax2.annotate(str(round(mwig40PE['MWIG40_PE'][-1],2)), (mwig40PE.index[-1], mwig40PE['MWIG40_PE'][-1]), xytext = (mwig40PE.index[-1]+ datetime.timedelta(weeks=35), mwig40PE['MWIG40_PE'][-1]),bbox=bbox_props,color='white')
			ax4.annotate(str(round(mwig40DY['MWIG40_DY'][-1],2)), (mwig40DY.index[-1], mwig40DY['MWIG40_DY'][-1]), xytext = (mwig40DY.index[-1]+ datetime.timedelta(weeks=43), mwig40DY['MWIG40_DY'][-1]),bbox=bbox_props,color='white')
			ax3.annotate(str(round(mwig40PB['MWIG40_PB'][-1],2)), (mwig40PB.index[-1], mwig40PB['MWIG40_PB'][-1]), xytext = (mwig40PB.index[-1]+ datetime.timedelta(weeks=35), mwig40PB['MWIG40_PB'][-1]),bbox=bbox_props,color='white')
			ax5.annotate(str(round(mwig40MV['MWIG40_MV'][-1],2)), (mwig40MV.index[-1], mwig40MV['MWIG40_MV'][-1]), xytext = (mwig40MV.index[-1]+ datetime.timedelta(weeks=43), mwig40MV['MWIG40_MV'][-1]),bbox=bbox_props,color='white')

			ax1.legend()
			ax2.legend()
			ax3.legend()
			ax4.legend()
			ax5.legend()

			plt.tight_layout()
			plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
			plt.show()

		elif index == 'sWIG80':

			mwig40 = downloader.stooq('SWIG80','d')
			mwig40TR = downloader.stooq('SWIG80TR','d')
			mwig40PE = downloader.stooq('SWIG80_PE','d')
			mwig40PB = downloader.stooq('SWIG80_PB','d')
			mwig40DY = downloader.stooq('SWIG80_DY','d')
			mwig40MV = downloader.stooq('SWIG80_MV','d')

			mwig40.set_index('Date',inplace=True,drop=True)
			mwig40TR.set_index('Date',inplace=True,drop=True)
			mwig40PE.set_index('Date',inplace=True,drop=True)
			mwig40PB.set_index('Date',inplace=True,drop=True)
			mwig40DY.set_index('Date',inplace=True,drop=True)
			mwig40MV.set_index('Date',inplace=True,drop=True)

			mwig40_mean = [np.mean(mwig40)]*len(mwig40)
			mwig40TR_mean = [np.mean(mwig40TR)]*len(mwig40TR)
			mwig40PE_mean = [np.mean(mwig40PE)]*len(mwig40PE)
			mwig40PB_mean = [np.mean(mwig40PB)]*len(mwig40PB)
			mwig40DY_mean = [np.mean(mwig40DY)]*len(mwig40DY)
			mwig40MV_mean = [np.mean(mwig40MV)]*len(mwig40MV)

			fig = plt.figure(figsize=(15,15))

			ax1 = plt.subplot2grid((10,2), (0,0),rowspan=4,colspan=2)
			ax2 = plt.subplot2grid((10,2), (4,0),rowspan=3,colspan=1)
			ax3 = plt.subplot2grid((10,2), (4,1),rowspan=3,colspan=1)
			ax4 = plt.subplot2grid((10,2), (7,0),rowspan=3,colspan=1)
			ax5 = plt.subplot2grid((10,2), (7,1),rowspan=3,colspan=1)

			ax1.set_title('sWIG40',fontsize=15)
			ax2.set_title('P/E')
			ax3.set_title('P/BV')
			ax4.set_title('Dividend Yield')
			ax5.set_title('Market Value')

			ax1.plot(mwig40,label='sWIG80')
			ax1.plot(mwig40TR, label='sWIG80 TR')
			ax2.plot(mwig40PE)
			ax4.plot(mwig40DY)
			ax3.plot(mwig40PB)
			ax5.plot(mwig40MV)

			ax1.plot(mwig40.index.values,mwig40_mean, linestyle='--', label='average')
			ax1.plot(mwig40TR.index.values,mwig40TR_mean, linestyle='--', label='TR average')
			ax2.plot(mwig40PE.index.values,mwig40PE_mean, linestyle='--', label='average')
			ax4.plot(mwig40DY.index.values,mwig40DY_mean, linestyle='--', label='average')
			ax3.plot(mwig40PB.index.values,mwig40PB_mean, linestyle='--', label='average')
			ax5.plot(mwig40MV.index.values,mwig40MV_mean, linestyle='--', label='average')

			bbox_props = dict(boxstyle='larrow')
			bbox_props_2 = dict(boxstyle='larrow', color='orange')
			ax1.annotate(str(round(mwig40['SWIG80'][-1],2)), (mwig40.index[-1], mwig40['SWIG80'][-1]), xytext = (mwig40.index[-1]+ datetime.timedelta(weeks=65), mwig40['SWIG80'][-1]),bbox=bbox_props,color='white')
			ax1.annotate(str(round(mwig40TR['SWIG80TR'][-1],2)), (mwig40TR.index[-1], mwig40TR['SWIG80TR'][-1]), xytext = (mwig40TR.index[-1]+ datetime.timedelta(weeks=65), mwig40TR['SWIG80TR'][-1]),bbox=bbox_props_2,color='black')
			ax2.annotate(str(round(mwig40PE['SWIG80_PE'][-1],2)), (mwig40PE.index[-1], mwig40PE['SWIG80_PE'][-1]), xytext = (mwig40PE.index[-1]+ datetime.timedelta(weeks=35), mwig40PE['SWIG80_PE'][-1]),bbox=bbox_props,color='white')
			ax4.annotate(str(round(mwig40DY['SWIG80_DY'][-1],2)), (mwig40DY.index[-1], mwig40DY['SWIG80_DY'][-1]), xytext = (mwig40DY.index[-1]+ datetime.timedelta(weeks=43), mwig40DY['SWIG80_DY'][-1]),bbox=bbox_props,color='white')
			ax3.annotate(str(round(mwig40PB['SWIG80_PB'][-1],2)), (mwig40PB.index[-1], mwig40PB['SWIG80_PB'][-1]), xytext = (mwig40PB.index[-1]+ datetime.timedelta(weeks=35), mwig40PB['SWIG80_PB'][-1]),bbox=bbox_props,color='white')
			ax5.annotate(str(round(mwig40MV['SWIG80_MV'][-1],2)), (mwig40MV.index[-1], mwig40MV['SWIG80_MV'][-1]), xytext = (mwig40MV.index[-1]+ datetime.timedelta(weeks=43), mwig40MV['SWIG80_MV'][-1]),bbox=bbox_props,color='white')

			ax1.legend()
			ax2.legend()
			ax3.legend()
			ax4.legend()
			ax5.legend()

			plt.tight_layout()
			plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
			plt.show()

		elif index == 'WIG20':

			mwig40 = downloader.stooq('WIG20','d')
			mwig40TR = downloader.stooq('WIG20TR','d')
			mwig40PE = downloader.stooq('WIG20_PE','d')
			mwig40PB = downloader.stooq('WIG20_PB','d')
			mwig40DY = downloader.stooq('WIG20_DY','d')
			mwig40MV = downloader.stooq('WIG20_MV','d')

			mwig40.set_index('Date',inplace=True,drop=True)
			mwig40TR.set_index('Date',inplace=True,drop=True)
			mwig40PE.set_index('Date',inplace=True,drop=True)
			mwig40PB.set_index('Date',inplace=True,drop=True)
			mwig40DY.set_index('Date',inplace=True,drop=True)
			mwig40MV.set_index('Date',inplace=True,drop=True)

			mwig40_mean = [np.mean(mwig40)]*len(mwig40)
			mwig40TR_mean = [np.mean(mwig40TR)]*len(mwig40TR)
			mwig40PE_mean = [np.mean(mwig40PE)]*len(mwig40PE)
			mwig40PB_mean = [np.mean(mwig40PB)]*len(mwig40PB)
			mwig40DY_mean = [np.mean(mwig40DY)]*len(mwig40DY)
			mwig40MV_mean = [np.mean(mwig40MV)]*len(mwig40MV)

			fig = plt.figure(figsize=(15,15))

			ax1 = plt.subplot2grid((10,2), (0,0),rowspan=4,colspan=2)
			ax2 = plt.subplot2grid((10,2), (4,0),rowspan=3,colspan=1)
			ax3 = plt.subplot2grid((10,2), (4,1),rowspan=3,colspan=1)
			ax4 = plt.subplot2grid((10,2), (7,0),rowspan=3,colspan=1)
			ax5 = plt.subplot2grid((10,2), (7,1),rowspan=3,colspan=1)

			ax1.set_title('WIG20',fontsize=15)
			ax2.set_title('P/E')
			ax3.set_title('P/BV')
			ax4.set_title('Dividend Yield')
			ax5.set_title('Market Value')

			ax1.plot(mwig40,label='WIG20')
			ax1.plot(mwig40TR, label='WIG20 TR')
			ax2.plot(mwig40PE)
			ax3.plot(mwig40DY)
			ax4.plot(mwig40PB)
			ax5.plot(mwig40MV)

			ax1.plot(mwig40.index.values,mwig40_mean, linestyle='--', label='average')
			ax1.plot(mwig40TR.index.values,mwig40TR_mean, linestyle='--', label='TR average')
			ax2.plot(mwig40PE.index.values,mwig40PE_mean, linestyle='--', label='average')
			ax4.plot(mwig40DY.index.values,mwig40DY_mean, linestyle='--', label='average')
			ax3.plot(mwig40PB.index.values,mwig40PB_mean, linestyle='--', label='average')
			ax5.plot(mwig40MV.index.values,mwig40MV_mean, linestyle='--', label='average')

			bbox_props = dict(boxstyle='larrow')
			bbox_props_2 = dict(boxstyle='larrow', color='orange')
			ax1.annotate(str(round(mwig40['WIG20'][-1],2)), (mwig40.index[-1], mwig40['WIG20'][-1]), xytext = (mwig40.index[-1]+ datetime.timedelta(weeks=65), mwig40['WIG20'][-1]),bbox=bbox_props,color='white')
			ax1.annotate(str(round(mwig40TR['WIG20TR'][-1],2)), (mwig40TR.index[-1], mwig40TR['WIG20TR'][-1]), xytext = (mwig40TR.index[-1]+ datetime.timedelta(weeks=65), mwig40TR['WIG20TR'][-1]),bbox=bbox_props_2,color='black')
			ax2.annotate(str(round(mwig40PE['WIG20_PE'][-1],2)), (mwig40PE.index[-1], mwig40PE['WIG20_PE'][-1]), xytext = (mwig40PE.index[-1]+ datetime.timedelta(weeks=35), mwig40PE['WIG20_PE'][-1]),bbox=bbox_props,color='white')
			ax3.annotate(str(round(mwig40DY['WIG20_DY'][-1],2)), (mwig40DY.index[-1], mwig40DY['WIG20_DY'][-1]), xytext = (mwig40DY.index[-1]+ datetime.timedelta(weeks=43), mwig40DY['WIG20_DY'][-1]),bbox=bbox_props,color='white')
			ax3.annotate(str(round(mwig40PB['WIG20_PB'][-1],2)), (mwig40PB.index[-1], mwig40PB['WIG20_PB'][-1]), xytext = (mwig40PB.index[-1]+ datetime.timedelta(weeks=35), mwig40PB['WIG20_PB'][-1]),bbox=bbox_props,color='white')
			ax5.annotate(str(round(mwig40MV['WIG20_MV'][-1],2)), (mwig40MV.index[-1], mwig40MV['WIG20_MV'][-1]), xytext = (mwig40MV.index[-1]+ datetime.timedelta(weeks=43), mwig40MV['WIG20_MV'][-1]),bbox=bbox_props,color='white')

			ax1.legend()
			ax2.legend()
			ax3.legend()
			ax4.legend()
			ax5.legend()

			plt.tight_layout()
			plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
			plt.show()
			
		elif index == 'NC':
		
			wig = downloader.stooq('NCINDEX','d')
			wigPE = downloader.stooq('NCINDEX_PE','d')
			wigPB = downloader.stooq('NCINDEX_PB','d')
			wigDY = downloader.stooq('NCINDEX_DY','d')
			wigMV = downloader.stooq('NCINDEX_MV','d')

			wig.set_index('Date',inplace=True,drop=True)
			wigPE.set_index('Date',inplace=True,drop=True)
			wigPB.set_index('Date',inplace=True,drop=True)
			wigDY.set_index('Date',inplace=True,drop=True)
			wigMV.set_index('Date',inplace=True,drop=True)

			wig_mean = [np.mean(wig)]*len(wig)
			wigPE_mean = [np.mean(wigPE)]*len(wigPE)
			wigPB_mean = [np.mean(wigPB)]*len(wigPB)
			wigDY_mean = [np.mean(wigDY)]*len(wigDY)
			wigMV_mean = [np.mean(wigMV)]*len(wigMV)

			fig = plt.figure(figsize=(15,15))

			ax1 = plt.subplot2grid((10,2), (0,0),rowspan=4,colspan=2)
			ax2 = plt.subplot2grid((10,2), (4,0),rowspan=3,colspan=1)
			ax3 = plt.subplot2grid((10,2), (4,1),rowspan=3,colspan=1)
			ax4 = plt.subplot2grid((10,2), (7,0),rowspan=3,colspan=1)
			ax5 = plt.subplot2grid((10,2), (7,1),rowspan=3,colspan=1)

			ax1.set_title('NC Index',fontsize=15)
			ax2.set_title('P/E')
			ax3.set_title('P/BV')
			ax4.set_title('Dividend Yield')
			ax5.set_title('Market Value')

			ax1.plot(wig)
			ax2.plot(wigPE)
			ax3.plot(wigPB)
			ax4.plot(wigDY)
			ax5.plot(wigMV)

			ax1.plot(wig.index.values,wig_mean, linestyle='--', label='average')
			ax2.plot(wigPE.index.values,wigPE_mean, linestyle='--', label='average')
			ax3.plot(wigPB.index.values,wigPB_mean, linestyle='--', label='average')
			ax4.plot(wigDY.index.values,wigDY_mean, linestyle='--', label='average')
			ax5.plot(wigMV.index.values,wigMV_mean, linestyle='--', label='average')

			bbox_props = dict(boxstyle='larrow')
			bbox_props_2 = dict(boxstyle='larrow', color='orange')
			ax1.annotate(str(round(wig['NCINDEX'][-1],2)), (wig.index[-1], wig['NCINDEX'][-1]), xytext = (wig.index[-1]+ datetime.timedelta(weeks=35), wig['NCINDEX'][-1]),bbox=bbox_props,color='white')
			ax2.annotate(str(round(wigPE['NCINDEX_PE'][-1],2)), (wigPE.index[-1], wigPE['NCINDEX_PE'][-1]), xytext = (wigPE.index[-1]+ datetime.timedelta(weeks=35), wigPE['NCINDEX_PE'][-1]),bbox=bbox_props,color='white')
			ax3.annotate(str(round(wigPB['NCINDEX_PB'][-1],2)), (wigPB.index[-1], wigPB['NCINDEX_PB'][-1]), xytext = (wigPB.index[-1]+ datetime.timedelta(weeks=43), wigPB['NCINDEX_PB'][-1]),bbox=bbox_props,color='white')
			ax4.annotate(str(round(wigDY['NCINDEX_DY'][-1],2)), (wigDY.index[-1], wigDY['NCINDEX_DY'][-1]), xytext = (wigDY.index[-1]+ datetime.timedelta(weeks=35), wigDY['NCINDEX_DY'][-1]),bbox=bbox_props,color='white')
			ax5.annotate(str(round(wigMV['NCINDEX_MV'][-1],2)), (wigMV.index[-1], wigMV['NCINDEX_MV'][-1]), xytext = (wigMV.index[-1]+ datetime.timedelta(weeks=43), wigMV['NCINDEX_MV'][-1]),bbox=bbox_props,color='white')

			ax1.legend()
			ax2.legend()
			ax3.legend()
			ax4.legend()
			ax5.legend()

			plt.tight_layout()
			plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
			plt.show()			
			