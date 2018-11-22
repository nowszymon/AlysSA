import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

"""
AlysSA v0.15b

NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1): Calculate the day return on the day after daily return of PreviousDay value occured.
BarsInaRow(Data, DataColumn): How many times there was a X number of down/up candles in a row.
SeasonalityPattern(DataFrame, Column): Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.
MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'): Calculate number of months when symbol exceeded Level move in Window days.
DownloadFromStooq(Symbol, Interval, Open=False, High=False, Low=False, Close=True, Volume=False): Download data from stooq.plot
SeasonalityDaily(DataFrame, Column): Calculate mean of daily returns and percentage of days with plus/minus daily returns.
Rebase(DataFrame): Calucalte percent changes to the first value in a column.
Fed_rates(): Load pickled df with details of FED funds rates 
Plot_X_values_on_Y_dates(df, Column, Dates, fred_api, Range_start=-20, Range_end=60,Title='',PlotAll=False): Plot min and max boundaries for the average price action on selected days.	
Bubbles(): Load pickled df with price bubbles data
HeatmapMonthly(df, DateColumn, CloseColumn, Day=False): Heatmap of daily/monthly returns.
CorrelationChart(df,col1,col2,window=20, CorrPrices=False):	Price/regression/correlation chart with basic statistics.
SpreadChart(df,col1,col2): Spread chart.
WordCloud(text,title='',language = 'English', additional_words=[''], UseStopwords = True): WordCloud from the text.
CurrencyVsYieldSpreads(start_date='2015-01-01', symbol='all'): Major currencies vs. 10y yields spread.
NYSEMargin(): Plots NYSE Investor credit balances and SP500.
MargaRET(df, Column, Cut_off_date):	MargaRET predicts future values based on correlation of actual quotes with historical ones.
"""



def ExtractWeekDay(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'WeekDay'] = DataFrame.loc[i, Column].weekday()
	return DataFrame

def ExtractDay(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Day'] = DataFrame.loc[i, Column].day
	return DataFrame
	
def ExtractMonth(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Month'] = DataFrame.loc[i, Column].month
	return DataFrame
	
def ExtractYears(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Year'] = DataFrame.loc[i, Column].year
	return DataFrame

def CalculateRollingPct(DataFrame, Column, Window):
	for i in range(Window+1,len(DataFrame)):
		DataFrame.loc[i, 'PCT'] = (DataFrame.loc[i,Column]/DataFrame.loc[i-Window, Column])-1
	return DataFrame
	
def PercentageChange(DataFrame, Column, Change):
	"""
	dataframe: pandas Dataframe, column: column with data, change: output column
	"""
	for i in range(1,len(DataFrame)):
		DataFrame.loc[i,Change] = (DataFrame.loc[i,Column]/DataFrame.loc[i-1,Column])-1
	return DataFrame

def RenameDays(DataFrame, Column):
		
	for day in range(0,len(DataFrame)):	
		if DataFrame.loc[day,Column] == 0:
			DataFrame.loc[day,Column] ="Monday"
		elif DataFrame.loc[day,Column] == 1:
			DataFrame.loc[day,Column] = "Tuesday"
		elif DataFrame.loc[day,Column] == 2:
			DataFrame.loc[day,Column] = "Wednesday"
		elif DataFrame.loc[day,Column] == 3:
			DataFrame.loc[day,Column] = "Thursday"
		elif DataFrame.loc[day,Column] == 4:
			DataFrame.loc[day,Column] = "Friday"
		elif DataFrame.loc[day,Column] == 5:
			DataFrame.loc[day,Column] = "Saturday"
		elif DataFrame.loc[day,Column] == 6:
			DataFrame.loc[day,Column] = "Sunday"
	return DataFrame
				
def RenameMonths(DataFrame):
	DataFrame.rename(index={1.0:'January', 2.0:'February', 3.0:'March', 4.0:'April', 5.0:'May',6.0:'June',7.0:'July',8.0:'August',9.0:'September', 10.0:'October',11.0:'November',12.0:'December'}, inplace=True)
	return DataFrame
	
def DownloadFromStooq(Symbol, Interval, Part = False, Date_from = '2000-01-01', Date_to = '2100-01-01', Open=False, High=False, Low=False, Close=True, Volume=False):
	"""
	Download data from stooq.plot
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
	
def NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1):

	"""
	Calculate the day return on the day after daily return of PreviousDay value occured.

	Data: Pandas DataFrame, DataColumn: Column with data, PreviousDay: Return on previous day, MoreLess: Operator to test, Days: 1 is next day, 2 are to days after event and so on
	"""
	
	values = PercentageChange(Data, DataColumn, 'Pct_Change')
	
	if MoreLess == 'Less':
		ind = values[values['Pct_Change']<PreviousDay]['Pct_Change'].index + Days
	else:
		ind = values[values['Pct_Change']>PreviousDay]['Pct_Change'].index + Days
			
	final = pd.DataFrame()
	
	for i in ind:
		try:
			final.loc[i, 'Pct_Change'] = values.loc[i,'Pct_Change']
		except:
			continue
		
		final.reset_index(inplace=True, drop=True)
	
	print(final.describe())
	print()
	print('Up days: {}'.format((final[final['Pct_Change']>0]['Pct_Change'].size)),' ', round(((final[final['Pct_Change']>0]['Pct_Change'].size)/((final[final['Pct_Change']>0]['Pct_Change'].size)+(final[final['Pct_Change']<0]['Pct_Change'].size)))*100,2),'%' )
	print('Down days: {}'.format((final[final['Pct_Change']<0]['Pct_Change'].size)),' ', round(((final[final['Pct_Change']<0]['Pct_Change'].size)/((final[final['Pct_Change']>0]['Pct_Change'].size)+(final[final['Pct_Change']<0]['Pct_Change'].size)))*100,2),'%' )
	print()
	print(final.plot(kind="hist", bins=30, figsize=(10,7)))
	
def BarsInaRow(Data, DataColumn):
	"""
	How many times there was a X number of down/up candles in a row.
	
	Data: Pandas DataFrame, DataColumn: Column with data
	"""
	
	values = PercentageChange(Data, DataColumn, 'PCT')
	
	indUP = Data[Data['PCT']>0].index
	resultsUP = pd.DataFrame()
	agg = 1
	for i in range(0,len(indUP)-1):
		if indUP[i+1] == indUP[i] + 1:
			agg += 1 
		else:
			if agg != 0:
				resultsUP.loc[i, 'UP'] = agg
			agg = 1
			
	indDOWN = Data[Data['PCT']<0].index
	resultsDOWN = pd.DataFrame()
	agg = -1
	for i in range(0,len(indDOWN)-1):
		if indDOWN[i+1] == indDOWN[i] + 1:
			agg += -1 
		else:
			if agg != 0:
				resultsDOWN.loc[i, 'UP'] = agg
			agg = -1		
	
	final = pd.DataFrame()
	final = resultsUP
	final = final.append(resultsDOWN)
	res = pd.DataFrame(final['UP'].value_counts())
	res.sort_index(inplace=True)

	anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\nSzymon Nowak (www.1do1.biz.pl)', loc=1)
		
	print(final['UP'].value_counts().sort_index())
	print()

	ax = res.plot(kind="bar", figsize=(12,8), rot=45, legend=False, title='Number of downward(-)/upward(+) candles in a row')
	ax.title.set_size(20)
	ax.add_artist(anchored_text)
	
def SeasonalityPattern(DataFrame, Column, title=''):
	"""
	Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.
	
	DataFrame: Pandas DataFrame, Column: Column with data
	"""
	
	values = ExtractYears(DataFrame, 'Date')
	
	listofyears = DataFrame['Year'].unique()
	listofyears = listofyears[1:]
	
	final = pd.DataFrame()
	for i in listofyears:
		oneyear = pd.DataFrame(DataFrame[DataFrame['Year']==i][Column])
		oneyear.reset_index(inplace=True,drop=True)
		final[i] = oneyear
	
	path = pd.DataFrame()
	for i in listofyears:
		for x in range(0,len(final)):
			path.loc[x,i] = final.loc[x,i]/final.loc[0,i]-1
			
	plotdata = pd.DataFrame()
	for i in range(0,len(path)):
		plotdata.loc[i,'Projection'] = path.iloc[i,:-1].mean()					
	plotdata[listofyears[-1]] = path[listofyears[-1]]
	
	plotdata = plotdata[:-5]

	anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=4)	
	ax = plotdata.plot(y=['Projection',listofyears[-1]], secondary_y=['Projection'],figsize=(12,8), title='Seasonality Pattern Projection ({} r = {}%)'.format(title,round(plotdata.corr().iloc[0,1]*100)))
	ax.set_xlabel('Days')
	ax.add_artist(anchored_text)
	
def SeasonalityDaily(DataFrame, Column):
	"""
	Calculate mean of daily returns and percentage of days with plus/minus daily returns.
	
	DataFrame: Pandas DataFrame, Column: Column with data
	"""
	
	ExtractWeekDay(DataFrame, 'Date')
	RenameDays(DataFrame, 'WeekDay')
	PercentageChange(DataFrame, Column, '{}_pct'.format(Column))
	
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Ones'] = 1
		if DataFrame.loc[i, '{}_pct'.format(Column)] > 0:
			DataFrame.loc[i,'More'] = 1
	
	Plus = 	(DataFrame['More'].groupby(DataFrame['WeekDay']).sum())/(DataFrame['Ones'].groupby(DataFrame['WeekDay']).sum())*100
	Minus = (DataFrame['Ones'].groupby(DataFrame['WeekDay']).sum()-DataFrame['More'].groupby(DataFrame['WeekDay']).sum())/(DataFrame['Ones'].groupby(DataFrame['WeekDay']).sum())*100

	Plus = Plus.to_frame()
	Minus = Minus.to_frame()
	
	results = pd.concat([Plus, Minus], axis=1)
	results.columns = ['Plus', 'Minus']
	
	print('Percentage of days with plus/minus daily returns (%)')
	print()
	print(results)
	print()
	print(DataFrame['{}_pct'.format(Column)].groupby(DataFrame['WeekDay']).mean().plot(kind='bar', figsize=(10,7), title='Mean of daily returns'))
	
def MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'):
	"""
	Calculate number of months when symbol exceeded Level move in Window days.
	
	DataFrame: pandas DataFrame, Column: Column with data, Window: Period of change, Level: Percentage move, MoreLess: More or Less
	"""
	
	CalculateRollingPct(DataFrame, Column, Window)
	ExtractMonth(DataFrame, 'Date')
	if MoreLess ==  'Less':
		final = DataFrame[DataFrame['PCT']<Level].groupby('Month').count()[Column]
	else:
		final = DataFrame[DataFrame['PCT']>Level].groupby('Month').count()[Column]
	RenameMonths(final)
	final.plot(kind='bar',figsize=(10, 7), rot=45, legend=False, title='Number of months when {} was down more than {}% in {} days. '.format(Column, Level*100, Window))
	
def Rebase(DataFrame):
	"""
	Calucalte percent changes to the first value in a column.
	"""
	
	return(DataFrame.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100)))
	
def Fed_rates():

	import pickle
	
	infile = open('data/fed_rates','rb')
	data = pd.read_pickle(infile)
	infile.close()
	return(data)
	
def Plot_X_values_on_Y_dates(df, Column, Dates, fred_api, Range_start=-20, Range_end=60,Title='',PlotAll=False):
	
	"""
	Plot min and max boundaries for the average price action on selected days.
	
	df: pandas DataFrame with values,Column: column in df with values, Dates: pandas DataFrame with dates, fred_api = api key for Fred
	"""
	
	from fredapi import Fred
	import pylab as plt
	
	df.reset_index(inplace=True)
	
	indexes = []
	for i in Dates:
		index = df[df['Date']==i]['index']
		indexes.extend(index)
		
	dates_pd = pd.DataFrame(Dates)
	dates_pd.reset_index(inplace=True)
	
	final = pd.DataFrame()
	for i in range(len(dates_pd)):
		for x in range(Range_start,Range_end):
			final.loc[x,i] = df.loc[indexes[i]+x][Column]
		
	for date in final:
		for i in range(len(final)):
			if i == abs(Range_start):
				continue
			#base = final.iloc[abs(Range_start),date]
			final.iloc[i,date] = (final.iloc[i,date]/final.iloc[abs(Range_start),date]-1)*100		
	
	for date in final:
		final.iloc[abs(Range_start),date] = 0	
	
	final = final.rename(columns=(lambda x:Dates[x]))	
	
	final['average'] = ''
	for i in range(len(final)):
		final.iloc[i,-1] = final.iloc[i,:-1].mean()

	final['min'] = ''
	for i in range(len(final)):
		final.iloc[i,-1] = final.iloc[i,:-1].min()
    
	final['max'] = ''
	for i in range(len(final)):
		final.iloc[i,-1] = final.iloc[i,:-1].max()
		
	final['average'] = pd.to_numeric(final['average'])
	final['min'] = pd.to_numeric(final['min'])
	final['max'] = pd.to_numeric(final['max'])	
	final['zero'] = '0'
	final['zero'] = pd.to_numeric(final['zero'])
	
	if PlotAll == False:
		anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=4)	
		ax = final.plot(y=['min'],figsize=(12,8), label='min',color='Red',title=Title)
		final['max'].plot(ax=ax,label='max',color='Green')
		final['average'].plot(ax=ax,label='average', linewidth=3)
		final['zero'].plot(ax=ax,color='Black',linestyle='dashed',linewidth=1,label='')
		ax.set_xlabel('Days')
		ax.set_ylabel('[%]')
		# ax1 = ax.twinx()
		# ax1.set_ylim(ax.get_ylim())
		plt.legend()
		ax.fill_between(final.index,final['max'],final['min'],alpha=0.3,color='Pink')
		ax.add_artist(anchored_text)
	else:
		anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=4)
		ax = final.plot(figsize=(12,8),legend=False, title=Title)
		ax.set_xlabel('Days')
		ax.set_ylabel('[%]')
		ax.add_artist(anchored_text)
		
def Bubbles():

	import pickle
	
	infile = open('data/bubbles','rb')
	data = pickle.load(infile)
	infile.close()
	return(data)
	
def HeatmapMonthly(df, DateColumn, CloseColumn, Day=False):

	"""
	Heatmap of daily/monthly returns.
	
	df: pandas DataFrame, DateColumn: column with dates, CloseColumn: column with close prices, Day=False: monthly returns
	"""

	import seaborn as sns
	
	df['Year'] = pd.DatetimeIndex(df[DateColumn]).year
	df['Month'] = pd.DatetimeIndex(df[DateColumn]).month
	df['Day'] = pd.DatetimeIndex(df[DateColumn]).weekday
	
	df['Return'] = df[CloseColumn].pct_change()*100
	
	if Day == False:
		avg = df[['Month', 'Return']].groupby(['Month'],as_index=False).mean()
	else:
		avg = df[['Day', 'Return']].groupby(['Day'],as_index=False).mean()
		
	avg['Year'] = 'Avg'
	new = pd.concat([df,avg])
	
	if Day == False:
		final = new.pivot_table(columns='Month',index='Year',values='Return')
	else:
		final = new.pivot_table(columns='Day',index='Year',values='Return')
		final = final.rename(columns={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'})
		
	fig, ax = plt.subplots(figsize=(15,12))
	g = sns.heatmap(final, annot=True, cmap='PiYG')
	ax.invert_yaxis()	

def CorrelationChart(df,col1,col2,window=20, CorrPrices=False):

	"""
	Price/regression/correlation chart with basic statistics.
	
	df: pandas DataFrame, col1: Column with first instrument data, col2: Column with second instrument data, window: rolling correlation winow, CorrPrices=False: Correllate returns instead of prices
	"""
		
	import seaborn as sns
	
	pct = df
	pct = pct.pct_change()
	
	pct['corr'] = pct[col1].rolling(window).corr(pct[col2])
	last_corr = pct['corr'][-1]*100
	full_corr = df.corr()
	full_corr = full_corr.iloc[0,1]*100
	
	print('-'*40)
	print('Returns:')
	print('-'*40)
	
	print(pct.describe())

	print('-'*40)
	print('Prices:')
	print('-'*40)	
	
	print(df.describe())
	
	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, figsize=(15,12), gridspec_kw = {'height_ratios':[3, 1]})	
	ax1.plot(df[col1], color='orange', label='{}'.format(col1))
	axy = ax1.twinx()
	axy.plot(df[col2], label='{}'.format(col2))
	ax2.plot(pct['corr'], linewidth=0.1)
	
	ax1.set_title('{} vs. {} (r={}%)'.format(col1,col2,"%.2f" % full_corr), fontsize=15)
	ax2.set_title('Correlation({}) = {}%'.format(window,"%.2f" % last_corr), fontsize=15)
	
	ax1.grid(False)
	ax1.xaxis.grid(True)
	axy.grid(False)

	ax1.legend(loc=2)
	axy.legend()

	ax1.set_ylabel('{}'.format(col1))
	axy.set_ylabel('{}'.format(col2))

	ax2.fill_between(pct.index.values, 0, pct['corr'], where=pct['corr']>0, interpolate=True, color='green')
	ax2.fill_between(pct.index.values, 0, pct['corr'], where=pct['corr']<0, interpolate=True, color='red')

	anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
	ax1.add_artist(anchored_text)
	
	if CorrPrices==False:
		jointplot = sns.jointplot(col1,col2,pct,kind='reg',size=15)
	else: 
		jointplot = sns.jointplot(col1,col2,df,kind='reg',size=15)	

def SpreadChart(df,col1,col2):

	"""
	Spread chart.
	
	df: pandas DataFrame, col1: Column with first instrument data, col2: Column with second instrument data
	"""
	
	import seaborn as sns

	df = Rebase(df)
	
	df['spread'] = df[col1] - df[col2]
	
	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, figsize=(15,12), gridspec_kw = {'height_ratios':[3, 1]})
	ax1.plot(df[col1],label='{}'.format(col1))
	ax1.plot(df[col2],label='{}'.format(col2))	
	ax2.plot(df['spread'], linewidth=0.1)

	ax1.set_title('{} vs. {}'.format(col1,col2),fontsize=15)
	ax2.set_title('Spread',fontsize=15)
	
	ax1.legend(loc=2)

	ax1.grid(False)
	ax1.xaxis.grid(True)
	ax2.grid(False)
	ax2.xaxis.grid(True)

	ax1.fill_between(df.index.values,df['XAUUSD'],df['CA_C.F'],where=df['XAUUSD']>df['CA_C.F'],interpolate=True,color=sns.xkcd_rgb["denim blue"], alpha=0.2)
	ax1.fill_between(df.index.values,df['XAUUSD'],df['CA_C.F'],where=df['XAUUSD']<df['CA_C.F'],interpolate=True,color=sns.xkcd_rgb["medium green"], alpha=0.2)
					 
	ax2.fill_between(df.index.values,0,df['spread'],where=df['spread']>0,interpolate=True,color=sns.xkcd_rgb["denim blue"], alpha=0.7)
	ax2.fill_between(df.index.values,0,df['spread'],where=df['spread']<0,interpolate=True,color=sns.xkcd_rgb["medium green"], alpha=0.7)

	


def WordCloud(text,title='',language = 'English', additional_words=[''], UseStopwords = True):

	"""
	WordCloud from the text.
	
	text: text, title: chart title, language: 'English' for eng (nltk) and 'pl' (2 letters) for rest of languages (many_stop_words), additional_words: add new stop words, UseStopwords=True: use built-in stopwords
	"""

	from nltk.corpus import stopwords
	import matplotlib.pyplot as plt
	from wordcloud import WordCloud
	import many_stop_words
	
	if UseStopwords == True:
		if language == 'English':
			stop_words = set(stopwords.words('English'))
		else:
			stop_words = many_stop_words.get_stop_words(language)
	else:
		stop_words = set([])

	for w in additional_words:
		stop_words.add(w)
		
	wc = WordCloud(background_color='white',stopwords=stop_words,width=1600, height=800)	
	wc.generate(text)
	
	fig = plt.figure( figsize=(20,10))
	fig.subplots_adjust(top=0.93)
	plt.imshow(wc, interpolation='bilinear')
	fig.suptitle(title,fontsize=20)
	plt.axis("off")
	plt.annotate('by @SzymonNowak1do1', (0,0), (950,-10), xycoords='axes fraction', textcoords='offset points', va='top')
	plt.figure()
	
def CurrencyVsYieldSpreads(start_date='2015-01-01', symbol='all'):

	"""
	Major currencies vs. 10y yields spread.
	
	start_date: Start date of data downloaded from stooq.pl, symbol='all': Plot 3x2 chart with 6 currencies and spreads, choose 'all' or one symbol from [eurusd,audusd,eurjpy,usdjpy,gbpusd,eurgbp,eurpln,usdpln]
	"""

	if symbol == 'all':
	
		# currencies
	
		eurusd = DownloadFromStooq('EURUSD','d',Part=True,Date_from=start_date)
		eurusd.set_index('Date',inplace=True,drop=True)
		
		audusd = DownloadFromStooq('AUDUSD','d',Part=True,Date_from=start_date)
		audusd.set_index('Date',inplace=True,drop=True)
		
		eurjpy = DownloadFromStooq('EURJPY','d',Part=True,Date_from=start_date)
		eurjpy.set_index('Date',inplace=True,drop=True)

		usdjpy = DownloadFromStooq('USDJPY','d',Part=True,Date_from=start_date)
		usdjpy.set_index('Date',inplace=True,drop=True)

		#eurpln = DownloadFromStooq('EURPLN','d',Part=True,Date_from=start_date)
		#eurpln.set_index('Date',inplace=True,drop=True)

		#usdpln = DownloadFromStooq('USDPLN','d',Part=True,Date_from=start_date)
		#usdpln.set_index('Date',inplace=True,drop=True)

		usdcad = DownloadFromStooq('USDCAD','d',Part=True,Date_from=start_date)
		usdcad.set_index('Date',inplace=True,drop=True)

		gbpusd = DownloadFromStooq('GBPUSD','d',Part=True,Date_from=start_date)
		gbpusd.set_index('Date',inplace=True,drop=True)

		eurgbp = DownloadFromStooq('EURGBP','d',Part=True,Date_from=start_date)
		eurgbp.set_index('Date',inplace=True,drop=True)
	
		# yields
	
		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		ger10y = DownloadFromStooq('10DEY.B','d',Part=True,Date_from=start_date)
		ger10y.set_index('Date',inplace=True,drop=True)

		aus10y = DownloadFromStooq('10AUY.B','d',Part=True,Date_from=start_date)
		aus10y.set_index('Date',inplace=True,drop=True)

		jpy10y = DownloadFromStooq('10JPY.B','d',Part=True,Date_from=start_date)
		jpy10y.set_index('Date',inplace=True,drop=True)

		#pol10y = DownloadFromStooq('10PLY.B','d',Part=True,Date_from=start_date)
		#pol10y.set_index('Date',inplace=True,drop=True)

		cad10y = DownloadFromStooq('10CAY.B','d',Part=True,Date_from=start_date)
		cad10y.set_index('Date',inplace=True,drop=True)

		uk10y = DownloadFromStooq('10UKY.B','d',Part=True,Date_from=start_date)
		uk10y.set_index('Date',inplace=True,drop=True)
		
			# spreads
	
		eurusd['spread'] = ger10y['10DEY.B'] - us10y['10USY.B']
		audusd['spread'] = aus10y['10AUY.B'] - us10y['10USY.B']
		eurjpy['spread'] = ger10y['10DEY.B'] - jpy10y['10JPY.B']
		usdjpy['spread'] = us10y['10USY.B'] - jpy10y['10JPY.B']
		#eurpln['spread'] = ger10y['10DEY.B'] - pol10y['10PLY.B']	
		#usdpln['spread'] = us10y['10USY.B'] - pol10y['10PLY.B']
		usdcad['spread'] = us10y['10USY.B'] - cad10y['10CAY.B']
		gbpusd['spread'] = uk10y['10UKY.B'] - us10y['10USY.B']
		eurgbp['spread'] = ger10y['10DEY.B'] - uk10y['10UKY.B']
		
	elif symbol == 'eurusd':
	
		eurusd = DownloadFromStooq('EURUSD','d',Part=True,Date_from=start_date)
		eurusd.set_index('Date',inplace=True,drop=True)
		
		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		ger10y = DownloadFromStooq('10DEY.B','d',Part=True,Date_from=start_date)
		ger10y.set_index('Date',inplace=True,drop=True)
	
		eurusd['spread'] = ger10y['10DEY.B'] - us10y['10USY.B']
		
	elif symbol == 'audusd':
	
		audusd = DownloadFromStooq('AUDUSD','d',Part=True,Date_from=start_date)
		audusd.set_index('Date',inplace=True,drop=True)

		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		aus10y = DownloadFromStooq('10AUY.B','d',Part=True,Date_from=start_date)
		aus10y.set_index('Date',inplace=True,drop=True)
		
		audusd['spread'] = aus10y['10AUY.B'] - us10y['10USY.B']		
		
	elif symbol == 'eurjpy':		
	
		eurjpy = DownloadFromStooq('EURJPY','d',Part=True,Date_from=start_date)
		eurjpy.set_index('Date',inplace=True,drop=True)
		
		ger10y = DownloadFromStooq('10DEY.B','d',Part=True,Date_from=start_date)
		ger10y.set_index('Date',inplace=True,drop=True)

		jpy10y = DownloadFromStooq('10JPY.B','d',Part=True,Date_from=start_date)
		jpy10y.set_index('Date',inplace=True,drop=True)		
		
		eurjpy['spread'] = ger10y['10DEY.B'] - jpy10y['10JPY.B']		

	elif symbol == 'usdjpy':	
		
		usdjpy = DownloadFromStooq('USDJPY','d',Part=True,Date_from=start_date)
		usdjpy.set_index('Date',inplace=True,drop=True)		
		
		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		jpy10y = DownloadFromStooq('10JPY.B','d',Part=True,Date_from=start_date)
		jpy10y.set_index('Date',inplace=True,drop=True)	

		usdjpy['spread'] = us10y['10USY.B'] - jpy10y['10JPY.B']		
	
	elif symbol == 'eurpln':		
	
		eurpln = DownloadFromStooq('EURPLN','d',Part=True,Date_from=start_date)
		eurpln.set_index('Date',inplace=True,drop=True)
		
		ger10y = DownloadFromStooq('10DEY.B','d',Part=True,Date_from=start_date)
		ger10y.set_index('Date',inplace=True,drop=True)

		pol10y = DownloadFromStooq('10PLY.B','d',Part=True,Date_from=start_date)
		pol10y.set_index('Date',inplace=True,drop=True)	

		eurpln['spread'] = ger10y['10DEY.B'] - pol10y['10PLY.B']		
		
	elif symbol == 'usdpln':	

		usdpln = DownloadFromStooq('USDPLN','d',Part=True,Date_from=start_date)
		usdpln.set_index('Date',inplace=True,drop=True)	
		
		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		pol10y = DownloadFromStooq('10PLY.B','d',Part=True,Date_from=start_date)
		pol10y.set_index('Date',inplace=True,drop=True)		
		
		usdpln['spread'] = us10y['10USY.B'] - pol10y['10PLY.B']		
		
	elif symbol == 'usdcad':		

		usdcad = DownloadFromStooq('USDCAD','d',Part=True,Date_from=start_date)
		usdcad.set_index('Date',inplace=True,drop=True)	
		
		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		cad10y = DownloadFromStooq('10CAY.B','d',Part=True,Date_from=start_date)
		cad10y.set_index('Date',inplace=True,drop=True)		
		
		usdcad['spread'] = us10y['10USY.B'] - cad10y['10CAY.B']		
		
	elif symbol == 'gbpusd':
	
		gbpusd = DownloadFromStooq('GBPUSD','d',Part=True,Date_from=start_date)
		gbpusd.set_index('Date',inplace=True,drop=True)	
		
		us10y = DownloadFromStooq('10USY.B','d',Part=True,Date_from=start_date)
		us10y.set_index('Date',inplace=True,drop=True)

		uk10y = DownloadFromStooq('10UKY.B','d',Part=True,Date_from=start_date)
		uk10y.set_index('Date',inplace=True,drop=True)		

		gbpusd['spread'] = uk10y['10UKY.B'] - us10y['10USY.B']		
		
	elif symbol == 'eurgbp':	

		eurgbp = DownloadFromStooq('EURGBP','d',Part=True,Date_from=start_date)
		eurgbp.set_index('Date',inplace=True,drop=True)	
		
		ger10y = DownloadFromStooq('10DEY.B','d',Part=True,Date_from=start_date)
		ger10y.set_index('Date',inplace=True,drop=True)

		uk10y = DownloadFromStooq('10UKY.B','d',Part=True,Date_from=start_date)
		uk10y.set_index('Date',inplace=True,drop=True)		
	
		eurgbp['spread'] = ger10y['10DEY.B'] - uk10y['10UKY.B']
	
	# plot
	
	if symbol == 'all':

		fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3, sharex=False, figsize=(15,15))

		fig.suptitle('Currency vs. 10Y yields spread',fontsize=20)

		ax1.plot(eurusd['EURUSD'],label='EURUSD')
		ax1.legend(loc=2)
		ax12 = ax1.twinx()
		ax12.plot(eurusd['spread'],label='Spread',color='red')
		ax12.legend()
		ax1.set_title('EURUSD')

		ax2.plot(gbpusd['GBPUSD'],label='GBPUSD')
		ax2.legend(loc=2)
		ax22 = ax2.twinx()
		ax22.plot(gbpusd['spread'],label='Spread',color='red')
		ax22.legend()
		ax2.set_title('GBPUSD')

		ax3.plot(usdjpy['USDJPY'],label='USDJPY')
		ax3.legend(loc=2)
		ax32 = ax3.twinx()
		ax32.plot(usdjpy['spread'],label='Spread',color='red')
		ax32.legend()
		ax3.set_title('USDJPY')

		ax4.plot(eurjpy['EURJPY'],label='EURJPY')
		ax4.legend(loc=2)
		ax42 = ax4.twinx()
		ax42.plot(eurjpy['spread'],label='Spread',color='red')
		ax42.legend()
		ax4.set_title('EURJPY')

		ax5.plot(usdcad['USDCAD'],label='USDCAD')
		ax5.legend(loc=2)
		ax52 = ax5.twinx()
		ax52.plot(usdcad['spread'],label='Spread',color='red')
		ax52.legend()
		ax5.set_title('USDCAD')

		ax6.plot(audusd['AUDUSD'],label='AUDUSD')
		ax6.legend(loc=2)
		ax62 = ax6.twinx()
		ax62.plot(audusd['spread'],label='Spread',color='red')
		ax62.legend()
		ax6.set_title('AUDUSD')
		
		"""
		ax7.plot(usdpln['USDPLN'],label='USDPLN')
		ax7.legend(loc=2)
		ax72 = ax7.twinx()
		ax72.plot(usdpln['spread'],label='Spread',color='red')
		ax72.legend(loc=1)
		ax7.set_title('USDPLN')

		ax8.plot(eurpln['EURPLN'],label='EURPLN')
		ax8.legend(loc=2)
		ax82 = ax8.twinx()
		ax82.plot(eurpln['spread'],label='Spread',color='red')
		ax82.legend()
		ax8.set_title('EURPLN')
		"""
		
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.annotate('by Szymon Nowak (www.szymonnowak.com)', (0,0), (250,-30), xycoords='axes fraction', textcoords='offset points', va='top')

		
	else:
	
		fig, ax = plt.subplots(figsize=(15,12))
		
		if symbol=='eurusd':
			ax.plot(eurusd['EURUSD'],label='EURUSD')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(eurusd['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('EURUSD vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)		
		
		elif symbol=='gbpusd':
			ax.plot(gbpusd['GBPUSD'],label='GBPUSD')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(gbpusd['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('GBPUSD vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)			
			
		elif symbol=='usdjpy':
			ax.plot(usdjpy['USDJPY'],label='USDJPY')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(usdjpy['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('USDJPY vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)

		elif symbol=='eurjpy':
			ax.plot(eurjpy['EURJPY'],label='EURJPY')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(eurjpy['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('EURJPY vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)			
			
		elif symbol=='audusd':
			ax.plot(audusd['AUDUSD'],label='AUDUSD')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(audusd['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('AUDUSD vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)			
			
		elif symbol=='usdcad':
			ax.plot(usdcad['USDCAD'],label='USDCAD')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(usdcad['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('USDCAD vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)				
			
		elif symbol=='usdpln':
			ax.plot(usdpln['USDPLN'],label='USDPLN')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(usdpln['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('USDPLN vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)				
			
		elif symbol=='eurpln':
			ax.plot(eurpln['EURPLN'],label='EURPLN')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(eurpln['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('EURPLN vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)	

		elif symbol=='eurgpb':
			ax.plot(eurgpb['EURGPB'],label='EURGPB')
			ax.legend(loc=2)
			ax2 = ax.twinx()
			ax2.plot(eurgpb['spread'],label='Spread',color='red')
			ax2.legend()
			ax.set_title('EURGPB vs. 10Y yields')		
			anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
			ax.add_artist(anchored_text)		

def NYSEMargin():

	"""
	Plots NYSE Investor credit balances and SP500.
	"""

	import quandl
	import numpy as np
	import pickle
	
	infile = open('data/nysecredit','rb')
	data = pickle.load(infile)
	infile.close()
	
	new = quandl.get('NYXDATA/MARKET_CREDIT', start_date='1992-01-01')
	
	data_index = data.index.values.tolist()
	new_index = new.index.values.tolist()
	
	not_included = [i for i in new_index if i not in data_index]
	
	if (len(not_included))>0:
		new_data = new[-(len(not_included)):]
		data = data.append(new_data)
		del data['Credit balance']
		
		spx = DownloadFromStooq('^spx','m', Part=True, Date_from='1992-01-01')
		
		data.reset_index(inplace=True)
		
		data['Month'] = pd.DatetimeIndex(data['End of month']).month
		data['Year'] = pd.DatetimeIndex(data['End of month']).year
		spx['Month'] = pd.DatetimeIndex(spx['Date']).month
		spx['Year'] = pd.DatetimeIndex(spx['Date']).year
		
		data.set_index('End of month',inplace=True,drop=True)
		
		data_index = data.index.values.tolist()
		data['nan'] = data['S&P500'].apply(np.isnan)
		index = data[data['nan']==True].index.values.tolist()
		nan_list = [data_index.index(i) for i in index]
		
		data.reset_index(inplace=True)
		
		for i in nan_list:
			for x in range(0,len(spx)):
				month = data.loc[i,'Month']
				year = data.loc[i,'Year']
				if ((spx.iloc[x,2] == month) & (spx.iloc[x,3] == year)):
					data.loc[i,'S&P500'] = spx.loc[x,'^spx']
					break
		
		data.set_index('End of month',inplace=True,drop=True)
		
		data['Free credit cash accounts'] = data.fillna(0)['Free credit cash accounts']
		data['Credit balance'] = (-(data['Free credit cash accounts'] + data['Credit balances in margin accounts'] - data['Margin debt']))/1000
		
		outfile = open('data/nysecredit','wb')
		pickle.dump(data,outfile)
		outfile.close()
		
	fig, ax = plt.subplots(figsize=(15,10))
	ax.plot(data['S&P500'], color='blue')
	ax2 = ax.twinx()
	ax2.plot(data['Credit balance'], linewidth=0.1,label='')

	ax.set_title('NYSE Investor Credit and the the S&P500',fontsize=15)

	ax.set_ylabel('S&P500')
	ax2.set_ylabel('Billions [USD]', rotation=270)

	ax2.fill_between(data.index.values,0,data['Credit balance'],where=data['Credit balance']>0,interpolate=True,color='red',alpha=0.3,label='Negative balance')
	ax2.fill_between(data.index.values,0,data['Credit balance'],where=data['Credit balance']<0,interpolate=True,color='green',alpha=0.3,label='Positive balance')

	ax.legend(loc=4)
	ax2.legend()

	anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
	ax.add_artist(anchored_text)
	
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
		
	max_corr = correlation_table['Correlation'].argmax()

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
	
	anchored_text = AnchoredText('                AlysSA v0.15b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
	ax.add_artist(anchored_text)
	
def Returns(type='indexes', method='YTD', Date_from=''):

	if method == 'YTD':
		Date_from='2017-01-01'

	if type == 'indexes':
		wig20 = DownloadFromStooq('WIG20','d', Part=True, Date_from=Date_from)
		wig = DownloadFromStooq('WIG','d', Part=True, Date_from=Date_from)
		spx = DownloadFromStooq('^spx','d', Part=True, Date_from=Date_from)
		dax = DownloadFromStooq('^dax','d', Part=True, Date_from=Date_from)
		rts = DownloadFromStooq('^rts','d', Part=True, Date_from=Date_from)
		#ftse = DownloadFromStooq('X,C','d', Part=True, Date_from=Date_from)
		
		symbols = [wig20,wig,spx,dax,rts]
		
		fig, ax = plt.subplots(figsize=(15,10))
		
		for symbol in symbols:
			symbol.set_index('Date',inplace=True,drop=True)

		wig20 = Rebase(wig20)
		wig = Rebase(wig)
		spx = Rebase(spx)
		dax = Rebase(dax)
		rts = Rebase(rts)
		#ftse = Rebase(ftse)			
			
		ax.plot(wig20, label='WIG20')
		ax.plot(wig, label='WIG')
		ax.plot(spx, label='SP500')
		ax.plot(dax, label='DAX30')
		ax.plot(dax, label='RTS')		
		#ax.plot(dax, label='FTSE100')		
		
		if method == 'YTD':		
			ax.set_title('Indexes return (YTD)', fontsize=15)
		else:
			ax.set_title('Indexes return (from {})'.format(Date_from), fontsize=15)

		ax.set_ylabel('[%]')				
		ax.legend()
		
	if type == 'yields':
		pl10y = DownloadFromStooq('10PLY.B','d', Part=True, Date_from=Date_from)
		us10y = DownloadFromStooq('10USY.B','d', Part=True, Date_from=Date_from)
		de10y = DownloadFromStooq('10DEY.B','d', Part=True, Date_from=Date_from)
		uk10y= DownloadFromStooq('10UKY.B','d', Part=True, Date_from=Date_from)
		gr10y = DownloadFromStooq('10GRY.B','d', Part=True, Date_from=Date_from)
		ca10y = DownloadFromStooq('10CAY.B','d', Part=True, Date_from=Date_from)
		
		symbols = [pl10y,us10y,de10y,uk10y,gr10y,ca10y]
		
		fig, ax = plt.subplots(figsize=(15,10))
		
		for symbol in symbols:
			symbol.set_index('Date',inplace=True,drop=True)
			
		pl10y = Rebase(pl10y)
		us10y = Rebase(us10y)
		de10y = Rebase(de10y)
		uk10y = Rebase(uk10y)
		gr10y = Rebase(gr10y)
		ca10y = Rebase(ca10y)
			
		ax.plot(pl10y, label='Poland')
		ax.plot(us10y, label='United States')
		ax.plot(de10y, label='Germany')
		ax.plot(uk10y, label='United Kingdom')
		ax.plot(gr10y, label='Greece')		
		ax.plot(ca10y, label='Canada')	

		if method == 'YTD':		
			ax.set_title('Yields return (YTD)', fontsize=15)
		else:
			ax.set_title('Yields return (from {})'.format(Date_from), fontsize=15)
			
		ax.set_ylabel('[%]')	
		ax.legend()		

		
	if type == 'commodities':
		gold = DownloadFromStooq('XAUUSD','d', Part=True, Date_from=Date_from)
		wti = DownloadFromStooq('CL.C','d', Part=True, Date_from=Date_from)
		coffee = DownloadFromStooq('KC.C','d', Part=True, Date_from=Date_from)
		sugar = DownloadFromStooq('SB.C','d', Part=True, Date_from=Date_from)
		copper = DownloadFromStooq('CA_C.F','d', Part=True, Date_from=Date_from)
		silver = DownloadFromStooq('SI.F','d', Part=True, Date_from=Date_from)
		
		symbols = [gold,wti,coffee,sugar,copper,silver]
		
		fig, ax = plt.subplots(figsize=(15,10))
		
		for symbol in symbols:
			symbol.set_index('Date',inplace=True,drop=True)
			
		gold = Rebase(gold)
		wti = Rebase(wti)
		coffee = Rebase(coffee)
		sugar = Rebase(sugar)
		copper = Rebase(copper)
		silver = Rebase(silver)				
			
		ax.plot(gold, label='Gold')
		ax.plot(wti, label='WTI')
		ax.plot(coffee, label='Coffee')
		ax.plot(sugar, label='Sugar')
		ax.plot(copper, label='Copper')		
		ax.plot(silver, label='Silver')	

		if method == 'YTD':		
			ax.set_title('Commodities return (YTD)', fontsize=15)
		else:
			ax.set_title('Commodities return (from {})'.format(Date_from), fontsize=15)

		ax.set_ylabel('[%]')			
		ax.legend()		
		
	if type == 'currencies':
		eurusd = DownloadFromStooq('EURUSD','d', Part=True, Date_from=Date_from)
		gbpusd = DownloadFromStooq('GBPUSD','d', Part=True, Date_from=Date_from)
		audusd = DownloadFromStooq('AUDUSD','d', Part=True, Date_from=Date_from)
		jpyusd = DownloadFromStooq('JPYUSD','d', Part=True, Date_from=Date_from)
		cadusd = DownloadFromStooq('CADUSD','d', Part=True, Date_from=Date_from)
		plnusd = DownloadFromStooq('PLNUSD','d', Part=True, Date_from=Date_from)
		nzdusd = DownloadFromStooq('NZDUSD','d', Part=True, Date_from=Date_from)
		
		symbols = [eurusd,gbpusd,audusd,jpyusd,cadusd,plnusd,nzdusd]
		
		fig, ax = plt.subplots(figsize=(15,10))
		
		for symbol in symbols:
			symbol.set_index('Date',inplace=True,drop=True)

		eurusd = Rebase(eurusd)
		gbpusd = Rebase(gbpusd)
		audusd = Rebase(audusd)
		jpyusd = Rebase(jpyusd)
		cadusd = Rebase(cadusd)
		plnusd = Rebase(plnusd)	
		nzdusd = Rebase(nzdusd)			
			
		ax.plot(eurusd, label='EUR')
		ax.plot(gbpusd, label='GBP')
		ax.plot(audusd, label='AUD')
		ax.plot(jpyusd, label='JPY')
		ax.plot(cadusd, label='CAD')		
		ax.plot(plnusd, label='PLN')	
		ax.plot(nzdusd, label='NZD')			

		if method == 'YTD':		
			ax.set_title('Currencies return (YTD)', fontsize=15)
		else:
			ax.set_title('Currencies return (from {})'.format(Date_from), fontsize=15)
			
		ax.set_ylabel('[%]')			
		ax.legend()				
		
def AAII(Buy_signal=0.58, Sell_signal=0.64):
	
	import quandl
	
	df = quandl.get('AAII/AAII_SENTIMENT',start_date="1996-01-01")
	spx = DownloadFromStooq('^spx','d',Part=True,Date_from='1996-01-01')
	spx.set_index('Date',inplace=True,drop=True)
	df['SP500'] = spx['^spx']
	df.reset_index(inplace=True)
	df['Sell signal'] = df['Bullish']>Sell_signal
	df['Buy Signal'] = df['Bearish']>Buy_signal
	
	marker_style_sell = dict(color='red', linestyle='-', marker='v',markersize=15, markerfacecoloralt='gray', label='Sell signal (more than {}% investors are bullish)'.format('%.2f' % (Sell_signal*100)))
	marker_style_buy = dict(color='mediumblue', linestyle='-', marker='^', markersize=15, markerfacecoloralt='gray', label='Buy signal (more than {}% investors are bearish)'.format('%.2f' % (Buy_signal*100)))

	fig, ax = plt.subplots(figsize=(15,12))
	# ax = data['S&P 500 Weekly Close'].plot()
	markers_on = df.loc[df['Sell signal']==True].index.values
	plt.plot(df['SP500'],'-gD',markevery=markers_on, **marker_style_sell)
	markers_on_buy = df.loc[df['Buy Signal']==True].index.values
	plt.plot(df['SP500'],'-gD', markevery=markers_on_buy, **marker_style_buy)

	a=ax.get_xticks().tolist()
	a = ['1991','1995','1999','2003','2007','2011','2015','2019']
	ax.set_xticklabels(a)
	
	ax.set_title('AAII Sentiment vs. SP500',fontsize=15)
	ax.legend()
	
def INI(excel='ini.xlsx'):
		
	ini = pd.read_excel(excel)
	wig = DownloadFromStooq('WIG','d',Part=True,Date_from='2011-05-19')
	wig.set_index('Date',inplace=True,drop=True)
	ini = ini*100
	
	fig, ax = plt.subplots(figsize=(15,10))
	ax.plot(ini['spadkowy'],color='red',alpha=0.3,label='Sentyment spadkowy - {}%'.format('%.2f' % (ini['spadkowy'][-1])))
	ax2 = ax.twinx()
	ax2.plot(wig, label='WIG - {}'.format(wig['WIG'][-1]))
	ax.set_ylim(ax.get_ylim()[::-1])
	ax2.set_ylabel('WIG')
	ax.set_ylabel('[%] inwestorów nastawionych na spadki (oś odwrócona)')
	ax.set_title('Sentyment spadkowy (INI) vs. WIG', fontsize=15)
	ax.legend(loc=2)
	ax2.legend(loc=1)
	ax.grid(False)
	ax2.grid(False)
	
	anchored_text = AnchoredText('by Szymon Nowak (@SzymonNowak1do1)', loc=3)
	ax.add_artist(anchored_text)
	
	fig2, ax3 = plt.subplots(figsize=(15,10))
	ax3.plot(ini['wzrostowy'],color='green',alpha=0.3,label='Sentyment wzrostowy - {}%'.format('%.2f' % (ini['wzrostowy'][-1])))
	ax4 = ax3.twinx()
	ax4.plot(wig, label='WIG - {}'.format(wig['WIG'][-1]))
	ax4.set_ylabel('WIG')
	ax3.set_ylabel('[%] inwestorów nastawionych na wzrosty')
	ax3.set_title('Sentyment wzrostowy (INI) vs. WIG', fontsize=15)
	ax3.legend(loc=2)
	ax4.legend(loc=1)
	ax3.grid(False)
	ax4.grid(False)
	
	anchored_text = AnchoredText('by Szymon Nowak (@SzymonNowak1do1)', loc=3)
	ax3.add_artist(anchored_text)
	
def SP500Indicators():

	import quandl
	import datetime
	import numpy as np
	
	shillerpe = quandl.get('MULTPL/SHILLER_PE_RATIO_MONTH')
	dividendgrowth = quandl.get('MULTPL/SP500_DIV_YIELD_MONTH')
	salesgrowth = quandl.get('MULTPL/SP500_SALES_GROWTH_QUARTER')
	earningsgrowth = quandl.get('MULTPL/SP500_EARNINGS_GROWTH_QUARTER')
	earnings = quandl.get('MULTPL/SP500_EARNINGS_MONTH')
	pbv = quandl.get('MULTPL/SP500_PBV_RATIO_QUARTER')
	
	shillerpe = shillerpe['1990-01-01':]
	dividendgrowth = dividendgrowth['1990-01-01':]
	earnings = earnings['1990-01-01':]
	pbv = pbv['2002-01-01':]
	
	shillerpe_mean = [np.mean(shillerpe)]*len(shillerpe)
	dividendgrowth_mean = [np.mean(dividendgrowth)]*len(dividendgrowth)
	salesgrowth_mean = [np.mean(salesgrowth)]*len(salesgrowth)
	earningsgrowth_mean = [np.mean(earningsgrowth)]*len(earningsgrowth)
	earnings_mean = [np.mean(earnings)]*len(earnings)
	pbv_mean = [np.mean(pbv)]*len(pbv)
	
	bbox_props = dict(boxstyle='larrow')

	fig, ((ax1, ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3,ncols=2, figsize=(15,15))

	fig.suptitle('SP500 indicators and mean values',fontsize=15)
	fig.subplots_adjust(top=0.93)

	ax1.set_title('Shiller PE')
	ax1.plot(shillerpe)
	ax1.plot(shillerpe.index.values,shillerpe_mean, linestyle='--')
	ax1.annotate(str(shillerpe['Value'][-1]), (shillerpe.index[-1], shillerpe['Value'][-1]), xytext = (shillerpe.index[-1]+ datetime.timedelta(weeks=120), shillerpe['Value'][-1]),bbox=bbox_props,color='white')

	ax2.set_title('P/BV')
	ax2.plot(pbv)
	ax2.plot(pbv.index.values,pbv_mean, linestyle='--')
	ax2.annotate(str(pbv['Value'][-1]), (pbv.index[-1], pbv['Value'][-1]), xytext = (pbv.index[-1]+ datetime.timedelta(weeks=65), pbv['Value'][-1]),bbox=bbox_props,color='white')

	ax3.set_title('Dividend growth')
	ax3.plot(dividendgrowth)
	ax3.plot(dividendgrowth.index.values,dividendgrowth_mean, linestyle='--')
	ax3.annotate(str(dividendgrowth['Value'][-1]), (dividendgrowth.index[-1], dividendgrowth['Value'][-1]), xytext = (dividendgrowth.index[-1]+ datetime.timedelta(weeks=120), dividendgrowth['Value'][-1]),bbox=bbox_props,color='white')

	ax4.set_title('Sales growth')
	ax4.plot(salesgrowth,label='Sales growth')
	ax4.plot(salesgrowth.index.values,salesgrowth_mean, linestyle='--')
	ax4.annotate(str(salesgrowth['Value'][-1]), (salesgrowth.index[-1], salesgrowth['Value'][-1]), xytext = (salesgrowth.index[-1]+ datetime.timedelta(weeks=65), salesgrowth['Value'][-1]),bbox=bbox_props,color='white')

	ax5.set_title('Earnings growth')
	ax5.plot(earningsgrowth,label='Earnings growth')
	ax5.plot(earningsgrowth.index.values,earningsgrowth_mean, linestyle='--')
	ax5.annotate(str(earningsgrowth['Value'][-1]), (earningsgrowth.index[-1], earningsgrowth['Value'][-1]), xytext = (earningsgrowth.index[-1]+ datetime.timedelta(weeks=120), earningsgrowth['Value'][-1]),bbox=bbox_props,color='white')

	ax6.set_title('Earnings')
	ax6.plot(earnings)
	ax6.plot(earnings.index.values,earnings_mean, linestyle='--')
	ax6.annotate(str(earnings['Value'][-1]), (earnings.index[-1], earnings['Value'][-1]), xytext = (earnings.index[-1]+ datetime.timedelta(weeks=120), earnings['Value'][-1]),bbox=bbox_props,color='white')
	
	plt.annotate('by @SzymonNowak1do1', (0,0), (250,-30), xycoords='axes fraction', textcoords='offset points', va='top')
	
def WIGIndicators(index='WIG'):

	import numpy as np
	import datetime

	if index == 'WIG':

		wig = DownloadFromStooq('WIG','d')
		wigPE = DownloadFromStooq('WIG_PE','d')
		wigPB = DownloadFromStooq('WIG_PB','d')
		wigDY = DownloadFromStooq('WIG_DY','d')
		wigMV = DownloadFromStooq('WIG_MV','d')

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
		ax3.set_title('Dividend Yield')
		ax4.set_title('P/BV')
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
		ax2.annotate(str(wigPE['WIG_PE'][-1]), (wigPE.index[-1], wigPE['WIG_PE'][-1]), xytext = (wigPE.index[-1]+ datetime.timedelta(weeks=35), wigPE['WIG_PE'][-1]),bbox=bbox_props,color='white')
		ax3.annotate(str(wigPB['WIG_PB'][-1]), (wigPB.index[-1], wigPB['WIG_PB'][-1]), xytext = (wigPB.index[-1]+ datetime.timedelta(weeks=43), wigPB['WIG_PB'][-1]),bbox=bbox_props,color='white')
		ax4.annotate(str(wigDY['WIG_DY'][-1]), (wigDY.index[-1], wigDY['WIG_DY'][-1]), xytext = (wigDY.index[-1]+ datetime.timedelta(weeks=35), wigDY['WIG_DY'][-1]),bbox=bbox_props,color='white')
		ax5.annotate(str(wigMV['WIG_MV'][-1]), (wigMV.index[-1], wigMV['WIG_MV'][-1]), xytext = (wigMV.index[-1]+ datetime.timedelta(weeks=43), wigMV['WIG_MV'][-1]),bbox=bbox_props,color='white')

		ax1.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		ax5.legend()

		plt.tight_layout()
		plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
		plt.show()
		
	elif index == 'mWIG40':

		mwig40 = DownloadFromStooq('MWIG40','d')
		mwig40TR = DownloadFromStooq('MWIG40TR','d')
		mwig40PE = DownloadFromStooq('MWIG40_PE','d')
		mwig40PB = DownloadFromStooq('MWIG40_PB','d')
		mwig40DY = DownloadFromStooq('MWIG40_DY','d')
		mwig40MV = DownloadFromStooq('MWIG40_MV','d')

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
		ax3.set_title('Dividend Yield')
		ax4.set_title('P/BV')
		ax5.set_title('Market Value')

		ax1.plot(mwig40,label='mWIG40')
		ax1.plot(mwig40TR, label='mWIG40 TR')
		ax2.plot(mwig40PE)
		ax3.plot(mwig40DY)
		ax4.plot(mwig40PB)
		ax5.plot(mwig40MV)

		ax1.plot(mwig40.index.values,mwig40_mean, linestyle='--', label='average')
		ax1.plot(mwig40TR.index.values,mwig40TR_mean, linestyle='--', label='TR average')
		ax2.plot(mwig40PE.index.values,mwig40PE_mean, linestyle='--', label='average')
		ax3.plot(mwig40DY.index.values,mwig40DY_mean, linestyle='--', label='average')
		ax4.plot(mwig40PB.index.values,mwig40PB_mean, linestyle='--', label='average')
		ax5.plot(mwig40MV.index.values,mwig40MV_mean, linestyle='--', label='average')

		bbox_props = dict(boxstyle='larrow')
		bbox_props_2 = dict(boxstyle='larrow', color='orange')
		ax1.annotate(str(mwig40['MWIG40'][-1]), (mwig40.index[-1], mwig40['MWIG40'][-1]), xytext = (mwig40.index[-1]+ datetime.timedelta(weeks=65), mwig40['MWIG40'][-1]),bbox=bbox_props,color='white')
		ax1.annotate(str(mwig40TR['MWIG40TR'][-1]), (mwig40TR.index[-1], mwig40TR['MWIG40TR'][-1]), xytext = (mwig40TR.index[-1]+ datetime.timedelta(weeks=65), mwig40TR['MWIG40TR'][-1]),bbox=bbox_props_2,color='black')
		ax2.annotate(str(mwig40PE['MWIG40_PE'][-1]), (mwig40PE.index[-1], mwig40PE['MWIG40_PE'][-1]), xytext = (mwig40PE.index[-1]+ datetime.timedelta(weeks=35), mwig40PE['MWIG40_PE'][-1]),bbox=bbox_props,color='white')
		ax3.annotate(str(mwig40DY['MWIG40_DY'][-1]), (mwig40DY.index[-1], mwig40DY['MWIG40_DY'][-1]), xytext = (mwig40DY.index[-1]+ datetime.timedelta(weeks=43), mwig40DY['MWIG40_DY'][-1]),bbox=bbox_props,color='white')
		ax4.annotate(str(mwig40PB['MWIG40_PB'][-1]), (mwig40PB.index[-1], mwig40PB['MWIG40_PB'][-1]), xytext = (mwig40PB.index[-1]+ datetime.timedelta(weeks=35), mwig40PB['MWIG40_PB'][-1]),bbox=bbox_props,color='white')
		ax5.annotate(str(mwig40MV['MWIG40_MV'][-1]), (mwig40MV.index[-1], mwig40MV['MWIG40_MV'][-1]), xytext = (mwig40MV.index[-1]+ datetime.timedelta(weeks=43), mwig40MV['MWIG40_MV'][-1]),bbox=bbox_props,color='white')

		ax1.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		ax5.legend()

		plt.tight_layout()
		plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
		plt.show()

	elif index == 'sWIG80':

		mwig40 = DownloadFromStooq('SWIG80','d')
		mwig40TR = DownloadFromStooq('SWIG80TR','d')
		mwig40PE = DownloadFromStooq('SWIG80_PE','d')
		mwig40PB = DownloadFromStooq('SWIG80_PB','d')
		mwig40DY = DownloadFromStooq('SWIG80_DY','d')
		mwig40MV = DownloadFromStooq('SWIG80_MV','d')

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
		ax3.set_title('Dividend Yield')
		ax4.set_title('P/BV')
		ax5.set_title('Market Value')

		ax1.plot(mwig40,label='sWIG80')
		ax1.plot(mwig40TR, label='sWIG80 TR')
		ax2.plot(mwig40PE)
		ax3.plot(mwig40DY)
		ax4.plot(mwig40PB)
		ax5.plot(mwig40MV)

		ax1.plot(mwig40.index.values,mwig40_mean, linestyle='--', label='average')
		ax1.plot(mwig40TR.index.values,mwig40TR_mean, linestyle='--', label='TR average')
		ax2.plot(mwig40PE.index.values,mwig40PE_mean, linestyle='--', label='average')
		ax3.plot(mwig40DY.index.values,mwig40DY_mean, linestyle='--', label='average')
		ax4.plot(mwig40PB.index.values,mwig40PB_mean, linestyle='--', label='average')
		ax5.plot(mwig40MV.index.values,mwig40MV_mean, linestyle='--', label='average')

		bbox_props = dict(boxstyle='larrow')
		bbox_props_2 = dict(boxstyle='larrow', color='orange')
		ax1.annotate(str(mwig40['SWIG80'][-1]), (mwig40.index[-1], mwig40['SWIG80'][-1]), xytext = (mwig40.index[-1]+ datetime.timedelta(weeks=65), mwig40['SWIG80'][-1]),bbox=bbox_props,color='white')
		ax1.annotate(str(mwig40TR['SWIG80TR'][-1]), (mwig40TR.index[-1], mwig40TR['SWIG80TR'][-1]), xytext = (mwig40TR.index[-1]+ datetime.timedelta(weeks=65), mwig40TR['SWIG80TR'][-1]),bbox=bbox_props_2,color='black')
		ax2.annotate(str(mwig40PE['SWIG80_PE'][-1]), (mwig40PE.index[-1], mwig40PE['SWIG80_PE'][-1]), xytext = (mwig40PE.index[-1]+ datetime.timedelta(weeks=35), mwig40PE['SWIG80_PE'][-1]),bbox=bbox_props,color='white')
		ax3.annotate(str(mwig40DY['SWIG80_DY'][-1]), (mwig40DY.index[-1], mwig40DY['SWIG80_DY'][-1]), xytext = (mwig40DY.index[-1]+ datetime.timedelta(weeks=43), mwig40DY['SWIG80_DY'][-1]),bbox=bbox_props,color='white')
		ax4.annotate(str(mwig40PB['SWIG80_PB'][-1]), (mwig40PB.index[-1], mwig40PB['SWIG80_PB'][-1]), xytext = (mwig40PB.index[-1]+ datetime.timedelta(weeks=35), mwig40PB['SWIG80_PB'][-1]),bbox=bbox_props,color='white')
		ax5.annotate(str(mwig40MV['SWIG80_MV'][-1]), (mwig40MV.index[-1], mwig40MV['SWIG80_MV'][-1]), xytext = (mwig40MV.index[-1]+ datetime.timedelta(weeks=43), mwig40MV['SWIG80_MV'][-1]),bbox=bbox_props,color='white')

		ax1.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		ax5.legend()

		plt.tight_layout()
		plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
		plt.show()

	elif index == 'WIG20':

		mwig40 = DownloadFromStooq('WIG20','d')
		mwig40TR = DownloadFromStooq('WIG20TR','d')
		mwig40PE = DownloadFromStooq('WIG20_PE','d')
		mwig40PB = DownloadFromStooq('WIG20_PB','d')
		mwig40DY = DownloadFromStooq('WIG20_DY','d')
		mwig40MV = DownloadFromStooq('WIG20_MV','d')

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
		ax3.set_title('Dividend Yield')
		ax4.set_title('P/BV')
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
		ax3.plot(mwig40DY.index.values,mwig40DY_mean, linestyle='--', label='average')
		ax4.plot(mwig40PB.index.values,mwig40PB_mean, linestyle='--', label='average')
		ax5.plot(mwig40MV.index.values,mwig40MV_mean, linestyle='--', label='average')

		bbox_props = dict(boxstyle='larrow')
		bbox_props_2 = dict(boxstyle='larrow', color='orange')
		ax1.annotate(str(mwig40['WIG20'][-1]), (mwig40.index[-1], mwig40['WIG20'][-1]), xytext = (mwig40.index[-1]+ datetime.timedelta(weeks=65), mwig40['WIG20'][-1]),bbox=bbox_props,color='white')
		ax1.annotate(str(mwig40TR['WIG20TR'][-1]), (mwig40TR.index[-1], mwig40TR['WIG20TR'][-1]), xytext = (mwig40TR.index[-1]+ datetime.timedelta(weeks=65), mwig40TR['WIG20TR'][-1]),bbox=bbox_props_2,color='black')
		ax2.annotate(str(mwig40PE['WIG20_PE'][-1]), (mwig40PE.index[-1], mwig40PE['WIG20_PE'][-1]), xytext = (mwig40PE.index[-1]+ datetime.timedelta(weeks=35), mwig40PE['WIG20_PE'][-1]),bbox=bbox_props,color='white')
		ax3.annotate(str(mwig40DY['WIG20_DY'][-1]), (mwig40DY.index[-1], mwig40DY['WIG20_DY'][-1]), xytext = (mwig40DY.index[-1]+ datetime.timedelta(weeks=43), mwig40DY['WIG20_DY'][-1]),bbox=bbox_props,color='white')
		ax4.annotate(str(mwig40PB['WIG20_PB'][-1]), (mwig40PB.index[-1], mwig40PB['WIG20_PB'][-1]), xytext = (mwig40PB.index[-1]+ datetime.timedelta(weeks=35), mwig40PB['WIG20_PB'][-1]),bbox=bbox_props,color='white')
		ax5.annotate(str(mwig40MV['WIG20_MV'][-1]), (mwig40MV.index[-1], mwig40MV['WIG20_MV'][-1]), xytext = (mwig40MV.index[-1]+ datetime.timedelta(weeks=43), mwig40MV['WIG20_MV'][-1]),bbox=bbox_props,color='white')

		ax1.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		ax5.legend()

		plt.tight_layout()
		plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
		plt.show()
		
	elif index == 'NC':
	
		wig = DownloadFromStooq('NCINDEX','d')
		wigPE = DownloadFromStooq('NCINDEX_PE','d')
		wigPB = DownloadFromStooq('NCINDEX_PB','d')
		wigDY = DownloadFromStooq('NCINDEX_DY','d')
		wigMV = DownloadFromStooq('NCINDEX_MV','d')

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
		ax3.set_title('Dividend Yield')
		ax4.set_title('P/BV')
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
		ax1.annotate(str(wig['NCINDEX'][-1]), (wig.index[-1], wig['NCINDEX'][-1]), xytext = (wig.index[-1]+ datetime.timedelta(weeks=35), wig['NCINDEX'][-1]),bbox=bbox_props,color='white')
		ax2.annotate(str(wigPE['NCINDEX_PE'][-1]), (wigPE.index[-1], wigPE['NCINDEX_PE'][-1]), xytext = (wigPE.index[-1]+ datetime.timedelta(weeks=35), wigPE['NCINDEX_PE'][-1]),bbox=bbox_props,color='white')
		ax3.annotate(str(wigPB['NCINDEX_PB'][-1]), (wigPB.index[-1], wigPB['NCINDEX_PB'][-1]), xytext = (wigPB.index[-1]+ datetime.timedelta(weeks=43), wigPB['NCINDEX_PB'][-1]),bbox=bbox_props,color='white')
		ax4.annotate(str(wigDY['NCINDEX_DY'][-1]), (wigDY.index[-1], wigDY['NCINDEX_DY'][-1]), xytext = (wigDY.index[-1]+ datetime.timedelta(weeks=35), wigDY['NCINDEX_DY'][-1]),bbox=bbox_props,color='white')
		ax5.annotate(str(wigMV['NCINDEX_MV'][-1]), (wigMV.index[-1], wigMV['NCINDEX_MV'][-1]), xytext = (wigMV.index[-1]+ datetime.timedelta(weeks=43), wigMV['NCINDEX_MV'][-1]),bbox=bbox_props,color='white')

		ax1.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		ax5.legend()

		plt.tight_layout()
		plt.annotate('by @SzymonNowak1do1', (0,0), (350,-30), xycoords='axes fraction', textcoords='offset points', va='top')
		plt.show()	
		
		
def FindExtremes(df, column, first_rolling=360, second_rolling=180):

	import matplotlib.dates as mdates
	
	df['mean'] = df[column].rolling(window=first_rolling).mean()
	df['mean_2'] = df[column].rolling(window=second_rolling).mean()
	
	if df.loc[first_rolling+second_rolling,'mean'] > df.loc[first_rolling+second_rolling,column]:
		mov_direction = 0
		mov_direction_fixed = 0 
	else:
		mov_direction = 1
		mov_direction_fixed = 1    
	
	dates = []
	
	for i in range(1,len(df)):
		if (mov_direction == 1) & (df.loc[i,'mean'] > df.loc[i,'mean_2']):
			dates.append(i-1)
			mov_direction = 0
		if (mov_direction == 0) & (df.loc[i,'mean'] < df.loc[i,'mean_2']):
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
			highs.append(df[mov_l[i]:mov_h[i]][column].idxmax())
			   
		for i in range(len(mov_l)):
			try:
				lows.append(df[mov_h[i]:mov_l[i+1]][column].idxmin())
			except:
				continue		
	else:
		for i in range(len(mov_l)):
			highs.append(df[mov_h[i]:mov_l[i]][column].idxmin())
			   
		for i in range(len(mov_h)):
			try:
				lows.append(df[mov_l[i]:mov_l[h+1]][column].idxmax())
			except:
				continue			
				
				
	extremes = []
	extremes = lows + highs		
	
	low = pd.DataFrame()
	high = pd.DataFrame()

	for i in highs:
		high.loc[i,'High'] = df.loc[i,'Date']
	high.reset_index(inplace=True,drop=True)
		
	for i in lows:    
		low.loc[i,'Low'] = df.loc[i,'Date']
	low.reset_index(inplace=True,drop=True)    

	final = pd.concat([high,low],axis=1)

	df.set_index('Date',inplace=True)
	
	myFmt = mdates.DateFormatter('%Y-%m')
	years = mdates.YearLocator()   
	months = mdates.MonthLocator()
	
	ax = df[column].plot(figsize=(15,12), markevery=extremes,style='s-')
	
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(myFmt)
	ax.xaxis.set_minor_locator(months)
	
	ax2 = df[[column,'mean','mean_2']].plot(figsize=(15,12))
	
	ax2.xaxis.set_major_locator(years)
	ax2.xaxis.set_major_formatter(myFmt)
	ax2.xaxis.set_minor_locator(months)
	
	return(final)
	
def ProjectHighsLows(df, column, df_dates, Low=True, High=True):

	from datetime import timedelta
	import numpy as np

	high_delta = []
	low_delta = []
	
	for i in range(1,len(df_dates)):
		high_delta.append((df_dates.loc[i,'High']-df_dates.loc[i-1,'High']).days)
		low_delta.append((df_dates.loc[i,'Low']-df_dates.loc[i-1,'Low']).days)
	
	if str(high_delta[-1]) == 'nan':
		high_index = -2
	else:
		high_index = -1
		
	if str(low_delta[-1]) == 'nan':
		low_index = -2
	else:
		low_index = -1   
	
	low_delta = [x for x in low_delta if str(x) != 'nan']    
	high_delta = [x for x in high_delta if str(x) != 'nan'] 
	
	forecasted_low = df_dates.iloc[low_index,1] + timedelta(np.average(low_delta))
	forecasted_high = df_dates.iloc[high_index,0] + timedelta(np.average(high_delta))
	
	if Low == True:
		plt.axvline(forecasted_low, color='r', linestyle='-', lw=2)
		plt.axvline((forecasted_low + timedelta(np.std(low_delta))), color='r', linestyle='--', lw=1)
		plt.axvline((forecasted_low - timedelta(np.std(low_delta))), color='r', linestyle='--', lw=1)

	if High == True:
		plt.axvline(forecasted_high, color='g', linestyle='-', lw=2)
		plt.axvline((forecasted_high + timedelta(np.std(high_delta))), color='g', linestyle='--', lw=1)
		plt.axvline((forecasted_high - timedelta(np.std(high_delta))), color='g', linestyle='--', lw=1)

	ax = df[column].plot(figsize=(15,12))
	
	col_names = ['High','Low']
	final = pd.DataFrame(columns=col_names)
	final.loc[0,'High'] = forecasted_high
	final.loc[0,'Low'] = forecasted_low
	
	return(final)

	

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
    