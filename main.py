import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from GRN import Cell
import numpy as np
import math
import re

import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

gene = np.random.randint(2, size=(71,32))
p_pro = np.array([[int(x) for x in list('01010101010101010101010111111111')] for i in range(4)])
tf_pro = np.array([[int(x) for x in list('01010101010101010101010100000000')] for i in range(5)])

gene = np.concatenate((gene, p_pro, tf_pro))


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("permutation", np.random.permutation, gene)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df['mAvg(10)'] = df['Adj Close'].rolling(window=10).mean()
df['mAvg(10)'] = (df['mAvg(10)'] - df['mAvg(10)'].min()) / (df['mAvg(10)'].max() - df['mAvg(10)'].min())
df['mChange(5)'] = df['Adj Close'].pct_change(periods=5) 
df['mChange(5)'] = (df['mChange(5)'] - df['mChange(5)'].min()) / (df['mChange(5)'].max() - df['mChange(5)'].min())
df['mChange(10)'] = df['Adj Close'].pct_change(periods=10)
df['mChange(10)'] = (df['mChange(10)'] - df['mChange(10)'].min()) / (df['mChange(10)'].max() - df['mChange(10)'].min()) 
df['sOsc(10)'] = ((df['Adj Close'] - df['Low'].rolling(window=10).min()) / (df['High'].rolling(window=10).max() - df['Low'].rolling(window=10).min())) 
df.dropna(inplace=True)
print(df.head(11))

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    ind1 = ind1.flatten()
    ind2 = ind2.flatten()
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1.reshape(80,32), ind2.reshape(80,32)

 def mut(ind):
 	ind = ind.flatten()
 	


"""
ax1 = plt.subplot2grid((7,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((7,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax3 = plt.subplot2grid((7,1), (6,0), rowspan=1, colspan=1, sharex=ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['mAvg(10)'])
ax2.plot(df.index, df['mChange(5)'])
ax2.plot(df.index, df['mChange(10)'])
ax3.plot(df.index, df['sOsc(10)'])
plt.show()
"""

ip = [[int(x) for x in '00000000000000000000000000000000'],
		[int(x) for x in'00000000000000001111111111111111'],
		[int(x) for x in'00000000111111110000000011111111'],
		[int(x) for x in'00000000111111111111111100000000']]

buy = [int(x) for x in '00000000000000001111111111111111']
sell = [int(x) for x in '00000000111111110000000011111111']
null = [int(x) for x in '00000000111111111111111100000000']

def evalOneMax(individual):
	c = Cell(i_proteins=ip, genome = individual)
	if c.invalid:
		return -1,
	funds = 100000
	positions = {}
	for idx, row in df[0:50].iterrows():
		i_concentrations = [0.1 * row['mAvg(10)'], # 00000000000000000000000000000000
							  0.1 * row['sOsc(10)'], # 00000000000000001111111111111111
							  0.1 * row['mChange(5)'], # 00000000111111110000000011111111
							  0.1 * row['mChange(10)']] # 00000000111111111111111100000000
		
		c.inject(i_concentrations)
		for i in range(2000):
			c.step()
			#tf_conc.append(np.copy(c.tf_concentrations))
			#p_conc.append(np.copy(c.p_concentrations))
		buy_signal = 0
		sell_signal = 0
		null_signal = 0
		for idx, prt in enumerate(c.p_proteins):
			buy_signal += c.p_concentrations[idx] * np.exp(np.count_nonzero(buy^prt)-32)
			sell_signal += c.p_concentrations[idx] * np.exp(np.count_nonzero(sell^prt)-32)
			null_signal += c.p_concentrations[idx] * np.exp(np.count_nonzero(null^prt)-32)
		print(buy_signal, sell_signal, null_signal)
		#print(positions)
		#print(row.name-pd.Timedelta(days=10))
		if positions:
			if row.name-pd.Timedelta(days=10) >= list(positions)[0]:
				
				if positions[list(positions)[0]] < 0:
					funds += (positions[list(positions)[0]] * row['Adj Close']) * 1.005
				elif positions[list(positions)[0]] > 0:
					funds += (positions[list(positions)[0]] * row['Adj Close']) * 0.995
				del positions[list(positions)[0]]
				print('closed position')

		if row.name < df.iloc[40].name:
			if buy_signal > sell_signal:
				if buy_signal > null_signal:
					s = math.floor((0.1*funds)/row['Adj Close'])
					if s >= 1:
						funds -= (s*row['Adj Close'])*1.005
						positions[row.name] = s
						print('bought {} shares'.format(s))
				else:
					print('do nothing')
			else:
				if sell_signal > null_signal:
					s = math.floor((0.1*funds)/row['Adj Close'])
					if s >= 1:
						funds += (s*row['Adj Close'])*0.995
						positions[row.name] = -s
						print('sold {} shares'.format(s))
				else:
					print('do nothing')
	return funds/100000,


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=9)
hof = tools.HallOfFame(1, similar=np.array_equal)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats,
                        halloffame=hof)

print(hof[0])
with open("best.txt", "wt") as out_file:
	out_file.write(hof)



"""
fig, ax = plt.subplots()
ax.plot(p_conc)
plt.ylim(-0.1,1.1)
ax.legend(('prt0', 'prt1', 'prt2', 'prt3','prt4','prt5','prt6','prt7','prt8','prt9'))
plt.show()

fig, ax = plt.subplots()
ax.plot(tf_conc)
plt.ylim(-0.1,1.1)
ax.legend(('tf0', 'tf1', 'tf2', 'tf3', 'tf4', 'tf5', 'tf6', 'tf7', 'tf8', 'tf9'))
plt.show()

fig, ax = plt.subplots()
ax.plot(funds_history)
#plt.ylim(-0.1,1000000)
plt.show()
"""