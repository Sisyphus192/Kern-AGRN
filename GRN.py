import numpy as np
import matplotlib.pyplot as plt
import random
import re

VERBOSE = False

class Cell:
	def __init__(self, i_proteins = None, genome = None):
		# scaling factors
		self.beta = 1
		self.delta = 1
		self.i_proteins = i_proteins
		self.invalid = False
		if genome is None:
			self.initRandomGenome()
		else:
			self.readGenome(genome.flatten())
		if not self.invalid:
			self.get_proteins()
			self.get_u_values() # precompute binding values

			# initialize cell state
			self.tf_concentrations = np.asarray([1 / len(self.tf_proteins)] * len(self.tf_proteins))
			self.p_concentrations = np.asarray([1 / len(self.p_proteins)] * len(self.p_proteins))
			self.i_concentrations = []
			self.injected = False
			for i in range(10000): # run until steady state
				self.step()
			print("steady state achieved")

	def initRandomGenome(self):
		# initialize genome 
		self.genome = []

		# add random number of transcription factors
		for i in range(np.random.randint(1,10)):
			g = Gene()
			g.promoter_site[-8:] = 0
			self.genome.append(g) 

		# add random number of production proteins
		for i in range(np.random.randint(1,10)):
			g = Gene()
			g.promoter_site[-8:] = 1
			self.genome.append(g) 
		
		random.shuffle(self.genome) # for test purposes

	def readGenome(self, raw_genome):
		self.genome = []
		promoter_sites = [m.start() for m in re.finditer('(01010101010101010101010111111111|01010101010101010101010100000000)', ''.join([str(x) for x in raw_genome]))]
		if promoter_sites:
			print("promoters sites found")
			print(promoter_sites)
			raw_genome = np.concatenate((raw_genome, raw_genome, raw_genome)) # padding for wrap arround so dont have to deal with string slicing wrap arounds
			for i in promoter_sites:
				x = int(i + len(raw_genome)/3)
				self.genome.append(Gene(enhancer_site=raw_genome[x-64:x-32], inhibitor_site=raw_genome[x-32:x], promoter_site=raw_genome[x:x+32], exon=raw_genome[x+32:x+192]))

		else:
			self.invalid = True





	def get_proteins(self):
		self.tf_proteins = []
		self.p_proteins = []

		for gene in self.genome:
			if not any(gene.promoter_site[-8:]):
				self.tf_proteins.append(gene.express_protein())
			else:
				self.p_proteins.append(gene.express_protein())

		self.N = len(self.tf_proteins)

	def get_u_values(self):
		'''precomputes the binding values of regulatory sites and proteins, these do not change'''
		self.u_max = []
		self.u_values = {}
		for gene in self.genome:
			self.u_values[gene] = {'enh':[],'inh':[]}
			for tf in self.tf_proteins: # only tf regulate so only care about how tf binds to regulatory sites 
				if VERBOSE:
					print("Calculating binding values:")
					print(gene.enhancer_site)
					print(tf)
					print(gene.enhancer_site^tf, np.count_nonzero(gene.enhancer_site^tf), '\n')
					print(gene.inhibitor_site)
					print(tf)
					print(gene.inhibitor_site^tf, np.count_nonzero(gene.inhibitor_site^tf), '\n')

				self.u_values[gene]['enh'].append(np.count_nonzero(gene.enhancer_site^tf))
				self.u_values[gene]['inh'].append(np.count_nonzero(gene.inhibitor_site^tf))

			for ip in self.i_proteins:
				self.u_values[gene]['enh'].append(np.count_nonzero(gene.enhancer_site^np.asarray(ip)))
				self.u_values[gene]['inh'].append(np.count_nonzero(gene.inhibitor_site^np.asarray(ip)))
			self.u_max.append(np.max(self.u_values[gene]['enh']))
			self.u_max.append(np.max(self.u_values[gene]['inh']))

		self.u_max = np.max(self.u_max)

	def step(self):
		e_tf = []
		h_tf = []
		e_p = []
		h_p = []


		for gene in self.genome:
			if not any(gene.promoter_site[-8:]):
				e_tf.append(np.sum(np.concatenate((self.tf_concentrations, self.i_concentrations), axis=None) * np.exp(self.beta * (self.u_values[gene]['enh'][:self.N]-self.u_max))) / self.N)
				h_tf.append(np.sum(np.concatenate((self.tf_concentrations, self.i_concentrations), axis=None) * np.exp(self.beta * (self.u_values[gene]['inh'][:self.N]-self.u_max))) / self.N)
			else:
				e_p.append(np.sum(np.concatenate((self.tf_concentrations, self.i_concentrations), axis=None) * np.exp(self.beta * (self.u_values[gene]['enh'][:self.N]-self.u_max))) / self.N)
				h_p.append(np.sum(np.concatenate((self.tf_concentrations, self.i_concentrations), axis=None) * np.exp(self.beta * (self.u_values[gene]['inh'][:self.N]-self.u_max))) / self.N)

		
		#if self.injected:
		#	print(self.tf_concentrations)
		#	print(e_tf)
		#	print(h_tf)

		dtfc_dt = self.delta * np.subtract(e_tf, h_tf) * self.tf_concentrations
		dpc_dt = self.delta * np.subtract(e_p, h_p) 

		#print(dtfc_dt)
		#print(dpc_dt)

		self.tf_concentrations += dtfc_dt
		self.p_concentrations += dpc_dt

		

		self.tf_concentrations[self.tf_concentrations < 0] = 0
		self.p_concentrations[self.p_concentrations < 0] = 0

		self.tf_concentrations /= np.sum(self.tf_concentrations)
		self.p_concentrations /= np.sum(self.p_concentrations)

		#print(self.tf_concentrations)

	def inject(self, i_concentrations):
		
		#self.tf_concentrations = np.copy(self.steady_state[0])
		self.i_concentrations = i_concentrations

		self.tf_concentrations /= (-1 / (np.sum(i_concentrations) - 1))			#(np.sum(self.tf_concentrations) + np.sum(i_concentrations))
		#print(np.sum(self.tf_concentrations))
		#print(np.sum(self.tf_concentrations) + np.sum(self.i_concentrations))
		self.N = len(self.tf_concentrations) + len(self.i_concentrations)
		self.injected = True

	def save_steady_state(self):
		self.steady_state = (np.copy(self.tf_concentrations), np.copy(self.N))
		




class Gene:
	def __init__(self, enhancer_site=None, inhibitor_site = None, promoter_site = None, exon = None):
		if enhancer_site is None:
			self.enhancer_site = np.random.randint(2, size=32)
		else:
			self.enhancer_site = enhancer_site
		
		if inhibitor_site is None:
			self.inhibitor_site = np.random.randint(2, size=32)
		else:
			self.inhibitor_site = inhibitor_site
		
		if promoter_site is None:
			self.promoter_site = np.random.randint(2, size=32)
		else:
			self.promoter_site = promoter_site
		
		if exon is None:
			self.exon = np.random.randint(2, size=(5,32))
		else:
			self.exon = exon.reshape(5,32)

	'''
	def __init__(self, gene_sequence):
		if isinstance(gene_sequence, str):
			self.enhancer_site = np.asarray(list(gene_sequence[:32]))
			self.inhibitor_site = np.asarray(list(gene_sequence[32:64]))
			self.promoter_site = np.asarray(list(gene_sequence[64:96]))
			self.exon = np.asarray(list(gene_sequence[96:256])).reshape((5,32))
		else:
			self.enhancer_site = gene_sequence[:32]
			self.inhibitor_site = gene_sequence[32:64]
			self.promoter_site = gene_sequence[64:96]
			self.exon = gene_sequence[96:256].reshape((5,32))

		print(self.enhancer_site)
		print(self.inhibitor_site)
		print(self.promoter_site)
		print(self.exon)
	'''

	def express_protein(self):
		u, indices = np.unique(self.exon, return_inverse=True)
		return u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(self.exon.shape),
                                None, np.max(indices) + 1), axis=0)]



if __name__ == "__main__":

	tf_conc = []
	p_conc = []
	enh = []
	inh = []

	#genome = "1015101000101110100101101111101011051110000010111010100010101111011511010100010010001000010111111115111010010110100001000010011100101010010100011110101010111101010111111101101010010010100010100010101101010010111010001010110010001010101010010110100101010001010010101001000000011100111010"

	c = Cell()

	tf_conc.append(np.copy(c.tf_concentrations))
	p_conc.append(np.copy(c.p_concentrations))
	#enh.append(np.copy(c.enhancing_signals))
	#inh.append(np.copy(c.inhibiting_signals))

	for i in range(100000):
		c.step()
		tf_conc.append(np.copy(c.tf_concentrations))
		p_conc.append(np.copy(c.p_concentrations))

		#enh.append(np.copy(c.enhancing_signals))
		#inh.append(np.copy(c.inhibiting_signals))

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

	#fig, ax = plt.subplots()
	#ax.plot(enh)
	#plt.show()

	#fig, ax = plt.subplots()
	#ax.plot(inh)
	#plt.show()



