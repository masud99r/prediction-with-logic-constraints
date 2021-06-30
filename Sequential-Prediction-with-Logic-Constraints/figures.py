import matplotlib.pyplot as plt
import numpy as np
# resource for formating plot: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.plot.html

def accuracy_comparison():
	data_dir = "../logic_constraints/results"
	
	ROBOT = 'Taurus'
	rfile = data_dir+'/mlayer_True_'+ROBOT+'_results_test_1405.txt'
	rfile2 = data_dir+'/mlayer_False_'+ROBOT+'_results_test_1405.txt'
	
	# ROBOT = 'Gym'
	# rfile = data_dir+'/mlayer_True_'+ROBOT+'_results_test_126.txt'
	# rfile2 = data_dir+'/mlayer_False_'+ROBOT+'_results_test_126.txt'
	
	end_epoch=20
	mulifactor = 100
	# PLOT
	series0 = []
	series_lstm = []
	series_mln = []
	series_lstm_mln = []
	series4 = []
	series5 = []
	series6 = []
	linecount = 0
	with open(rfile) as f:
		next(f) # header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series0.append(int(row[2])+1)
			# series_lstm_mln.append(float(row[4])*mulifactor)
			# series_mln.append(float(row[5])*mulifactor)
			series_lstm_mln.append(float(row[6])*mulifactor)
	linecount = 0
	with open(rfile2) as f:
		next(f) # header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			# series0.append(int(row[2])+1)
			series_lstm.append(float(row[4])*mulifactor)
			series_mln.append(float(row[5])*mulifactor)
			# series_lstm_mln.append(float(row[6])*mulifactor)

	plt.plot(series0, series_lstm)
	plt.plot(series0, series_mln)
	plt.plot(series0, series_lstm_mln)
	print('ROBOT', ROBOT)
	print('Model BestAccuracy BestEpoch')
	print('LSTM'+'\t'+str(max(series_lstm))+'\t'+str(series_lstm.index(max(series_lstm))+1))
	print('MLN'+'\t'+str(max(series_mln))+'\t'+str(series_mln.index(max(series_mln))+1))
	print('LSTM+MLN'+'\t'+str(max(series_lstm_mln))+'\t'+str(series_lstm_mln.index(max(series_lstm_mln))+1))
	# plt.plot(series0, series4)
	# plt.plot(series0, series5)
	# plt.plot(series0, series6)

	legends = ['LSTM', 'MLN', 'LSTM+MLN']
	plt.legend(legends, fontsize = 12, loc='lower right')
	plt.ylim(0, 100)
	plt.xlim(1, 21)
	plt.xticks(np.arange(0, end_epoch+1, 5))
	plt.ylabel('Testing Accuracy (%)', fontsize = 14)
	plt.xlabel('LSTM Training Epoch', fontsize = 14)
	# plt.title(title, fontsize = 14)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.grid()

	filename = rfile.split("/")[-1]
	filename = filename.replace(".txt", "")
	plt.savefig('./results/'+ROBOT+'.png')
	print(series_lstm)
	print(series_mln)
	print(series_lstm_mln)
def accuracy_comparison_permethod(mtype="PiorLayer"):
	data_dir = "../logic_constraints/results"
	
	ROBOT = 'Taurus'
	rfile = data_dir+'/mlayer_True_'+ROBOT+'_results_test_1405.txt'
	rfile2 = data_dir+'/mlayer_False_'+ROBOT+'_results_test_1405.txt'
	
	# ROBOT = 'Gym'
	# rfile = data_dir+'/mlayer_True_'+ROBOT+'_results_test_126.txt'
	# rfile2 = data_dir+'/mlayer_False_'+ROBOT+'_results_test_126.txt'
	
	end_epoch=20
	mulifactor = 100
	# PLOT
	series0 = []
	series_lstm = []
	series_mln = []
	series_lstm_mln = []
	series4 = []
	series5 = []
	series6 = []
	linecount = 0
	with open(rfile) as f:
		next(f) # header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series0.append(int(row[2])+1)
			if mtype == "PiorLayer":
				series_lstm_mln.append(float(row[4])*mulifactor)
				series_mln.append(float(row[5])*mulifactor)
			
	linecount = 0
	with open(rfile2) as f:
		next(f) # header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			if mtype == "PiorLayer":
				series_lstm.append(float(row[4])*mulifactor)
				# series_mln.append(float(row[5])*mulifactor)
				# series_lstm_mln.append(float(row[6])*mulifactor)
			else:
				series_lstm.append(float(row[4])*mulifactor)
				series_mln.append(float(row[5])*mulifactor)
				series_lstm_mln.append(float(row[6])*mulifactor)

	plt.plot(series0, series_lstm)
	plt.plot(series0, series_mln)
	plt.plot(series0, series_lstm_mln)
	print('Model BestAccuracy BestEpoch')
	print('LSTM'+'\t'+str(max(series_lstm))+'\t'+str(series_lstm.index(max(series_lstm))+1))
	print('MLN'+'\t'+str(max(series_mln))+'\t'+str(series_mln.index(max(series_mln))+1))
	print('LSTM+MLN'+'\t'+str(max(series_lstm_mln))+'\t'+str(series_lstm_mln.index(max(series_lstm_mln))+1))
	# plt.plot(series0, series4)
	# plt.plot(series0, series5)
	# plt.plot(series0, series6)

	legends = ['LSTM', 'MLN', 'LSTM+'+mtype]
	plt.legend(legends, fontsize = 12, loc='lower right')
	plt.ylim(0, 100)
	plt.xlim(1, 21)
	plt.xticks(np.arange(0, end_epoch+1, 5))
	plt.ylabel('Testing Accuracy (%)', fontsize = 14)
	plt.xlabel('LSTM Training Epoch', fontsize = 14)
	# plt.title(title, fontsize = 14)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.grid()

	filename = rfile.split("/")[-1]
	filename = filename.replace(".txt", "")
	plt.savefig('./results/'+ROBOT+'_'+mtype+'.png')
	plt.cla()
	print(series_lstm)
	print(series_mln)
	print(series_lstm_mln)
def accuracy_comparison_perclass():
	data_dir = "../logic_constraints/results"
	
	# ROBOT = 'Taurus'
	# rfile = data_dir+'/mlayer_True_'+ROBOT+'_results_test_1405.txt'
	# rfile2 = data_dir+'/mlayer_False_'+ROBOT+'_results_test_1405.txt'
	
	ROBOT = 'Gym'
	rfile = data_dir+'/mlayer_True_'+ROBOT+'_results_test_126.txt'
	rfile2 = data_dir+'/mlayer_False_'+ROBOT+'_results_test_126.txt'
	
	end_epoch=20
	mulifactor = 100
	# PLOT
	series0 = []
	if ROBOT == 'Gym':
		series_lstm = [[], [], [], []]
		series_mln = [[], [], [], []]
		series_lstm_mln = [[], [], [], []]
		labelcount = 4
	else:
		series_lstm = [[], [], [], [], [], [], []]
		series_mln = [[], [], [], [], [], [], []]
		series_lstm_mln = [[], [], [], [], [], [], []]
		labelcount = 7

	print(series_lstm)
	linecount = 0
	with open(rfile) as f:
		next(f) # header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series0.append(int(row[2])+1)
			# series1.append(float(row[4])*mulifactor)
			# series_mln.append(float(row[5])*mulifactor)
			# series_lstm_mln.append(float(row[6])*mulifactor)
	linecount = 0
	with open(rfile2) as f:
		next(f) # header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			# series0.append(int(row[2])+1)
			series_list_lstm = eval(row[14])
			series_list_mln = eval(row[15])
			series_list_lstm_mln = eval(row[16])
			print(series_list_lstm_mln)
			for i in range(len(series_list_lstm)):
				series_lstm[i].append(float(series_list_lstm[i])*mulifactor)
				series_mln[i].append(float(series_list_mln[i])*mulifactor)
				series_lstm_mln[i].append(float(series_list_lstm_mln[i])*mulifactor)

			# series_mln.append(float(row[5])*mulifactor)
			# series_lstm_mln.append(float(row[6])*mulifactor)

	for i in range(labelcount):
		# print(series_lstm[i])
		plt.plot(series0, series_lstm[i])
		plt.plot(series0, series_mln[i])
		plt.plot(series0, series_lstm_mln[i])
		
		print('\nClass', i+1)
		print('Model BestAccuracy BestEpoch')
		print('LSTM'+'\t'+str(max(series_lstm[i]))+'\t'+str(series_lstm[i].index(max(series_lstm[i]))+1))
		print('MLN'+'\t'+str(max(series_mln[i]))+'\t'+str(series_mln[i].index(max(series_mln[i]))+1))
		print('LSTM+MLN'+'\t'+str(max(series_lstm_mln[i]))+'\t'+str(series_lstm_mln[i].index(max(series_lstm_mln[i]))+1))
	

		legends = ['LSTM', 'MLN', 'LSTM+MLN']
		plt.legend(legends, fontsize = 12, loc='lower right')
		plt.ylim(0, 101)
		plt.xlim(1, 10)
		# plt.ylabel('Testing Accuracy (%)', fontsize = 14)
		# plt.xlabel('LSTM Training Epoch', fontsize = 14)
		# plt.title(title, fontsize = 14)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		plt.tick_params(axis='both', which='major', labelsize=12)
		plt.grid()

		filename = rfile.split("/")[-1]
		filename = filename.replace(".txt", "")
		plt.savefig('./results/'+ROBOT+'_perclass_'+str(i)+'.png')
		plt.cla()
	# plt.plot(series0, series_lstm[3])
	# plt.plot(series0, series_mln[3])
	# plt.plot(series0, series_lstm_mln[3])

	# plt.plot(series0, series4)
	# plt.plot(series0, series5)
	# plt.plot(series0, series6)

	# legends = ['LSTM', 'MLN', 'LSTM+MLN']
	# plt.legend(legends, fontsize = 12, loc='upper left')
	
	# print(series_lstm)
	# print(series_mln)
	# print(series_lstm_mln)
def train_test_loss_comparison():
	thr = 19
	
	data_dir = "../logic_constraints/results"
	
	mulifactor = 100
	
	ROBOT = 'Taurus'
	end_epoch=20
	rfile_train = data_dir+'/mlayer_False_'+ROBOT+'_results_train_1405.txt'
	rfile_test = data_dir+'/mlayer_False_'+ROBOT+'_results_test_1405.txt'
	rfile_train2 = data_dir+'/mlayer_True_'+ROBOT+'_results_train_1405.txt'
	rfile_test2 = data_dir+'/mlayer_True_'+ROBOT+'_results_test_1405.txt'
	
	# ROBOT = 'Gym'
	# end_epoch=30
	# rfile_train = data_dir+'/mlayer_False_'+ROBOT+'_results_train_126.txt'
	# rfile_test = data_dir+'/mlayer_False_'+ROBOT+'_results_test_126.txt'
	# rfile_train2 = data_dir+'/mlayer_True_'+ROBOT+'_results_train_126.txt'
	# rfile_test2 = data_dir+'/mlayer_True_'+ROBOT+'_results_test_126.txt'
	

	filename = ROBOT+"_train_test_tradeoff"
	# filename = ROBOT+"_train_test_tradeoff_combined"

	
	# PLOT
	series0 = []
	series_lstm_train = []
	series_lstm_mln_train = []
	series_lstm_test = []
	series_lstm_mln_test = []
	
	linecount = 0
	with open(rfile_train) as f: # no PriorLayer, train
		next(f) # skip header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series0.append(int(row[2])+1)
			series_lstm_train.append(float(row[4])*mulifactor)
			# series_mln.append(float(row[5])*mulifactor)
			# series_lstm_mln.append(float(row[6])*mulifactor)
			
	
	linecount = 0
	with open(rfile_test) as f: # no PriorLayer, test
		next(f) # skip header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series_lstm_test.append(float(row[4])*mulifactor)
			# series_mln.append(float(row[5])*mulifactor)
			# series_lstm_mln.append(float(row[6])*mulifactor)
	
	linecount = 0
	with open(rfile_train2) as f: # with PriorLayer, train
		next(f) # skip header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series_lstm_mln_train.append(float(row[4])*mulifactor)
	
	linecount = 0
	with open(rfile_test2) as f: # with PriorLayer, test
		next(f) # skip header
		for line in f:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line = line.strip(" ")
			row = line.split("\t")
			linecount += 1
			if linecount > end_epoch:
				break
			series_lstm_mln_test.append(float(row[4])*mulifactor) # only priorlayer
			# series_lstm_mln_test.append(float(row[6])*mulifactor) # combined Conflation+PriorLayer


	
	# plt.plot(series0, series1)
	# fmt = '[marker][line][color]'
	# plot(x, y, 'go--', linewidth=2, markersize=12)
	legends = ['LSTM -> Train', 'LSTM -> Test', 'LSTM+PriorLayer -> Train', 'LSTM+PriorLayer -> Test']
	
	plt.plot(series0, series_lstm_train, color='SteelBlue', marker='.', linestyle='solid', linewidth=2, markersize=8) 
	plt.plot(series0, series_lstm_test, color='SteelBlue', linestyle='solid', linewidth=2, markersize=8)

	plt.plot(series0, series_lstm_mln_train, color='orange', marker='.', linestyle='solid', linewidth=2, markersize=8)
	plt.plot(series0, series_lstm_mln_test, color='orange', linestyle='solid', linewidth=2, markersize=8)
	# plt.plot(series0, series3)

	# # framewise
	# plt.plot(series0, series2)
	# plt.plot(series0, series4)

	plt.legend(legends, fontsize = 12, loc='lower right')
	plt.ylim(0, 110)
	plt.ylabel('Accuracy (%)', fontsize = 14)
	plt.xlabel('Epoch', fontsize = 14)
	# plt.title(title, fontsize = 14)
	plt.xlim(0, end_epoch+1) 
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.xticks(np.arange(0, end_epoch+2, 5))
	plt.grid()

	
	# filename = "taurus_prior_bias_train_test_tradeof"
	# filename = "taurus_prior_bias_train_test_tradeof_5perdata"
	# plt.savefig('./figures/'+filename+'_early_50per'+'.png')
	plt.savefig('./results/'+filename+'.png')
	plt.cla()
if __name__ == "__main__":
	accuracy_comparison_perclass()
	accuracy_comparison()
	accuracy_comparison_permethod(mtype="PiorLayer")
	accuracy_comparison_permethod(mtype="Conflation")
	train_test_loss_comparison()