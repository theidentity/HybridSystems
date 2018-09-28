import pandas as pd


def load_orig_dataset(dataset):
	if dataset == 'train':
		df = pd.read_csv('data/orig_data/trialPromoResults.csv')
	elif dataset == 'test':
		df = pd.read_csv('data/orig_data/custdatabase.csv')
		df = df.iloc[:,:-1]
	elif dataset == 'actual':
		df = pd.read_csv('data/orig_data/Cust_Actual.csv')
	else:
		print('dataset not available')
		return None
	return df



if __name__ == '__main__':
	df = load_orig_dataset('train')
	print(df.columns)
	print(df.shape)

	df = load_orig_dataset('test')
	print(df.columns)
	print(df.shape)
	
	df = load_orig_dataset('actual')
	print(df.columns)
	print(df.shape)
	