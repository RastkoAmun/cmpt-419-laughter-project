from torch.utils.data import Dataset
from helpers.helpers import get_full_path
from helpers.helpers import extract_mfcc

class LaughterDataset(Dataset):
	def __init__(self, root_dir, labels):
		self.root_dir = root_dir
		self.labels = labels		

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		audio_id, category, numerical = self.labels.iloc[index]
		path = get_full_path(self.root_dir, audio_id)
		audio = extract_mfcc(path)

		return audio, audio_id, category, numerical