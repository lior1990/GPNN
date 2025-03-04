import numpy as np
import torch
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
from torch.nn.functional import fold, unfold
from .utils import *
import cv2


class GPNN_w_Segmentation:
	def __init__(self, config):
		# general settings
		self.T = config['iters']
		self.PATCH_SIZE = (config['patch_size'], config['patch_size'])
		self.COARSE_DIM = (config['coarse_dim'], config['coarse_dim'])
		self.STRIDE = (config['stride'], config['stride'])
		self.LABEL_PATCH_SIZE = (config['label_patch_size'], config['label_patch_size'])
		self.R = config['pyramid_ratio']
		self.ALPHA = config['alpha']

		# cuda init
		global device
		if config['no_cuda']:
			device = torch.device('cpu')
		else:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			if torch.cuda.is_available():
				print('cuda initialized!')

		# input image
		img_path = config['input_img']
		label_path = config['input_img'].replace(".jpg", ".png")
		self.input_img = img_read(img_path)
		self.input_label = img_read(label_path, is_label=True)
		self.unique_labels = np.unique(self.input_label)
		self.input_label = self.input_label[:, :, np.newaxis]
		assert self.input_img.shape[:2] == self.input_label.shape[:2]

		if config['out_size'] != 0:
			if self.input_img.shape[0] > config['out_size']:
				self.input_img = rescale(self.input_img, config['out_size'] / self.input_img.shape[0], multichannel=True)
				self.input_label = cv2.resize(self.input_label, (config['out_size'], config['out_size']), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]

		# pyramids
		pyramid_depth = np.log(min(self.input_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
		self.add_base_level = True if np.ceil(pyramid_depth) > pyramid_depth else False
		pyramid_depth = int(np.ceil(pyramid_depth))
		self.x_pyramid = list(
			tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))

		self.y_pyramid = [0] * (pyramid_depth + 1)

		# out_file
		filename = os.path.splitext(os.path.basename(img_path))[0]
		self.out_folder = os.path.join(config['out_dir'], config['task'], filename)

		# coarse settings
		noise = np.random.normal(0, config['sigma'], self.x_pyramid[-1].shape[:2])[..., np.newaxis]
		self.coarse_img = self.x_pyramid[-1] + noise

	def run(self, sample_id, to_save=True):
		print(f"Working on sample: {sample_id}")
		for i in reversed(range(len(self.x_pyramid))):
			print(f"Working on scale: {i}")
			if i == len(self.x_pyramid) - 1:
				queries = self.coarse_img
				keys = self.x_pyramid[i]
			else:
				queries = resize(self.y_pyramid[i + 1], self.x_pyramid[i].shape)
				keys = resize(self.x_pyramid[i + 1], self.x_pyramid[i].shape)

			for j in range(self.T):
				self.y_pyramid[i] = self.PNN(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE, self.ALPHA, gpu=i>0)
				queries = self.y_pyramid[i]
				keys = self.x_pyramid[i]

		if to_save:
			print(f"Saving sample {sample_id}")
			img_save(self.y_pyramid[0], os.path.join(self.out_folder, f"{sample_id}.png"))
		else:
			return self.y_pyramid[0]

		# free some memory
		del self.x_pyramid
		generated_img = self.y_pyramid[0]
		del self.y_pyramid
		patch_size = self.LABEL_PATCH_SIZE
		stride = self.LABEL_PATCH_SIZE
		queries = extract_patches(self.input_img.astype(float), patch_size, stride)
		keys = extract_patches(generated_img, patch_size, stride)
		labels = extract_patches(self.input_label.astype(float), patch_size, stride, channels=1)
		dist = compute_distances_l1(queries, keys)
		# norm_dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
		NNs = torch.argmin(dist, dim=1)  # find_NNs
		labels = labels[NNs]
		new_label = combine_patches(labels, patch_size, stride, self.input_label.shape, channels=1)
		new_label = new_label.astype(np.uint8)
		img_save(new_label, os.path.join(self.out_folder, f"{sample_id}_label.png"))
		label_save(new_label, os.path.join(self.out_folder, f"{sample_id}_label_rgb.png"))
		img_save(self.input_label, os.path.join(self.out_folder, f"orig_label.png"))
		label_save(self.input_label, os.path.join(self.out_folder, f"orig_label_rgb.png"))
		new_unique_labels = np.unique(new_label)
		assert set(new_unique_labels).issubset(self.unique_labels), f"labels: {self.unique_labels} vs new unique labels: {new_unique_labels}"

	def PNN(self, x, x_scaled, y_scaled, patch_size, stride, alpha, gpu=True):
		queries = extract_patches(y_scaled, patch_size, stride)
		keys = extract_patches(x_scaled, patch_size, stride)
		values = extract_patches(x, patch_size, stride)
		dist = compute_distances(queries, keys, gpu=gpu)
		dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
		NNs = torch.argmin(dist, dim=1)  # find_NNs
		values = values[NNs]
		y = combine_patches(values, patch_size, stride, x_scaled.shape)
		return y


def extract_patches(src_img, patch_size, stride, channels=3):
	img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
	return torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
		.squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])


def compute_distances(queries, keys, gpu=True):
	if gpu:
		dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float16, device=device)
		for i in range(len(queries)):
			dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3))
	else:
		dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float32)
		for i in range(len(queries)):
			dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3)).cpu()

	return dist_mat


def compute_distances_l1(queries, keys):
	dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float32)
	for i in range(len(queries)):
		dist_mat[i] = torch.mean((queries[i] - keys), dim=(1, 2, 3)).cpu()

	return dist_mat


def combine_patches(O, patch_size, stride, img_shape, channels=3):
	O = O.permute(1, 0, 2, 3).unsqueeze(0)
	patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
		.permute(0, 1, 3, 2) \
		.contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
	combined = fold(patches, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

	# normal fold matrix
	input_ones = torch.ones((1, img_shape[2], img_shape[0], img_shape[1]), dtype=O.dtype, device=device)
	divisor = unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
	divisor = fold(divisor, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

	divisor[divisor == 0] = 1.0
	return (combined / divisor).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
