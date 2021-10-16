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

		# faiss init
		self.is_faiss = config['faiss']
		if self.is_faiss:
			global faiss, res
			import faiss
			res = faiss.StandardGpuResources()
			print('faiss initialized!')

		# input image
		img_path = config['input_img']
		label_path = config['input_img'].replace(".jpg", ".png")
		self.input_img = img_read(img_path)
		self.input_label = img_read(label_path, is_label=True)
		self.input_label = self.input_label[:, :, np.newaxis]
		assert self.input_img.shape[:2] == self.input_label.shape[:2]

		if config['out_size'] != 0:
			if self.input_img.shape[0] > config['out_size']:
				self.input_img = rescale(self.input_img, config['out_size'] / self.input_img.shape[0], multichannel=True)
				self.input_label = rescale(self.input_label, config['out_size'] / self.input_label.shape[0], multichannel=False)

		# pyramids
		pyramid_depth = np.log(min(self.input_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
		self.add_base_level = True if np.ceil(pyramid_depth) > pyramid_depth else False
		pyramid_depth = int(np.ceil(pyramid_depth))
		self.x_pyramid = list(
			tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))

		self.y_pyramid = [0] * (pyramid_depth + 1)
		self.y_label_pyramid = [0] * (pyramid_depth + 1)

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

			label_pyramid_i = cv2.resize(self.input_label, self.x_pyramid[i].shape[:2], interpolation=cv2.INTER_NEAREST)
			label_pyramid_i = label_pyramid_i[:, :, np.newaxis].astype(float)

			for j in range(self.T):
				self.y_pyramid[i], self.y_label_pyramid[i] = self.PNN(self.x_pyramid[i], label_pyramid_i, keys, queries, self.PATCH_SIZE, self.STRIDE, self.ALPHA)
				queries = self.y_pyramid[i]
				keys = self.x_pyramid[i]

			if to_save:
				print(f"Saving sample {sample_id}")
				img_save(self.y_pyramid[i], os.path.join(self.out_folder, f"{sample_id}_scale{i}.png"))
				self.y_label_pyramid[i] = self.y_label_pyramid[i].astype(np.uint8)
				img_save(self.y_label_pyramid[i].astype(np.uint8), os.path.join(self.out_folder, f"{sample_id}_label_scale{i}.png"))
				label_save(self.y_label_pyramid[i], os.path.join(self.out_folder, f"{sample_id}_label_rgb_scale{i}.png"))
			else:
				return (self.y_pyramid[i], self.y_label_pyramid[i])

	def PNN(self, x, label, x_scaled, y_scaled, patch_size, stride, alpha):
		queries = extract_patches(y_scaled, patch_size, stride)
		keys = extract_patches(x_scaled, patch_size, stride)
		values = extract_patches(x, patch_size, stride)
		label_values = extract_patches(label, patch_size, stride, channels=1)
		dist = compute_distances(queries, keys)
		norm_dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
		NNs = torch.argmin(norm_dist, dim=1)  # find_NNs
		values = values[NNs]
		label_values = label_values[NNs]
		y = combine_patches(values, patch_size, stride, x_scaled.shape)
		y_label = combine_patches(label_values, patch_size, stride, label.shape, channels=1)
		return y, y_label

	def PNN_faiss(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None, new_keys=True):
		queries = extract_patches(y_scaled, patch_size, stride)
		keys = extract_patches(x_scaled, patch_size, stride)
		values = extract_patches(x, patch_size, stride)
		if mask is not None:
			queries = queries[mask]
			keys = keys[~mask]
		queries_flat = np.ascontiguousarray(queries.reshape((queries.shape[0], -1)).cpu().numpy(), dtype='float32')
		keys_flat = np.ascontiguousarray(keys.reshape((keys.shape[0], -1)).cpu().numpy(), dtype='float32')

		if new_keys:
			self.index = faiss.IndexFlatL2(keys_flat.shape[-1])
			self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
			self.index.add(keys_flat)
		D, I = self.index.search(queries_flat, 1)
		if mask is not None:
			values[mask] = values[~mask][I.T]
		else:
			values = values[I.T]
			#O = values[I.T]
		y = combine_patches(values, patch_size, stride, x_scaled.shape)
		return y


def extract_patches(src_img, patch_size, stride, channels=3):
	img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
	return torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
		.squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])


def compute_distances(queries, keys):
	dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float32, device=device)
	for i in range(len(queries)):
		dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3))
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
