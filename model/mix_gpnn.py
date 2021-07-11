import glob
from collections import defaultdict

import numpy as np
import torch
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
from torch.nn.functional import fold, unfold
from .utils import *


class gpnn:
	def __init__(self, config):
		# general settings
		self.T = config['iters']
		self.PATCH_SIZE = (config['patch_size'], config['patch_size'])
		self.COARSE_DIM = (config['coarse_dim'], config['coarse_dim'])
		if config['task'] == 'inpainting':
			mask = img_read(config['mask'])
			mask_patch_ratio = np.max(np.sum(mask, axis=0), axis=0) // self.PATCH_SIZE
			coarse_dim = mask.shape[0] / mask_patch_ratio
			self.COARSE_DIM = (coarse_dim, coarse_dim)
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
		if config['task'] == 'structural_analogies':
			img_path = config['img_a']
		else:
			img_path = config['input_img']

		assert config['out_size'] == 0

		# pyramids
		imgs = glob.glob(f"{config['input_dir']}/*.jpg", recursive=True)
		first_img = imgs[0]
		first_img = img_read(first_img)
		pyramid_depth = np.log(min(first_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
		pyramid_depth = int(np.ceil(pyramid_depth))

		self.x_pyramid = [[] for _ in range(pyramid_depth+1)]

		for img in imgs:
			input_img = img_read(img)
			img_pyramid = (list(
				tuple(pyramid_gaussian(input_img, pyramid_depth, downscale=self.R, multichannel=True))))
			for idx, scaled_img in enumerate(img_pyramid):
				self.x_pyramid[idx].append(scaled_img)

		self.y_pyramid = [0] * (pyramid_depth + 1)

		# out_file
		filename = os.path.splitext(os.path.basename(img_path))[0]
		self.out_folder = os.path.join(config['out_dir'], config['task'], filename)

		# coarse settings
		self.coarse_img = img_read(config['input_img'])
		self.coarse_img = resize(self.coarse_img, self.x_pyramid[-1][0].shape)

	def run(self, sample_id, to_save=True):
		print(f"Working on sample: {sample_id}")
		for i in reversed(range(len(self.x_pyramid))):
			print(f"Working on scale: {i}")
			if i == len(self.x_pyramid) - 1:
				queries = self.coarse_img
				keys = self.x_pyramid[i]
			else:
				queries = resize(self.y_pyramid[i + 1], self.x_pyramid[i][0].shape)
				keys = [resize(x, self.x_pyramid[i][0].shape) for x in self.x_pyramid[i + 1]]
			new_keys = True
			for j in range(self.T):
				if self.is_faiss:
					self.y_pyramid[i] = self.PNN_faiss(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
													   self.ALPHA, mask=None, new_keys=new_keys)
				else:
					self.y_pyramid[i] = self.PNN(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
												 self.ALPHA)
				queries = self.y_pyramid[i]
				keys = self.x_pyramid[i]
				if j > 1:
					new_keys = False
			if to_save:
				print(f"Saving sample {sample_id}")
				img_save(self.y_pyramid[i], os.path.join(self.out_folder, f"{sample_id}_scale{i}.png"))
			else:
				return self.y_pyramid[i]

	def PNN(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None):
		queries = extract_patches(y_scaled, patch_size, stride)
		keys = extract_patches(x_scaled, patch_size, stride)
		values = extract_patches(x, patch_size, stride)
		if mask is None:
			dist = compute_distances(queries, keys)
		else:
			dist = compute_distances(queries[mask], keys[~mask])
		norm_dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
		NNs = torch.argmin(norm_dist, dim=1)  # find_NNs
		if mask is None:
			values = values[NNs]
		else:
			values[mask] = values[~mask][NNs]
			# O = values[NNs]  # replace_NNs(values, NNs)
		y = combine_patches(values, patch_size, stride, x_scaled[0].shape)
		return y

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


def extract_patches(src_img, patch_size, stride):
	channels = 3
	patches = None
	if type(src_img) == list:
		for img in src_img:
			img = torch.from_numpy(img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
			current_patches = torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
				.squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])
			if patches is None:
				patches = current_patches
			else:
				patches = torch.cat([patches, current_patches])
	else:
		img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
		patches = torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
			.squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])

	return patches

def compute_distances(queries, keys):
	dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float32, device=device)
	for i in range(len(queries)):
		dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3))
	return dist_mat


def combine_patches(O, patch_size, stride, img_shape):
	channels = 3
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
