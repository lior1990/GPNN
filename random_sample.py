import argparse
import os
import glob
from pathlib import Path


from model.gpnn import gpnn
from model.parser import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser = parser_general(parser)
	parser = parser_sample(parser)
	config = vars(parser.parse_args())

	imgs = []
	if os.path.isdir(config["input_img"]):
		imgs = glob.glob(f"{config['input_img']}/*.jpg", recursive=True)
	else:
		imgs = [config["input_img"]]

	for img in imgs:
		config["input_img"] = img
		print(f"Working on img {img}")
		img_name = Path(img).stem
		for i in range(config["n_samples"]):
			model = gpnn(config)
			model.run(f"{img_name}_{i}")

	print("Done")
