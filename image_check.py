from PIL import Image
import glob
import os

for subdir, dirs, files in os.walk('ucf11_160x120/tests'):
	for file in files:
		path = os.path.join(subdir, file)
		im = Image.open(path)
		width, height = im.size
		if width < 160:
			print(path)
