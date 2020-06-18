"""

"""
# 
from PIL import Image
from sys import argv
import numpy as np

# 
from Kmeans import kmeans, runKmeansTuned

def main(k=-1):
	if len(argv) < 2 or len(argv) > 3:
		usage()
	outFile = argv[2] if len(argv) == 3 else "output.jpg"
	CUSTOM_K = 2
	K = k if k else CUSTOM_K

	img = Image.open(argv[1]) # open input file
	imgArray = np.array(img)
	ogDim = imgArray.shape
	imgArray = imgArray.reshape((ogDim[0]*ogDim[1], 3))
	poi = kmeans(imgArray, K=K, PRINT=True)
	print(poi)
	for i in range(len(imgArray)):
		color = int(poi[1][i])
		imgArray[i] = poi[0][color]
	imgArray = imgArray.reshape((ogDim[0], ogDim[1], 3))
	img = Image.fromarray(imgArray, "RGB")
	img.save(outFile)
	return img

def usage():
	msg = "USAGE:\n" \
	"decolor.py INPUT_FILE\n" \
	"decolor.py INPUT_FILE OUTPUT_FILE"

	print(msg)


if __name__ == "__main__":
	if len(argv) < 4:
		main()
	elif len(argv) > 3 and type(int(argv[3])) is int:
		imgs = [main(k) for k in range(1,int(argv[3])+1)]
		print(imgs)
		imgs[0].save("output.gif", save_all=True, optimize=True, append_images=imgs[1:], duration=500, loop=0)

