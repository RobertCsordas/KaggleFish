import random
import cv2
import numpy as np
import threading
import tensorflow as tf
import os
import json
from tqdm import tqdm
import BoxAwareRandZoom

class KaggleFishLoader(object):
	FISHES = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
	FISHES_LONG = ["Albacore tuna", "Bigeye tuna", "Mahi Mahi", "Opah", "No fish", "Other", "Shark", "Yellowfin tuna"]

	QUEUE_CAPACITY=4

	def __init__(self, path, testRatio=0.15, downscale=2, randZoom=True):
		self.root = path
		self.data = self.loadCache(path)
		self.imgSize = (int(1280/downscale),int(704/downscale))
		self.downscale = downscale
		self.boundingBoxes={}
		self.randZoom=randZoom

		if self.data is None:
			allData = self.addTrain(self.root+"/train")
			train, test = self.splitTrain(allData, testRatio)
			self.data = {
				"train": train, 
				"test": test,
				"all": allData,
				"benchmark": self.addTest(self.root+"/test_stg1")
			}
			self.saveCache(path, self.data)

		self.printStats("train")
		self.printStats("test")

		with tf.name_scope('dataset') as scope:
			self.queue = tf.FIFOQueue(shapes=[[self.imgSize[1], self.imgSize[0], 3], [1]],
			       dtypes=[tf.float32, tf.uint8],
			       capacity=self.QUEUE_CAPACITY)

			self.image = tf.placeholder(dtype=tf.float32, shape=[1,self.imgSize[1], self.imgSize[0], 3])
			self.category = tf.placeholder(dtype=tf.uint8, shape=[1,1])

			self.enqueueOp = self.queue.enqueue_many([self.image, self.category])

		self.calculateCategoryWeights()

		self.addAllBoundingBoxes(path+"/bbox/")

	def addAllBoundingBoxes(self, path):
		for fn in os.listdir(path):
			if fn.split(".")[-1].lower()!="json":
				continue

			print("Loading "+fn)
			self.addBoundingBoxes(path+"/"+fn)

	def addBoundingBoxes(self, file):
		with open(file,"r") as data_file:
			flist=json.load(data_file)

		for file in flist:
			fname = file["filename"].split("/")[-1]
			if fname in self.boundingBoxes:
				print("WARNING: File "+fname+" already exists.")
				continue
			annList = []
			
			for ann in file["annotations"]:
				if ann["class"]!="rect":
					print("WARNING: Invalid annotation class: "+ann["class"])
					continue	

				annList.append({
					"x": int(ann["x"]/self.downscale),
					"y": int(ann["y"]/self.downscale),
					"h": int(ann["height"]/self.downscale),
					"w": int(ann["width"]/self.downscale),
				})	
			
			self.boundingBoxes[fname] = annList


	def getBboxList(self, name):
		if name in self.boundingBoxes:
			return self.boundingBoxes[name]
		else:
			return []

	def printStats(self, name):
		nFiles = len(self.data[name]["files"])
		print("Loaded "+str(nFiles)+" "+name+" images")
		for c, cat in self.data[name]["categories"].items():
			print("   "+KaggleFishLoader.FISHES[c]+": "+str(len(cat))+ "\t("+str(len(cat)*100//nFiles)+" %)")

	def calculateCategoryWeights(self):
		self.categoryWeights = {}
		for dset in self.data:
			if dset=="benchmark":
				continue

			self.categoryWeights[dset]=np.zeros((len(self.FISHES),), dtype=np.float32)
			for f in range(len(self.FISHES)):
				self.categoryWeights[dset][f]=len(self.data[dset]["categories"][f])/len(self.data[dset]["files"])

		
	def getCategoryWeights(self, dataset="train"):
		return self.categoryWeights[dataset]

	def getCategoryCount(self):
		return len(KaggleFishLoader.FISHES)

	def getCacheFilename(self, path):
		if not os.path.isdir("./cache"):
			os.makedirs("./cache")

		return "./cache/"+("_".join(path.split("/")))

	def loadCache(self, path):
		cfile=self.getCacheFilename(path)
		if not os.path.isfile(cfile):
			return None

		with open(cfile,"r") as data_file:
			return self.fixJSON(json.load(data_file))

	def fixIntDict(self, d):
		return {int(k):v for k,v in d.items()}

	def fixJSON(self, obj):
		if obj is None:
			return None

		for s, o in obj.items():
			if s=="benchmark":
				continue

			o["categories"]=self.fixIntDict(o["categories"])

		return obj


	def saveCache(self, path, data):
		cfile=self.getCacheFilename(path)
		with open(cfile, 'w') as outfile:
			json.dump(data, outfile)


	def addTest(self, path):
		res=[]

		for fn in os.listdir(path):
			if fn.split(".")[-1].lower()!="jpg":
				continue

			res.append(fn)

		res.sort()
		return res

	def splitTrain(self, trainSet, ratio):
		def copySet(dest, src, flist, indices):
			for i in indices:
				fIndex = flist[i]

				f = src["files"][fIndex]
				dest["files"].append(f)
				if f[1] not in dest["categories"]:
					dest["categories"][f[1]]=[]

				dest["categories"][f[1]].append(len(dest["files"])-1)


		rnd = np.random.RandomState(0xB0C1FA52)

		train = {
			"files": [],
			"categories": {},
		}

		test = {
			"files": [],
			"categories": {},
		}

		for c, flist in trainSet["categories"].items():
			testCnt = int(len(flist)*ratio)

			perm = rnd.permutation(len(flist))
			testIndices = np.sort(perm[0:testCnt])
			trainIndices = np.sort(perm[testCnt:])

			copySet(train, trainSet, flist, trainIndices)
			copySet(test, trainSet, flist, testIndices)

		return train, test

	def addTrain(self, path):
		dataset={
			"files": [],
			"categories": {},
		};

		fArray = []
		reverseCategories=[]

		path+="/"
		for fn in os.listdir(path):
			if not os.path.isdir(path+fn):
				print("WARNING: invalid training data format. File found at root of train data: "+fn)
				continue;

			if not fn in KaggleFishLoader.FISHES:
				print("WARNING: unknown fish type: "+fn)
				continue

			category = KaggleFishLoader.FISHES.index(fn)

			for img in os.listdir(path+fn):
				fullFn = fn+"/"+img

				if img.split(".")[-1].lower()!="jpg":
					print("WARNING: invalid image: "+fullFn)
					continue;

				fArray.append(fullFn)
				reverseCategories.append(category)

		#ensure deterministic ordering
		indices = np.argsort(fArray)
	
		for i in range(len(indices)):
			c = reverseCategories[indices[i]]
			if c not in dataset["categories"]:
				dataset["categories"][c]=[]

			dataset["categories"][c].append(i)
			dataset["files"].append([fArray[indices[i]], c])
		
		return dataset

	def analizeSizes(self):
		sizes={}
		for a in tqdm(range(len(self.data["train"]["files"]))):
			f=self.data["train"]["files"][a][0]
			img = cv2.imread(self.root+"/train/"+f)

			s=str(img.shape[1])+"x"+str(img.shape[0])+"x"+str(img.shape[2])
			if s not in sizes:
				sizes[s]=0

			sizes[s]+=1
		return sizes

	def sampleFileIndex(self, dataset):
		return random.randint(0, len(self.data[dataset]["files"])-1)


	def fetch(self, dataset="train"):
		while True:
			f=self.data[dataset]["files"][self.sampleFileIndex(dataset)]

			img = cv2.imread(self.root+"/train/"+f[0],3)
			if img is None:
				print("WARNING: failed to load image "+f[0])
				continue

			img = cv2.resize(img, self.imgSize)

			if self.randZoom:
				img = BoxAwareRandZoom.randZoom(img, self.getBboxList(f[0].split("/")[-1]))

			return np.expand_dims(img,0), np.full((1,1),f[1],dtype=np.uint8)

	def threadFn(self, tid, sess):
	 	while True:
	 		img, category=self.fetch()
	 		sess.run(self.enqueueOp,feed_dict={self.image:img, self.category:category})


	def startThreads(self, sess, nThreads=4):
		self.threads=[]
		for n in range(nThreads):
			t=threading.Thread(target=self.threadFn, args=(n,sess))
			t.daemon = True
			t.start()
			self.threads.append(t)

	def get(self, batchSize=8):
	 	return self.queue.dequeue_many(batchSize)

	def count(self, set="train"):
		if set=="benchmark":
			return len(self.data["benchmark"])
		elif set not in self.data:
			assert False, "Invalid dataset"
		else:
			return len(self.data[set]["files"])
		

	def getBenchmarkSamples(self, count, pos=0):
		flist=[]
		count = np.min([count, len(self.data["benchmark"])-pos])
		res = np.empty((count, self.imgSize[1], self.imgSize[0],3), dtype=np.float32)
		for i in range(count):
			img = cv2.imread(self.root+"/test_stg1/"+self.data["benchmark"][i+pos], 3)
			assert img is not None, "Failed to open image "+self.data["benchmark"][i+pos]
			res[i] = cv2.resize(img, self.imgSize)
			flist.append(self.data["benchmark"][i+pos])

		return res, flist
