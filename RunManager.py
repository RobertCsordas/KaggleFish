class RunManager():
	def __init__(self, sess):
		self.sess=sess
		self.groups={}

	def add(self, name, tensorList, enabled=True, modRun=0):
		self.groups[name]={
			"inList": tensorList,
			"enabled": enabled,
			"modRun": modRun
		}

	def appendToInput(self, name, list):
		startIndex = len(self.inputTensors)
		self.inputTensors+=self.groups[name]["inList"]
		self.indexList.append({
			"name": name,
			"index": startIndex
		})

	def clearInput(self):
		self.indexList=[]
		self.inputTensors=[]

	def buildInputFromNames(self, names):
		self.clearInput()
		for k in names:
			self.appendToInput(k, self.groups[k]["inList"])

	def buildInputFromEnabled(self):
		self.clearInput()
		for k in self.groups:
			g=self.groups[k]
			if g["enabled"] != True:
				continue

			self.appendToInput(k, g["inList"])

	def buildInputMod(self, counter):
		self.clearInput()
		for k in self.groups:
			g=self.groups[k]
			m=g["modRun"]
			if m<=0 or g["enabled"] != True or counter % m != 0:
				continue

			self.appendToInput(k, g["inList"])

	def runAndMerge(self, feed_dict=None, options=None, run_metadata=None):
		res = self.sess.run(self.inputTensors, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
		result = {}

		for i in self.indexList:
			name=i["name"]
			cnt=len(self.groups[name]["inList"])
			startIndex=i["index"]
			r=res[startIndex:startIndex+cnt]
			result[name]=r if cnt > 1 else r[0]

		return result

	def run(self, names=None, feed_dict=None, options=None, run_metadata=None):
		if names is None:
			self.buildInputFromEnabled()
		else:
			self.buildInputFromNames(names)

		return self.runAndMerge(feed_dict, options=options, run_metadata=run_metadata)

	def modRun(self, counter, feed_dict=None, options=None, run_metadata=None):
		self.buildInputMod(counter=counter)
		return self.runAndMerge(feed_dict, options=options, run_metadata=run_metadata)

	def enable(self, name, enabled=True):
		self.groups[name]["enabled"]=enabled

	def disable(self, name):
		self.enable(name, False)
