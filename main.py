#!/usr/bin/python

import tensorflow as tf
from FishDataset import *
import Augment
from RunManager import *
from CheckpointLoader import *
from tqdm import tqdm
from ArgSave import *
import sys
import Summary
from InceptionFishNetwork import *
import Model

Model.download()

parser = StorableArgparse(description='Kaggle fish trainer.')
parser.add_argument('-learningRate', type=float, default=0.01, help='Learning rate')
parser.add_argument('-batchSize', type=int, default=8, help='Batch size')
parser.add_argument('-dataset', type=str, default="/data/Datasets/KaggleFish", help="Path to kaggle dataset")
parser.add_argument('-name', type=str, default="save", help="Directory to save checkpoints")
parser.add_argument('-saveInterval', type=int, default=500, help='Save model for this amount of iterations')
parser.add_argument('-reportInterval', type=int, default=5, help='Repeat after this amount of iterations')
parser.add_argument('-displayInterval', type=int, default=10, help='Display after this amount of iterations')
parser.add_argument('-learnFeatures', type=int, default=0, help='Enable learning of googlenet features')
parser.add_argument('-randZoom', type=int, default=1, help='Enable random zooming')
parser.add_argument('-weight', type=int, default=0, help='Enable class weighting')
parser.add_argument('-optimizer', type=str, default='adam', help='sgd/adam/rmsprop')
parser.add_argument('-resume', type=str, help='Resume from this file', save=False)
parser.add_argument('-report', type=str, default="", help='Create report here', save=False)
parser.add_argument('-trainFrom', type=str, default="Conv2d_7b_1x1", help='Train from this layer')

opt=parser.parse_args()

if not os.path.isdir(opt.name):
	os.makedirs(opt.name)

opt = parser.load(opt.name+"/args.json")
parser.save(opt.name+"/args.json")

globalStep = tf.Variable(0, name='globalStep', trainable=False)
totalCount = tf.Variable(0, name='totalCount', trainable=False)
totalCountInc = totalCount.assign_add(opt.batchSize)
globalStepInc=tf.assign_add(globalStep,1)

if not os.path.isdir(opt.name+"/log"):
	os.makedirs(opt.name+"/log")

if not os.path.isdir(opt.name+"/save"):
	os.makedirs(opt.name+"/save")

if not os.path.isdir(opt.name+"/preview"):
	os.makedirs(opt.name+"/preview")

dataset = KaggleFishLoader(opt.dataset, randZoom=opt.randZoom==1)

if opt.weight==1:
	print("Using class weighting")
	weight = dataset.getCategoryWeights()
	print(weight)
else:
	weight = None

img, category = dataset.get(batchSize=1)
augmenter = Augment.Augmenter(img, category)

img, category = augmenter.get(batchSize = opt.batchSize)

net = InceptionFishNetwork("fishnet", img, nCategories=dataset.getCategoryCount(), trainFeatures=opt.learnFeatures == 1, trainFrom=opt.trainFrom)
if opt.optimizer=="sgd":
	optimizer=tf.train.MomentumOptimizer(learning_rate=opt.learningRate, momentum=0.9)
elif opt.optimizer=="adam":
	optimizer=tf.train.AdamOptimizer(learning_rate=opt.learningRate, epsilon=1)
elif opt.optimizer=="rmsprop":
	optimizer=tf.train.RMSPropOptimizer(learning_rate=opt.learningRate, epsilon=1, momentum=0.9)
else:
	print("Invalid optimizer: "+opt.optimizer)
	sys.exit(-1)

with tf.name_scope('testNetwork') as scope:
	testInput=tf.placeholder(tf.float32, shape=(1,dataset.imgSize[1],dataset.imgSize[0],3))
	testNet=InceptionFishNetwork("fishnet", testInput, nCategories=dataset.getCategoryCount(), training=False, reuse=True)
	testNetOut=testNet.getOutputs()

def createReport(sess, fname):
	with open(fname, 'w+') as out:
		out.write("image,"+(",".join(dataset.FISHES))+"\n")
		n = dataset.count("benchmark")
		for i in tqdm(range(n)):
			data, files = dataset.getBenchmarkSamples(count=1, pos=i)
			res = sess.run([testNetOut], feed_dict={testInput: data})[0]

			r = files[0]
			for j in range(res.shape[1]):
				r+=",%.17f"%(res[0][j])

			out.write(r+"\n")


def test(sess):
	n = dataset.count("test")
	l = 0
	for i in tqdm(range(n)):
		img, category = dataset.fetch("test")
		category=category[0][0]

		res = sess.run([testNetOut], feed_dict={testInput: img})[0]
		l-=np.log(res[0][category])

	return l/n

log = tf.summary.FileWriter(opt.name+"/log", graph=tf.get_default_graph())

net.createLoss(category)

def createUpdateOp(gradClip=None):
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	totalLoss = slim.losses.get_total_loss()
	grads = optimizer.compute_gradients(totalLoss, var_list=net.getVars())
	if gradClip is not None:
		grads = [(tf.clip_by_value(grad, -float(gradClip), float(gradClip)), var) for grad, var in grads]
	update_ops.append(optimizer.apply_gradients(grads))
	return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

trainStep = createUpdateOp()

trainLossSum, trainLossFeed = Summary.pyhtonFloatSummary("trainLoss")
testLossSum, testLossFeed = Summary.pyhtonFloatSummary("testLoss")

saver=tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=300)

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
	print("Loading ckeckpoint/initial data")
	if not loadCheckpoint(sess, opt.name+"/save/", opt.resume):
		print("Loading GoogleNet")
		net.importWeights(sess, Model.FILENAME)
		print("Done.")

	if opt.report!="":
		createReport(sess, opt.report)
		sys.exit(0)

	dataset.startThreads(sess, nThreads=4)
	augmenter.startThreads(sess, nThreads=2)

	runManager = RunManager(sess)
	runManager.add("train", [globalStepInc,totalCountInc,trainStep], modRun=1)
	
	i=0
	cycleCnt=0
	lossSum=0

	while True:
		res = runManager.modRun(i)
		i, samplesSeen, loss=res["train"]

		lossSum+=loss
		cycleCnt+=1

		if "testImage" in res:
			dimg=res["testImage"][0]
			dcat=res["testImage"][1]
			cv2.putText(dimg, dataset.FISHES[dcat], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
			cv2.imwrite(opt.name+"/preview/perview.jpg", dimg )

		if "summary" in res:
			log.add_summary(res["summary"], global_step=samplesSeen)

		if i % opt.reportInterval == 1:
			loss=lossSum/cycleCnt

			lossS=sess.run(trainLossSum, feed_dict={
				trainLossFeed: loss
			})
			log.add_summary(lossS, global_step=samplesSeen)

			epoch="%.2f" % (float(samplesSeen) / dataset.count())
			print("Iteration "+str(i)+" (epoch: "+epoch+"): loss: "+str(loss))
			lossSum=0
			cycleCnt=0

		if i % opt.saveInterval == 0:
			print("Testing...")
			loss = test(sess)

			lossS=sess.run(testLossSum, feed_dict={
				testLossFeed: loss
			})
			log.add_summary(lossS, global_step=samplesSeen)

			print("Test loss: "+str(loss))
			print("Saving checkpoint "+str(i))
			saver.save(sess, opt.name+"/save/model_"+str(samplesSeen)+"_"+str(loss), write_meta_graph=False)
