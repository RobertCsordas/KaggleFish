import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
import threading

class Augmenter:
	def nonlinear(self, imageList, lower, upper):
		with tf.name_scope('nonlinear') as scope:
			factor = random_ops.random_uniform([], lower, upper)

			res=[]
			for i in imageList:
				res.append(tf.pow(i, factor))

			return res

	def randomNormal(self, imageList, stddev):
		with tf.name_scope('randomNormal') as scope:
			factor = random_ops.random_uniform([], 0, stddev)

			res=[]
			for i in imageList:
				res.append(i+tf.random_normal(tf.shape(i), mean=0.0, stddev=factor))

			return res

	def mirror(self, image):
		with tf.name_scope('mirror') as scope:
			uniform_random = random_ops.random_uniform([], 0, 1.0)
			uniform_random2 = random_ops.random_uniform([], 0, 1.0)
			mirror = math_ops.less(array_ops.pack([1.0, uniform_random2, uniform_random, 1.0]), 0.5)
			return array_ops.reverse(image, mirror)
			
	def __init__(self, image, category):
		with tf.name_scope('augmentation') as scope:
			image=image/255.0
			image = self.nonlinear([image], 0.8, 1.2)[0]

			image = self.mirror(image)

			image = tf.image.random_contrast(image, lower=0.3, upper=1.3)
			image = tf.image.random_brightness(image, max_delta=0.3)

			image = self.randomNormal([image], 0.025)[0]

			image = tf.clip_by_value(image, 0, 1.0)*255

			iShape = image.get_shape().as_list()[1:]
			
			self.queue = tf.RandomShuffleQueue(shapes=[iShape, [1]],
				dtypes=[tf.float32, tf.uint8],
				capacity=64,
				min_after_dequeue=16)

			self.enqueue=self.queue.enqueue_many([image, category])

	def startThreads(self, sess, nThreads=2):
		self.threads=[]
		for n in range(nThreads):
			t=threading.Thread(target=self.threadFn, args=(n,sess))
			t.daemon = True
			t.start()
		self.threads.append(t)

	def threadFn(self, tid, sess):
		while True:
			sess.run(self.enqueue)

	def get(self, batchSize):
		return self.queue.dequeue_many(batchSize)