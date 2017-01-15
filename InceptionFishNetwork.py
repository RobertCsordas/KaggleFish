import tensorflow as tf
import tensorflow.contrib.slim as slim
from InceptionResnetV2 import *
from tensorflow.python.ops import control_flow_ops

class InceptionFishNetwork:
	def __init__(self, name, input, nCategories, trainFeatures=False, training=True, reuse=False, weightDecay=0.00004, dropout_keep_prob=0.8, trainFrom='Conv2d_7b_1x1'):
		self.name=name
		self.nCategories=nCategories
		self.loss=None
		self.trainFeatures=trainFeatures
		self.trainFrom = trainFrom

		if trainFeatures:
			trFrom = "start"
		elif training:
			trFrom = trainFrom
		else:
			trFrom = None

		with tf.variable_scope(name, values=[input], reuse=reuse):
			self.googleNet = InceptionResnetV2("features", input, reuse=reuse, trainFrom = trFrom)

			net = self.googleNet.getOutput('PrePool')
			with tf.variable_scope('classifier', values=[net], reuse=reuse):
				with slim.arg_scope([slim.fully_connected],
						weights_regularizer=slim.l2_regularizer(weightDecay),
						biases_regularizer=slim.l2_regularizer(weightDecay)):
					
					net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
					net = slim.flatten(net)

					net = slim.dropout(net, dropout_keep_prob, is_training=training, scope='Dropout')
					
					self.outputs=slim.fully_connected(net, nCategories, activation_fn=None, scope='Logits')
			
	def importWeights(self, sess, filename):
		self.googleNet.importWeights(sess, filename, includeTraining=True)

	def getVars(self, includeFeatures=False):
		if not (includeFeatures or self.trainFeatures):
			vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/classifier/")
			scopes = self.googleNet.getScopes(fromLayer=self.trainFrom, inclusive=True)
			for s in scopes:
				vars +=  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=s)
		else:
			vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/")
		
		return vars

	def createLoss(self, referenceOutput, weights=None):
		with tf.name_scope('fish_loss') as scope:
			wFalse = 0.0001
			wTrue = 1.0 - (self.nCategories-1) * wFalse
			referenceOutput = tf.reshape(referenceOutput,[-1])

			# with tf.name_scope('softmax') as scope:
			# 	referenceOutput = tf.one_hot(referenceOutput, self.nCategories, on_value = 1.0 - (self.nCategories-1) * wFalse, off_value=wFalse, dtype=tf.float32)
			# 	if weights is None:
			# 		loss = tf.nn.softmax_cross_entropy_with_logits(self.outputs, referenceOutput)
			# 	else:
			# 		print("Weighted loss")
			# 		loss = tf.nn.weighted_cross_entropy_with_logits(self.outputs, referenceOutput, weights)

			# loss = tf.reduce_mean( loss )
			# loss = tf.add_n([loss] + self.getRegLosses(includeFeatures))

			ref = slim.one_hot_encoding(referenceOutput, self.nCategories, on_value = wTrue, off_value=wFalse)
			loss = slim.losses.softmax_cross_entropy(self.outputs, ref, scope="faszom")
			return loss

	def getOutputs(self):
		return tf.nn.softmax(self.outputs)
