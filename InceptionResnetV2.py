import tensorflow as tf
import tensorflow.contrib.slim as slim
import CheckpointLoader

class InceptionResnetV2:
	@staticmethod
	def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
		"""Builds the 35x35 resnet block."""
		with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
			with tf.variable_scope('Branch_1'):
				tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
				tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
			with tf.variable_scope('Branch_2'):
				tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
				tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
				tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
			mixed = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2])
			up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
			net += scale * up
			if activation_fn:
				net = activation_fn(net)
		return net

	@staticmethod
	def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
		"""Builds the 17x17 resnet block."""
		with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
			with tf.variable_scope('Branch_1'):
				tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
				tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
				tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1], scope='Conv2d_0c_7x1')
			mixed = tf.concat(3, [tower_conv, tower_conv1_2])
			up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
			net += scale * up
			if activation_fn:
				net = activation_fn(net)
		return net

	@staticmethod
	def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
		"""Builds the 8x8 resnet block."""
		with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
			with tf.variable_scope('Branch_1'):
				tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
				tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3], scope='Conv2d_0b_1x3')
				tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1], scope='Conv2d_0c_3x1')
			mixed = tf.concat(3, [tower_conv, tower_conv1_2])
			up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
			net += scale * up
			if activation_fn:
				net = activation_fn(net)
		return net

	@staticmethod
	def define(inputs, reuse, weightDecay, scope='InceptionResnetV2', trainFrom=None):
		"""Creates the Inception Resnet V2 model.
		Args:
			inputs: a 4-D tensor of size [batch_size, height, width, 3].
			num_classes: number of predicted classes.
			is_training: whether is training or not.
			reuse: whether or not the network and its variables should be reused. To be
			  able to reuse 'scope' must be given.
			scope: Optional variable_scope.
		Returns:
			logits: the logits outputs of the model.
			end_points: the set of end_points from the inception model.
		"""

		with tf.name_scope('preprocess'):
			#BGR -> RGB
			inputs = tf.reverse(inputs, [False, False, False, True])
			#Normalize
			inputs = 2.0*(inputs/255.0 - 0.5)

		end_points = {}
		scopes = []


		trainBatchNormScope = slim.arg_scope([slim.batch_norm], is_training=True)
		weightDecayScope = slim.arg_scope([slim.conv2d, slim.fully_connected],
				weights_regularizer=slim.l2_regularizer(weightDecay),
				biases_regularizer=slim.l2_regularizer(weightDecay))

		trainBnEntered = False

		currBlock = ""
		def beginBlock(name):
			nonlocal trainBnEntered
			nonlocal currBlock

			currBlock = name
			if (trainFrom is not None) and (not trainBnEntered) and (trainFrom==name or trainFrom=="start"):
				print("Enabling training on "+trainFrom)
				trainBatchNormScope.__enter__()
				weightDecayScope.__enter__()
				trainBnEntered=True

		def endBlock(net, scope=True, name=None):
			if name is None:
				name = currBlock
			end_points[name]=net
			if scope:
				scopes.append(name)
		
		def endAll():
			if trainBnEntered:
				trainBatchNormScope.__exit__(None, None, None)
				weightDecayScope.__exit__(None,None,None)

		with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse) as scope:
			with slim.arg_scope([slim.batch_norm], is_training=False):
				with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

					# 149 x 149 x 32
					beginBlock('Conv2d_1a_3x3')
					net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
					endBlock(net)
					# 147 x 147 x 32
					beginBlock('Conv2d_2a_3x3')
					net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
					endBlock(net)
					# 147 x 147 x 64
					beginBlock('Conv2d_2b_3x3')
					net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
					endBlock(net)
					# 73 x 73 x 64
					beginBlock('MaxPool_3a_3x3')
					net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
					endBlock(net)
					# 73 x 73 x 80
					beginBlock('Conv2d_3b_1x1')
					net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
					endBlock(net)
					# 71 x 71 x 192
					beginBlock('Conv2d_4a_3x3')
					net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
					endBlock(net)
					# 35 x 35 x 192
					beginBlock('MaxPool_5a_3x3')
					net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
					endBlock(net)

					# 35 x 35 x 320
					beginBlock('Mixed_5b')
					with tf.variable_scope('Mixed_5b'):
						with tf.variable_scope('Branch_0'):
							tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
						with tf.variable_scope('Branch_1'):
							tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
							tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
						with tf.variable_scope('Branch_2'):
							tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
							tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
							tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
						with tf.variable_scope('Branch_3'):
							tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
							tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
						net = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1])

					endBlock(net)
					beginBlock('Repeat')
					net = slim.repeat(net, 10, InceptionResnetV2.block35, scale=0.17)
					endBlock(net)

					# 17 x 17 x 1024
					beginBlock('Mixed_6a')
					with tf.variable_scope('Mixed_6a'):
						with tf.variable_scope('Branch_0'):
							tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
						with tf.variable_scope('Branch_1'):
							tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
							tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
							tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
						with tf.variable_scope('Branch_2'):
							tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
						net = tf.concat(3, [tower_conv, tower_conv1_2, tower_pool])
					endBlock(net)

					beginBlock('Repeat_1')
					net = slim.repeat(net, 20, InceptionResnetV2.block17, scale=0.10)
					endBlock(net)
					endBlock(net, scope=False, name='aux')

					beginBlock('Mixed_7a')
					with tf.variable_scope('Mixed_7a'):
						with tf.variable_scope('Branch_0'):
							tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
							tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
						with tf.variable_scope('Branch_1'):
							tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
							tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
						with tf.variable_scope('Branch_2'):
							tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
							tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3, scope='Conv2d_0b_3x3')
							tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
						with tf.variable_scope('Branch_3'):
							tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
						net = tf.concat(3, [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool])
					endBlock(net)

					beginBlock('Repeat_2')
					net = slim.repeat(net, 9, InceptionResnetV2.block8, scale=0.20)
					endBlock(net)

					beginBlock('Block8')
					net = InceptionResnetV2.block8(net, activation_fn=None)
					endBlock(net)

					beginBlock('Conv2d_7b_1x1')
					net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
					endBlock(net)
					endBlock(net, scope=False, name='PrePool')

					endAll()

			return end_points, scope, scopes


	def __init__(self, name, inputs, trainFrom = None, reuse=False, weightDecay=0.00004, batchNormDecay=0.9997, batchNormEpsilon=0.001):
		self.inputShape = inputs.get_shape().as_list()
		self.name = name
		self.inputs = inputs
		self.trainFrom = trainFrom

		with slim.arg_scope([slim.conv2d, slim.fully_connected],
				weights_regularizer=None,
				biases_regularizer=None):

			batch_norm_params = {
				'decay': batchNormDecay,
				'epsilon': batchNormEpsilon,
			}
			# Set activation_fn and parameters for batch_norm.
			with slim.arg_scope([slim.conv2d],
					activation_fn=tf.nn.relu,
					normalizer_fn=slim.batch_norm,
					normalizer_params=batch_norm_params) as scope:

				self.endPoints, self.scope, self.scopeList = InceptionResnetV2.define(inputs, weightDecay = weightDecay, trainFrom=trainFrom, scope=name, reuse = reuse)

	def importWeights(self, sess, filename, includeTraining=False):
		ignores = [] if includeTraining or (self.trainFrom is None) else self.getScopes(fromLayer = self.trainFrom, inclusive = True)
		print("Ignoring blocks:")
		print(ignores)
		CheckpointLoader.importIntoScope(sess, filename, fromScope="InceptionResnetV2", toScope=self.scope.name, ignore=ignores)

	def getOutput(self, name=None):
		if name is None:
			return self.endPoints
		else:
			return self.endPoints[name]

	def getScopes(self, fromLayer = None, toLayer = None, inclusive=False):
		l=[]
		print(fromLayer)
		if fromLayer is not None:
			assert(toLayer is None)
			i = self.scopeList.index(fromLayer)
			if not inclusive:
				i += 1
			l=self.scopeList[i:]
		elif toLayer is not None:
			assert(fromLayer is None)
			i = self.scopeList.index(toLayer)
			if not inclusive:
				i -= 1
			if i<0:
				l=[]
			else:
				l=self.scopeList[:(i+1)]
		else:
			l=self.scopeList

		return [self.scope.name+"/"+s+"/" for s in l]



#
# for v in reader.get_variable_to_shape_map().keys():
# 	var=tf.contrib.slim.get_variables_by_name(v)
# 	if len(var)<1:
# 		print("WARNING: failed to read variable "+v)
# 		continue;
#
# 	print("Loading "+v)
# 	loadedVars.append(v)
# 	for va in var:
# 		toRun.append(tf.reshape(va.assign(reader.get_tensor(v)), [-1])[0])
#
# return toRun, loadedVars
# 		vlist=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
# 		AdasTensor.readVars(sess, dirname, vlist, self.name, {
# 			"conv1_1/3x3_s2/weights": {"RGBBGR" : permutateRgb}
# 		})