import tensorflow as tf

def variableSummary(var):
  if not isinstance(var, list):
    var=[var]

  for v in var:
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(v)
      tf.summary.scalar('mean/' + v.op.name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(v - mean)))
      tf.summary.scalar('stddev/' + v.op.name, stddev)
      tf.summary.scalar('max/' + v.op.name, tf.reduce_max(v))
      tf.summary.scalar('min/' + v.op.name, tf.reduce_min(v))
      tf.histogram_summary(v.op.name, v)

def createSummaryForAllVars():
  variableSummary(tf.trainable_variables())

def pyhtonFloatSummary(name):
  p=tf.placeholder(tf.float32)
  s=tf.summary.scalar(name, p)
  return s, p

def imageSummary(var):
  res=[]
  for name in var:
    res.append(tf.image_summary(name, var[name]))

  return res