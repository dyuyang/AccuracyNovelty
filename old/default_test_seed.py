import numpy as np
import tensorflow as tf

ITERATIONS=20

tf.set_random_seed(42)
np.random.seed(42)

x_data = np.random.normal(size=[32, 10])
y_data = np.random.normal(size=[32, 1])
x_test = np.random.normal(size=[32, 10])

x_in  = tf.placeholder(tf.float32, [None, 10])
y_in  = tf.placeholder(tf.float32, [None, 1])
x     = x_in
x     = tf.layers.dense(x, 200, tf.nn.relu)
x     = tf.layers.dense(x, 1, tf.nn.relu)
loss  = tf.losses.mean_squared_error(y_in, x)

mvars = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

opt   = tf.train.AdamOptimizer(use_locking=True)
train = opt.minimize(loss)
config= tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
sess  = tf.Session(config=config)

allvars = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

sess.run(tf.global_variables_initializer())
init_vals = sess.run(allvars)

def run():
  for val, v in zip(init_vals, allvars):
    sess.run(tf.assign(v, val))
  
  ivals = sess.run(allvars)
  out = []
  allvals = []

  for i in range(ITERATIONS):
    l, _ = sess.run([loss, train], feed_dict={x_in: x_data, y_in: y_data})
    out.append(sess.run(x, feed_dict={x_in: x_test}))
    allvals.append(sess.run(allvars))

  fvals = sess.run(allvars)
  # return np.asarray(ivals), np.asarray(fvals), np.asarray(out)
  return np.asarray(ivals), np.asarray(fvals), np.asarray(out), allvals

ivals1, fvals1, out1, all1 = run()
ivals2, fvals2, out2, all2 = run()

same_init = [np.all(v1 == v2) for v1, v2 in zip(ivals1, ivals2)] 
same_fin = [np.all(v1 == v2) for v1, v2 in zip(fvals1, fvals2)] 
print("Forward passes were the same: {}".format( np.all(out1 == out2) ))
print("Final value of variables are the same: {}".format( np.all(same_fin) ))
print("Variables initialized to same values: {}".format( np.all(same_init) ))