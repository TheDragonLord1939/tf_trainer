import tensorflow as tf
import sys

checkpoint=sys.argv[1]
print("checkpoint:",checkpoint)
new_feature_path=sys.argv[2]
print("new_feature_path",new_feature_path)
outpath=sys.argv[3]
print("outpath",outpath)
embedding_dim = int(sys.argv[4])
print("embedding_dim",str(embedding_dim))

old_saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))
sess= tf.Session()
old_saver.restore(sess,checkpoint)
var_list = tf.trainable_variables()
embed = [ x for x in var_list if x.name.startswith("embedding_") and x.name.endswith("embeddings:0") ]
graph = tf.get_default_graph()
keys = graph.get_all_collection_keys()
for e in keys:
   graph.clear_collection(e)

for e in embed:
   print(e)
   graph.add_to_collection("trainable_variables",e)

new_var = []
new_var_info = []
f = open(new_feature_path,"r")
line=f.readline()
while line:
   if line.startswith("#"):
      print(line)
      line=f.readline()
      continue
   line_f = line.strip("\n").split(" ")
   field = line_f[0]
   sz = int(line_f[1])
   new_var_info.append([field,sz])
   line=f.readline()
for e in new_var_info:
   new_var.append(tf.get_variable("embedding_{}/embeddings".format(e[0]), [e[1], embedding_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer( minval=-0.05, maxval=0.05)))
    
sess.run(tf.variables_initializer(new_var))

c = tf.trainable_variables()
all_var=[]
for e in c:
    print(e)
    all_var.append(e)

new_saver = tf.train.Saver(all_var)
new_saver.save(sess,outpath)

