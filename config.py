# Dimensionality of character embedding
embedding_size = 300
embedding_size_kernel = 300
# filter sizes
filter_sizes = [3, 4, 5]
# choose 100-600 range
# Number of filters per filter size
num_filters = 600
# L2 regularizaion lambda
l2_reg_lambda = 11

# Dropout keep probability
dropout_keep_prob = 0.7
# Batch Size
batch_size = 64
# Number of training epochs
num_epochs = 200
# Evaluate model on dev set after this many steps
evaluate_every = 50
# Save model after this many steps
checkpoint_every = 100
# learning rate
learning_rate = 1e-3
# same_accuracy_iter_max = 2
# accuracy_diff_epsilon = 10**(-4)

gen_loss_alpha = 25
epsilon_val_mov_avg = 1e-5
successive_strips = 5


checkpoint_path = './checkpoint'
out_dir = './output'

resume = False

word2vec_path = './data/word2vec/GoogleNews-vectors-negative300.bin'
word2vec_embed_path = "./output/vocab/embed_subsample_glove.npy"
# word2vec_embed_path = "./output/vocab/embed_subsample.npy"
word2vec_embed_size = 300

alphabet_string = '''abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"/\|_@#$%^&*~`+-=<>()[]{}\n'''
