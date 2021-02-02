# Training
OPTIMIZERS = ['SGD', 'Adam']
CRITERIONS = ['crossentropy']
INPUT_TYPES = [None, 'visual', 'tactile', 'pose', 'visuotactile']
PROBLEM_TYPES = ['regression', 'reconstruction', 'seq_modeling', 'dyn_modeling']

# Models
ARCHITECTURES = ['mlp', 'cnn']
MODELS = ['mlp-vae', 'cnn-vae', 'cnn-mvae', 'regressor']
