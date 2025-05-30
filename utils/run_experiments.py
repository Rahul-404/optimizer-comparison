from utils.train import train_model
import tensorflow as tf

datasets = ['mnist', 'fashion_mnist', 'cifar10']
model_sizes = ['small', 'medium', 'large']
optimizers = {
    'SGD': tf.keras.optimizers.SGD(),
    'Adam': tf.keras.optimizers.Adam(),
    'RMSprop': tf.keras.optimizers.RMSprop()
}

for dataset in datasets:
    for size in model_sizes:
        for name, opt in optimizers.items():
            print(f"\n--- Training {size} model on {dataset} with {name} ---")
            train_model(model_size=size, dataset_name=dataset, optimizer=opt)
