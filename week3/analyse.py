import tensorflow as tf
import cifar10


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(cifar10.FLAGS.train_dir):
        tf.gfile.DeleteRecursively(cifar10.FLAGS.train_dir)
    tf.gfile.MakeDirs(cifar10.FLAGS.train_dir)
    train()


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()


...


if __name__ == '__main__':
    tf.app.run()