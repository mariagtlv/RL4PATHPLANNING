import logging

import numpy as np
import tensorflow as tf

from Utils.child_network import ChildCNN
from Utils.cifar10_processor import get_tf_datasets_from_numpy
from Utils.config import child_network_params, controller_params

import pandas as pd
import time
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    
    Args:
        values (list): A list of rewards 
    Returns:
        (float) The last value of the exponential moving average
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]

def fgsm_attack(x, loss, eps=0.03):
    grad = tf.gradients(loss, x)[0]
    signed_grad = tf.sign(grad)
    x_adv = tf.stop_gradient(x + eps * signed_grad)
    return tf.clip_by_value(x_adv, 0.0, 1.0)


class Controller(object):

    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.num_cell_outputs = controller_params['components_per_layer'] * controller_params['max_layers']
        self.reward_history = []
        self.architecture_history = []
        self.divison_rate = 100

        self.training_metrics = []
        self.search_metrics = [] 

        with self.graph.as_default():
            self.build_controller()


    def network_generator(self, nas_cell_hidden_state):
        # number of output units we expect from a NAS cell
        with tf.name_scope('network_generator'):
            nas = tf.contrib.rnn.NASCell(self.num_cell_outputs)
            network_architecture, nas_cell_hidden_state = tf.nn.dynamic_rnn(nas, tf.expand_dims(
                nas_cell_hidden_state, -1), dtype=tf.float32)
            bias_variable = tf.Variable([0.01] * self.num_cell_outputs)
            network_architecture = tf.nn.bias_add(network_architecture, bias_variable)
            return network_architecture[:, -1:, :]

    def generate_child_network(self, child_network_architecture):
        with self.graph.as_default():
            return self.sess.run(self.cnn_dna_output, {self.child_network_architectures: child_network_architecture})

    def build_controller(self):
        logger.info('Building controller network')
        # Build inputs and placeholders
        with tf.name_scope('controller_inputs'):
            # Input to the NASCell
            self.child_network_architectures = tf.placeholder(tf.float32, [None, self.num_cell_outputs], 
                                                              name='controller_input')
            # Discounted rewards
            self.discounted_rewards = tf.placeholder(tf.float32, (None, ), name='discounted_rewards')

        # Build controller
        with tf.name_scope('network_generation'):
            with tf.variable_scope('controller'):
                self.controller_output = tf.identity(self.network_generator(self.child_network_architectures), 
                                                     name='policy_scores')
                self.cnn_dna_output = tf.cast(tf.scalar_mul(self.divison_rate, self.controller_output), tf.int32,
                                              name='controller_prediction')

        # Set up optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.99, self.global_step, 500, 0.96, staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        # Gradient and loss computation
        with tf.name_scope('gradient_and_loss'):
            # Define policy gradient loss for the controller
            self.policy_gradient_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.controller_output[:, -1, :],
                labels=self.child_network_architectures))
            # L2 weight decay for Controller weights
            self.l2_loss = tf.reduce_sum(tf.add_n([tf.nn.l2_loss(v) for v in
                                                   tf.trainable_variables(scope="controller")]))
            # Add the above two losses to define total loss
            self.total_loss = self.policy_gradient_loss + self.l2_loss * controller_params["beta"]
            # Compute the gradients
            self.gradients = self.optimizer.compute_gradients(self.total_loss)

            # Gradients calculated using REINFORCE
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

        with tf.name_scope('train_controller'):
            # The main training operation. This applies REINFORCE on the weights of the Controller
            self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

        logger.info('Successfully built controller')


    def train_child_network(self, cnn_dna, child_id):
        """
        Trains a child network and returns reward, or the validation accuracy
        Args:
            cnn_dna (list): List of tuples representing the child network's DNA
            child_id (str): Name of child network
        Returns:
            (float) validation accuracy
        """
        logger.info("Training with dna: {}".format(cnn_dna))
        child_graph = tf.Graph()
        with child_graph.as_default():
            sess = tf.Session()

            child_network = ChildCNN(cnn_dna=cnn_dna, child_id=child_id, **child_network_params)

            start_time = time.time()

            # Create input pipeline
            train_dataset, valid_dataset, test_dataset, num_train_batches, num_valid_batches, num_test_batches = \
                get_tf_datasets_from_numpy(batch_size=child_network_params["batch_size"])

            # Generic iterator
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            next_tensor_batch = iterator.get_next()

            # Separate train and validation set init ops
            train_init_ops = iterator.make_initializer(train_dataset)
            valid_init_ops = iterator.make_initializer(valid_dataset)

            # Build the graph
            input_tensor, labels = next_tensor_batch

            # Build the child network, which returns the pre-softmax logits of the child network
            logits = child_network.build(input_tensor)
            
            # Define the loss function for the child network
            loss_ops = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name="loss")

            # Define the training operation for the child network
            train_ops = tf.train.AdamOptimizer(learning_rate=child_network_params["learning_rate"]).minimize(loss_ops)

            # The following operations are for calculating the accuracy of the child network
            pred_ops = tf.nn.softmax(logits, name="preds")
            correct = tf.equal(tf.argmax(pred_ops, 1), tf.argmax(labels, 1), name="correct")
            accuracy_ops = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

            # FGSM adversarial examples
            x_adv = fgsm_attack(input_tensor, loss_ops)
            logits_adv = child_network.build(x_adv)

            pred_adv = tf.nn.softmax(logits_adv)
            correct_adv = tf.equal(tf.argmax(pred_adv, 1), tf.argmax(labels, 1))
            adv_accuracy_ops = tf.reduce_mean(tf.cast(correct_adv, tf.float32))


            initializer = tf.global_variables_initializer()

            # Training
            sess.run(initializer)
            sess.run(train_init_ops)

            logger.info("Training child CNN {} for {} epochs".format(child_id, child_network_params["max_epochs"]))
            for epoch_idx in range(child_network_params["max_epochs"]):
                avg_loss, avg_acc = [], []
                epoch_preds = []
                epoch_labels = []

                for batch_idx in range(num_train_batches):
                    loss, _, accuracy, preds, labs = sess.run(
                            [loss_ops, train_ops, accuracy_ops,
                            tf.argmax(pred_ops, 1),
                            tf.argmax(labels, 1)]
                        )
                    avg_loss.append(loss)
                    avg_acc.append(accuracy)
                    epoch_preds.extend(preds)
                    epoch_labels.extend(labs)

                train_f1 = f1_score(epoch_labels, epoch_preds, average="macro")
                logger.info("\tEpoch {}:\tloss - {:.6f}\taccuracy - {:.3f}".format(epoch_idx,
                                                                                   np.mean(avg_loss), np.mean(avg_acc)))
                self.training_metrics.append({
                    "timestamp": pd.Timestamp.now(),
                    "epoch": "final",
                    "genotype_full": str(cnn_dna),
                    "val_accuracy": float(np.mean(avg_val_acc)),
                    "adv_accuracy": float(np.mean(avg_adv_acc)),
                    "val_f1": float(val_f1),
                    "adv_f1": float(adv_f1),
                    "reward": float(reward),
                    "total_time_sec": time.time() - start_time
                })


                pd.DataFrame(self.training_metrics).to_parquet(
                    "child_training_results.parquet", index=False
                )

            # Validate and return reward
            logger.info("Finished training, now calculating validation accuracy")
            sess.run(valid_init_ops)
            avg_val_loss, avg_val_acc = [], []

            all_preds = []
            all_labels = []

            avg_val_acc = []
            avg_adv_acc = []

            all_preds = []
            all_labels = []
            all_adv_preds = []

            for batch_idx in range(num_valid_batches):
                val_acc, adv_acc, preds, labs, adv_preds = sess.run(
                    [
                        accuracy_ops,
                        adv_accuracy_ops,
                        tf.argmax(pred_ops, 1),
                        tf.argmax(labels, 1),
                        tf.argmax(pred_adv, 1)
                    ]
                )

                avg_val_acc.append(val_acc)
                avg_adv_acc.append(adv_acc)

                all_preds.extend(preds)
                all_labels.extend(labs)
                all_adv_preds.extend(adv_preds)

            logger.info("Valid loss - {:.6f}\tValid accuracy - {:.3f}".format(np.mean(avg_val_loss),
                                                                              np.mean(avg_val_acc)))
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            adv_f1 = f1_score(all_labels, all_adv_preds, average="macro")

            self.training_metrics.append({
                "timestamp": pd.Timestamp.now(),
                "epoch": "final",
                "genotype_full": str(cnn_dna),
                "val_accuracy": float(np.mean(avg_val_acc)),
                "clean_loss": float(np.mean(avg_val_loss)),
                "f1": float(val_f1),
                "total_time_sec": time.time() - start_time
            })

            pd.DataFrame(self.training_metrics).to_parquet(
                "child_training_results.parquet", index=False
            )

        alpha = 0.5  # trade-off clean vs robust
        reward = alpha * np.mean(avg_val_acc) + (1 - alpha) * np.mean(avg_adv_acc)
        return reward


    def train_controller(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        step = 0
        total_rewards = 0
        child_network_architecture = np.array([[10.0, 128.0, 1.0, 1.0] *
                                               controller_params['max_layers']], dtype=np.float32)

        for episode in range(controller_params['max_episodes']):
            logger.info('=============> Episode {} for Controller'.format(episode))
            step += 1
            episode_reward_buffer = []

            for sub_child in range(controller_params["num_children_per_episode"]):
                # Generate a child network architecture
                child_network_architecture = self.generate_child_network(child_network_architecture)[0]

                if np.any(np.less_equal(child_network_architecture, 0.0)):
                    reward = -1.0
                else:
                    reward = self.train_child_network(cnn_dna=child_network_architecture,
                                                      child_id='child/{}'.format("{}_{}".format(episode, sub_child)))
                episode_reward_buffer.append(reward)

            mean_reward = np.mean(episode_reward_buffer)

            self.search_metrics.append({
                "timestamp": pd.Timestamp.now(),
                "epoch": episode,
                "genotype_full": str(child_network_architecture.ravel()),
                "valid_acc": float(mean_reward)
            })


            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network_architecture)
            total_rewards += mean_reward

            child_network_architecture = np.array(self.architecture_history[-step:]).ravel() / self.divison_rate
            child_network_architecture = child_network_architecture.reshape((-1, self.num_cell_outputs))
            baseline = ema(self.reward_history)
            last_reward = self.reward_history[-1]
            rewards = [last_reward - baseline]
            logger.info("Buffers before loss calculation")
            logger.info("States: {}".format(child_network_architecture))
            logger.info("Rewards: {}".format(rewards))

            with self.graph.as_default():
                _, loss = self.sess.run([self.train_op, self.total_loss],
                                        {self.child_network_architectures: child_network_architecture,
                                         self.discounted_rewards: rewards})

            logger.info('Episode: {} | Loss: {} | DNA: {} | Reward : {}'.format(
                episode, loss, child_network_architecture.ravel(), mean_reward))
            
            pd.DataFrame(self.search_metrics).to_parquet(
                "controller_search_results.parquet", index=False
            )
