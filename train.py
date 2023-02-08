from typing import Any, NamedTuple
from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import dataset
import utils
from model import MixedCfcCell
from utils import lecun_tanh


TRAIN_BATCH_SIZE = flags.DEFINE_integer('train_batch_size', 32, '')
EVAL_BATCH_SIZE = flags.DEFINE_integer('eval_batch_size', 1000, '')
SEQUENCE_LENGTH = flags.DEFINE_integer('sequence_length', 128, '')
HIDDEN_SIZE = flags.DEFINE_integer('hidden_size', 256, '')
SAMPLE_LENGTH = flags.DEFINE_integer('sample_length', 128, '')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.0005, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100_000, '')
EVALUATION_INTERVAL = flags.DEFINE_integer('evaluation_interval', 100, '')
SAMPLING_INTERVAL = flags.DEFINE_integer('sampling_interval', 100, '')
SEED = flags.DEFINE_integer('seed', 42, '')


class LoopValues(NamedTuple):
  tokens: jnp.ndarray
  state: Any
  rng_key: jnp.ndarray


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState


def make_network() -> hk.RNNCore:
  model = hk.DeepRNN([
      lambda x: jax.nn.one_hot(x, num_classes=dataset.NUM_CHARS),
      MixedCfcCell(
        HIDDEN_SIZE.value,
        backbone_units=64,
        backbone_activation=lecun_tanh
      )
  ])
  return model


def make_optimizer() -> optax.GradientTransformation:
  return optax.adam(LEARNING_RATE.value)


def sequence_loss(batch: dataset.Batch) -> jnp.ndarray:
  core = make_network()
  sequence_length, batch_size = batch['input'].shape
  initial_state = core.initial_state(batch_size)
  logits, _ = hk.dynamic_unroll(core, batch['input'], initial_state)
  log_probs = jax.nn.log_softmax(logits)
  one_hot_labels = jax.nn.one_hot(batch['target'], num_classes=logits.shape[-1])
  return -jnp.sum(one_hot_labels * log_probs) / (sequence_length * batch_size)


@jax.jit
def update(state: TrainingState, batch: dataset.Batch) -> TrainingState:
  _, optimizer = make_optimizer()
  _, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
  gradients = jax.grad(loss_fn)(state.params, batch)
  updates, new_opt_state = optimizer(gradients, state.opt_state)
  new_params = optax.apply_updates(state.params, updates)
  return TrainingState(params=new_params, opt_state=new_opt_state)


def sample(
    rng_key: jnp.ndarray,
    context: jnp.ndarray,
    sample_length: int,
) -> jnp.ndarray:
  assert context.ndim == 1
  core = make_network()

  def body_fn(t: int, v: LoopValues) -> LoopValues:
    token = v.tokens[t]
    next_logits, next_state = core(token, v.state)
    key, subkey = jax.random.split(v.rng_key)
    next_token = jax.random.categorical(subkey, next_logits, axis=-1)
    new_tokens = v.tokens.at[t + 1].set(next_token)
    return LoopValues(tokens=new_tokens, state=next_state, rng_key=key)

  logits, state = hk.dynamic_unroll(core, context, core.initial_state(None))
  key, subkey = jax.random.split(rng_key)
  first_token = jax.random.categorical(subkey, logits[-1])
  tokens = jnp.zeros(sample_length, dtype=np.int32)
  tokens = tokens.at[0].set(first_token)
  initial_values = LoopValues(tokens=tokens, state=state, rng_key=key)
  values: LoopValues = lax.fori_loop(0, sample_length, body_fn, initial_values)

  return values.tokens


def main(_):
  flags.FLAGS.alsologtostderr = True

  train_data = dataset.load(
      tfds.Split.TRAIN,
      batch_size=TRAIN_BATCH_SIZE.value,
      sequence_length=SEQUENCE_LENGTH.value)

  eval_data = {
      split: dataset.load(
          split,
          batch_size=EVAL_BATCH_SIZE.value,
          sequence_length=SEQUENCE_LENGTH.value)
      for split in [tfds.Split.TRAIN, tfds.Split.TEST]
  }

  params_init, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
  _, sample_fn = hk.without_apply_rng(hk.transform(sample))
  opt_init, _ = make_optimizer()

  loss_fn = jax.jit(loss_fn)
  sample_fn = jax.jit(sample_fn, static_argnums=[3])

  rng = hk.PRNGSequence(SEED.value)
  initial_params = params_init(next(rng), next(train_data))
  initial_opt_state = opt_init(initial_params)
  state = TrainingState(params=initial_params, opt_state=initial_opt_state)

  for step in range(TRAINING_STEPS.value + 1):
    train_batch = next(train_data)
    state = update(state, train_batch)

    if step % SAMPLING_INTERVAL.value == 0:
      context = train_batch['input'][:, 0]
      assert context.ndim == 1
      rng_key = next(rng)
      samples = sample_fn(state.params, rng_key, context, SAMPLE_LENGTH.value)

      prompt = dataset.decode(context)
      continuation = dataset.decode(samples)

      logging.info('Prompt: %s', prompt)
      logging.info('Continuation: %s', continuation)

    if step % EVALUATION_INTERVAL.value == 0:
      for split, ds in eval_data.items():
        eval_batch = next(ds)
        loss = loss_fn(state.params, eval_batch)
        logging.info({
            'step': step,
            'loss': float(loss),
            'split': split,
        })

if __name__ == '__main__':
  app.run(main)