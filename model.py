import jax
import jax.numpy as jnp
from jax.numpy import abs, exp
import haiku as hk
from typing import Callable, Optional, Tuple
from utils import lecun_tanh, add_batch


class CfcCell(hk.RNNCore):
    def __init__(
            self,
            hidden_size: int,
            backbone_layers: int = 1,
            backbone_units: int = 128,
            backbone_activation: Callable = jax.nn.silu,
            dropout: Optional[float] = None,
            no_gate: bool = False,
            minimal: bool = False,
            **kwargs
    ):
        super(CfcCell, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._activation = backbone_activation
        self._no_gate = no_gate
        self._minimal = minimal
        self._dropout = dropout

        init = hk.initializers.VarianceScaling(
            1.0, 'fan_avg', 'uniform'
        )

        self._backbone = []
        for i in range(backbone_layers):
            self._backbone.append(
                hk.Linear(backbone_units, init)
            )

        if self._minimal:
            self._ff1 = hk.Linear(self._hidden_size, w_init=init)
            self._tau = hk.get_parameter(
                'tau',
                (1, self._hidden_size),
                jnp.zeros
            )
            self._A = hk.get_parameter(
                'A',
                (1, self._hidden_size),
                jnp.ones
            )
        else:
            self._ff1 = hk.Linear(self._hidden_size, w_init=init)
            self._ff2 = hk.Linear(self._hidden_size, w_init=init)
            self._time_a = hk.Linear(self._hidden_size, w_init=init)
            self._time_b = hk.Linear(self._hidden_size, w_init=init)


    def __call__(self, inputs, prev_state, **kwargs):
        t = 1.0
        x = jnp.concatenate([inputs, prev_state], -1)

        for layer in self._backbone:
            x = self._activation(layer(x))
            if self._dropout is not None:
                x = hk.dropout(hk.next_rng_key(), self._dropout, x)

        ff1 = self._ff1(x)

        if self._minimal:
            hidden_state = (
                    -self._A
                    * exp(-t * (abs(self._tau) + abs(ff1)))
                    * ff1
                    + self._A
            )
        else:
            ff2 = lecun_tanh(self._ff2(x))
            t_a = self._time_a(x)
            t_b = self._time_b(x)
            t_interp = jax.nn.sigmoid(-t_a * t + t_b)

            if self._no_gate:
                hidden_state = ff1 + t_interp * ff2
            else:
                hidden_state = ff1 * (1.0 - t_interp) + t_interp * ff2

        return hidden_state, hidden_state


    def initial_state(self, batch_size: Optional[int]):
        state = jnp.zeros([self._hidden_size])
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state

class MixedCfcCell(hk.RNNCore):
    def __init__(
            self,
            hidden_size: int,
            backbone_layers: int = 1,
            backbone_units: int = 128,
            backbone_activation: Callable = jax.nn.silu,
            dropout: Optional[float] = None,
            **kwargs
    ):
        super(MixedCfcCell, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._init = hk.initializers.VarianceScaling(
            1.0, 'fan_avg', 'uniform'
        )
        self._recurrent_init = hk.initializers.Orthogonal()
        self._cfc = CfcCell(self._hidden_size, backbone_layers, backbone_units, backbone_activation, dropout)


    def __call__(
            self,
            inputs: jnp.ndarray,
            state: hk.LSTMState,
            **kwargs
    ) -> Tuple[jnp.ndarray, hk.LSTMState]:

        if len(inputs.shape) > 2 or not inputs.shape:
            raise ValueError("LSTM input must be rank-1 or rank-2.")

        x = hk.Linear(4*self._hidden_size, w_init=self._init)(inputs)
        h = hk.Linear(4*self._hidden_size, with_bias=False,
                      w_init=self._recurrent_init)(state.hidden)
        z = x + h

        i, g, f, o = jnp.split(z, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)
        c = f * state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        ode_input = jax.nn.sigmoid(o) * jnp.tanh(c)

        ode_output, ode_state = self._cfc(ode_input, state.hidden)
        return ode_output, hk.LSTMState(ode_state, c)


    def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
        state = hk.LSTMState(
            hidden=jnp.zeros([self._hidden_size]),
            cell=jnp.zeros([self._hidden_size])
        )
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state
