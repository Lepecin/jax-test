import jax
import numpy

import math
import warnings

warnings.simplefilter("ignore", ResourceWarning)


def init_parameters(
    num_features: int,
    num_labels: int,
    bias: bool = False,
    seed: int | None = None,
    dtype: jax.numpy.dtype = jax.numpy.double,
) -> dict[str, jax.Array]:
    bound = 1 / math.sqrt(num_features)
    generator = numpy.random.default_rng(seed)
    weights: dict[str, jax.Array] = {}

    w = generator.uniform(-bound, bound, (num_features, num_labels))
    weights["weights"] = jax.numpy.array(w, dtype)
    if bias:
        b = generator.uniform(-bound, bound, (1, num_labels))
        weights["bias"] = jax.numpy.array(b, dtype)
    return weights


@jax.jit
def model(
    parameters: dict[str, jax.Array],
    predictors: jax.Array,
) -> jax.Array:
    weights = parameters["weights"]
    bias = parameters["bias"]

    output = predictors.dot(weights) + bias
    return output


@jax.jit
def ridge_regulariser(parameters: dict[str, jax.Array]) -> jax.Array:
    def frobenius(array: jax.Array) -> jax.Array:
        return jax.numpy.power(array, 2).sum()

    zero = jax.numpy.zeros(())

    return sum(
        jax.tree_util.tree_leaves(jax.tree_util.tree_map(frobenius, parameters)), zero
    )


@jax.jit
def loss_function(
    parameters: dict[str, jax.Array],
    predictors: jax.Array,
    predictees: jax.Array,
    regulariser: float = 0.0,
) -> jax.Array:
    predictions = model(parameters, predictors)
    loss = jax.numpy.mean((predictions - predictees) ** 2)
    return loss + regulariser * ridge_regulariser(parameters)


@jax.jit
def update(
    parameters: dict[str, jax.Array],
    predictors: jax.Array,
    predictees: jax.Array,
    regulariser: float = 0.0,
    lr: float = 0.1,
) -> dict[str, jax.Array]:
    gradients = jax.grad(loss_function)(parameters, predictors, predictees, regulariser)

    def sgd(parameters, gradients):
        return parameters - gradients * lr

    return jax.tree_util.tree_map(sgd, parameters, gradients)
