import jax

from ..linear.library import model as linear_model, ridge_regulariser


@jax.jit
def model(parameters, predictors: jax.Array) -> jax.Array:
    output = linear_model(parameters, predictors)
    return jax.nn.softmax(output, axis=1)


@jax.jit
def loss_function(
    parameters,
    predictors: jax.Array,
    predictees: jax.Array,
    regulariser: float = 0.0,
) -> jax.Array:
    output = model(parameters, predictors)

    logits = -jax.numpy.log(output)
    indices = predictees.flatten() - 1
    obs_range = jax.numpy.arange(0, len(output))
    loss = logits[obs_range, indices].mean()

    return loss + regulariser * ridge_regulariser(parameters)


@jax.jit
def update(
    parameters,
    predictors: jax.Array,
    predictees: jax.Array,
    regulariser: float = 0.0,
    lr: float = 0.1,
):
    gradients = jax.grad(loss_function)(
        parameters,
        predictors,
        predictees,
        regulariser,
    )

    def sgd(parameters, gradients):
        return parameters - gradients * lr

    return jax.tree_util.tree_map(sgd, parameters, gradients)


@jax.jit
def predict(
    parameters,
    predictors: jax.Array,
) -> jax.Array:
    output = model(parameters, predictors)
    return jax.numpy.argmax(output, axis=1, keepdims=True) + 1


@jax.jit
def accuracy(
    parameters,
    predictors: jax.Array,
    predictees: jax.Array,
) -> jax.Array:
    preds = predict(parameters, predictors).flatten()
    labels = predictees.flatten()
    return jax.numpy.equal(preds, labels).mean()
