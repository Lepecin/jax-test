import jax

from .library import model, init_parameters, loss_function, update


class LinearModel:
    parameters: dict[str, jax.Array]
    regularisation: float
    learning_rate: float

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        bias: bool = False,
        seed: int | None = None,
        regularisation: float = 0.0,
        learning_rate: float = 0.1,
        dtype: jax.numpy.dtype = jax.numpy.single,
    ):
        self.parameters = init_parameters(
            num_features,
            num_labels,
            bias,
            seed,
            dtype,
        )
        self.regularisation = regularisation
        self.learning_rate = learning_rate

    def feedforward(self, predictors) -> jax.Array:
        return model(self.parameters, predictors)

    def __call__(self, *args, **kwargs):
        return self.feedforward(*args, **kwargs)

    def loss(self, predictors: jax.Array, predictees: jax.Array) -> jax.Array:
        return loss_function(
            self.parameters,
            predictors,
            predictees,
            self.regularisation,
        )

    def update(self, predictors: jax.Array, predictees: jax.Array):
        self.parameters = update(
            self.parameters,
            predictors,
            predictees,
            self.regularisation,
            self.learning_rate,
        )
        return self

    def get_parameters(self):
        return self.parameters
