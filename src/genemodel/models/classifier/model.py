import jax
from ..linear import LinearModel

from .library import model, loss_function, update, accuracy, predict


class ClassModel(LinearModel):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        bias: bool = False,
        seed: int | None = None,
        regularisation: float = 0,
        learning_rate: float = 0.1,
        epochs: int = 1,
        dtype: jax.numpy.dtype = jax.numpy.single,
    ):
        super().__init__(
            num_features, num_labels, bias, seed, regularisation, learning_rate, dtype
        )
        self.epochs = epochs

    def feedforward(self, predictors) -> jax.Array:
        return model(self.parameters, predictors)

    def __call__(self, *args, **kwargs):
        return predict(self.parameters, *args, **kwargs)

    def loss(
        self,
        predictors: jax.Array,
        predictees: jax.Array,
    ) -> jax.Array:
        return loss_function(
            self.parameters,
            predictors,
            predictees,
            self.regularisation,
        )

    def accuracy(
        self,
        predictors: jax.Array,
        predictees: jax.Array,
    ) -> jax.Array:
        return accuracy(
            self.parameters,
            predictors,
            predictees,
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

    def fit(self, predictors: jax.Array, predictees: jax.Array):
        history = {"loss": [], "accuracy": []}

        for _ in range(self.epochs):
            loss = self.loss(predictors, predictees).item()
            accuracy = self.accuracy(predictors, predictees).item()
            history["loss"].append(loss)
            history["accuracy"].append(accuracy)

            self.update(predictors, predictees)
            print(f"Loss: {loss} || Accuracy: {accuracy}")

        return history
