import numpy as np
from random import shuffle


def stochastic_gradient_descent_constant(dots: np.ndarray, batch_size: int, start_value: np.ndarray,
                                         learning_rate: float):
    # y = a * x + b
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    for _ in range(20):
        shuffle(dots_i)
        for epoch in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[epoch * batch_size: (epoch + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            # need gradient of sum((dot[1] - (prev_args[0] * dot[0] + prev_args[1])) ** 2 for dot in minibatch)
            grad = np.array([0, 0])
            for dot in minibatch_dots:
                grad[0] += 2 * (dot[1] - (prev_args[0] * dot[0] + prev_args[1])) * (-dot[0])
                grad[1] += 2 * (dot[1] - (prev_args[0] * dot[0] + prev_args[1]))
            new_args = prev_args - learning_rate * grad
            way.append(new_args)
            prev_args = new_args
    return np.array(way)
