import numpy as np
from random import shuffle
from typing import Callable
import math


def linear_learning_rate_scheduler(schedule_coef: float, epoch_step: int) -> Callable[[float, int], float]:
    def func(init_learning_rate: float, epoch: int) -> float:
        return init_learning_rate * math.pow(schedule_coef, (1 + epoch) // epoch_step)

    return func


def exponent_learning_rate_scheduler(schedule_coef: float) -> Callable[[float, int], float]:
    def func(init_learning_rate: float, epoch: int) -> float:
        return init_learning_rate * math.exp(-schedule_coef * epoch)

    return func


def stochastic_gradient_descent(dots: np.ndarray, batch_size: int, start_value: np.ndarray,
                                init_learning_rate: float, learning_rate_scheduler: Callable[[float, int], float]):
    # y = a * x + b
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    for epoch in range(50):
        learning_rate = learning_rate_scheduler(init_learning_rate, epoch)
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0, 0.0])
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * prev_args[0] - dot[1] + prev_args[1])
                grad[1] += 2 * (prev_args[1] - dot[1] + prev_args[0] * dot[0])
            new_args = prev_args - learning_rate * grad
            way.append(new_args)
            prev_args = new_args
    return np.array(way)
