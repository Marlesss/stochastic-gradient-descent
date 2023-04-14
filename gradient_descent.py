import numpy as np
from random import shuffle
from typing import Callable
import math

EPS = 10 ** -8


def constant_learning_rate():
    def func(init_learning_rate: float, epoch: int):
        return init_learning_rate

    return func


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
            #     sum( (dot[1] - (func_arg[0] * dot[0] + func_arg[1])) ** 2 for dot in dots)
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * prev_args[0] - dot[1] + prev_args[1])
                grad[1] += 2 * (prev_args[1] - dot[1] + prev_args[0] * dot[0])
            new_args = prev_args - learning_rate * grad
            way.append(new_args)
            prev_args = new_args
    return np.array(way)


def running_average_stochastic_gradient_descent_constant(dots: np.ndarray, batch_size: int,
                                                         start_value: np.ndarray,
                                                         learning_rate: float,
                                                         gamma: float):
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    prev_v = np.array([0.0, 0.0])
    for epoch in range(50):
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0, 0.0])
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * prev_args[0] - dot[1] + prev_args[1])
                grad[1] += 2 * (prev_args[1] - dot[1] + prev_args[0] * dot[0])

            v = gamma * prev_v + (1 - gamma) * grad

            new_args = prev_args - learning_rate * v

            way.append(new_args)
            prev_args = new_args
            prev_v = v
    return np.array(way)


def nesterov_stochastic_gradient_descent_constant(dots: np.ndarray, batch_size: int,
                                                  start_value: np.ndarray,
                                                  learning_rate: float,
                                                  gamma: float):
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    prev_v = np.array([0.0, 0.0])
    for epoch in range(50):
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0, 0.0])
            grad_arg = prev_args - learning_rate * gamma * prev_v
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * grad_arg[0] - dot[1] + grad_arg[1])
                grad[1] += 2 * (grad_arg[1] - dot[1] + grad_arg[0] * dot[0])

            v = gamma * prev_v + (1 - gamma) * grad

            new_args = prev_args - learning_rate * v

            way.append(new_args)
            prev_args = new_args
            prev_v = v
    return np.array(way)


def adagrad_stochastic_gradient_descent_constant(dots: np.ndarray, batch_size: int,
                                                 start_value: np.ndarray,
                                                 learning_rate: float):
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    G = np.array([0.0, 0.0])
    for epoch in range(50):
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0, 0.0])
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * prev_args[0] - dot[1] + prev_args[1])
                grad[1] += 2 * (prev_args[1] - dot[1] + prev_args[0] * dot[0])

            G += grad ** 2

            new_args = prev_args - learning_rate * ((G + EPS) ** (-1 / 2)) * grad

            way.append(new_args)
            prev_args = new_args
    return np.array(way)


def rmsprop_stochastic_gradient_descent_constant(dots: np.ndarray, batch_size: int,
                                                 start_value: np.ndarray,
                                                 learning_rate: float, beta: float):
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    prev_s = np.array([0.0, 0.0])
    for epoch in range(50):
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0, 0.0])
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * prev_args[0] - dot[1] + prev_args[1])
                grad[1] += 2 * (prev_args[1] - dot[1] + prev_args[0] * dot[0])

            s = beta * prev_s + (1 - beta) * (grad ** 2)

            new_args = prev_args - learning_rate * grad * ((s + EPS) ** (-1 / 2))

            way.append(new_args)
            prev_args = new_args
            prev_s = s
    return np.array(way)


def adam_stochastic_gradient_descent_constant(dots: np.ndarray, batch_size: int,
                                              start_value: np.ndarray,
                                              learning_rate: float, beta1: float, beta2: float):
    way = [start_value]

    dots_i = list(range(len(dots)))
    prev_args = start_value
    prev_v = np.array([0.0, 0.0])
    prev_s = np.array([0.0, 0.0])
    beta1_powered = beta1
    beta2_powered = beta2
    for epoch in range(50):
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0, 0.0])
            for dot in minibatch_dots:
                grad[0] += 2 * dot[0] * (dot[0] * prev_args[0] - dot[1] + prev_args[1])
                grad[1] += 2 * (prev_args[1] - dot[1] + prev_args[0] * dot[0])

            v = beta1 * prev_v + (1 - beta1) * grad
            s = beta2 * prev_s + (1 - beta2) * (grad ** 2)

            new_args = (prev_args -
                        learning_rate * (v / (1 - beta1_powered)) * (s / (1 - beta2_powered) + EPS) ** (-1 / 2))

            way.append(new_args)
            prev_args = new_args
            prev_v = v
            prev_s = s
            beta1_powered *= beta1
            beta2_powered *= beta2
    return np.array(way)
