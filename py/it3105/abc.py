from abc import ABC, abstractmethod
from typing import Optional, Generator, Any

from itertools import islice, cycle
from random import sample, shuffle


def split(list, at):
    return (list[:at], list[at:])


def cycle_reshuffle(list):
    while True:
        yield from list
        shuffle(list)


class Dataset(ABC):
    @abstractmethod
    def nth_case(self, n) -> Any:
        pass

    @property
    @abstractmethod
    def size(self) -> Optional[int]:
        pass

    def split(self, casef, valf, testf):
        shuffled_cases = sample(range(self.size), self.size)
        cases_len = int(self.size * casef)

        training = shuffled_cases[:cases_len]
        validations, training = split(training, int(cases_len * valf))
        tests, training = split(training, int(cases_len * testf))

        return (
            (self.nth_case(n) for n in cycle_reshuffle(training)),
            (self.nth_case(n) for n in validations),
            (self.nth_case(n) for n in tests)
        )

    def stream_cases(self):
        cases = range(self.size)

        for n in cycle(cases):
            yield self.nth_case(n)

    def stream_shuffled_cases(self):
        shuffled_cases = sample(range(self.size), self.size)

        for n in cycle_reshuffle(shuffled_cases):
            yield self.nth_case(n)
