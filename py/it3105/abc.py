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
        cases = int(self.size * casef)

        training = sample(range(cases), cases)
        validations, training = split(training, int(cases * valf))
        tests, training = split(training, int(cases * testf))

        return (
            (self.nth_case(n) for n in cycle_reshuffle(training)),
            (self.nth_case(n) for n in validations),
            (self.nth_case(n) for n in tests)
        )
