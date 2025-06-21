#!/usr/bin/env python3
"""Poisson distribution class"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates PMF at k"""
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        lambtha = self.lambtha

        def factorial(n):
            return 1 if n == 0 else n * factorial(n - 1)

        return (lambtha ** k * e ** -lambtha) / factorial(k)

    def cdf(self, k):
        """Calculates CDF at k"""
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        lambtha = self.lambtha

        def factorial(n):
            return 1 if n == 0 else n * factorial(n - 1)

        cdf = 0
        for i in range(k + 1):
            cdf += (lambtha ** i * e ** -lambtha) / factorial(i)
        return cdf
