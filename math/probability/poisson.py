#!/usr/bin/env python3
"""
Defines a class for Poisson distribution.
"""
import math


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson instance.

        Args:
            data (list, optional): Data to estimate the distribution. Defaults to None.
            lambtha (float, optional): Expected number of occurrences. Defaults to 1.

        Raises:
            ValueError: If lambtha is not positive or data has insufficient values.
            TypeError: If data is not a list.
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF) for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: PMF value for k, or 0 if k is out of range.
        """
        if not isinstance(k, int):
            k = int(k)  # Convert to integer if not already

        if k < 0:
            return 0.0  # k out of range (Poisson is defined for k >= 0)

        # Calculate PMF using Poisson formula: (e^-λ * λ^k) / k!
        return (math.exp(-self.lambtha) * (self.lambtha ** k)) / \
            math.factorial(k)

    def cdf(self, k):
        """
        Calculates the Cumulative Distribution Function (CDF) for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: CDF value for k, or 0 if k is out of range.
        """
        if not isinstance(k, int):
            k = int(k)  # Convert to integer if not already

        if k < 0:
            return 0.0  # k out of range (Poisson is defined for k >= 0)

        # Calculate CDF by summing PMFs from 0 to k
        cdf_value = 0.0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
