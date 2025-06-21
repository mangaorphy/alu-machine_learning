#!/usr/bin/env python3
"""
Defines a class for Exponential distribution.
"""
import math


class Exponential:
    """
    Represents an exponential distribution.

    Attributes:
        lambtha (float): The rate parameter of the distribution (inverse of the mean).
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential instance.

        Args:
            data (list, optional): Data to estimate the distribution. Defaults to None.
            lambtha (float, optional): Rate parameter of the distribution. Defaults to 1.

        Raises:
            ValueError: If lambtha is not positive or data has insufficient values.
            TypeError: If data is not a list.
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the inverse of the mean of data
            self.lambtha = 1.0 / (sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Calculates the Probability Density Function (PDF) for a given time period.

        Args:
            x (float): Time period (must be ≥ 0).

        Returns:
            float: PDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0.0
        return self.lambtha * math.exp(-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the Cumulative Distribution Function (CDF) for a given time period.

        Args:
            x (float): Time period (must be ≥ 0).

        Returns:
            float: CDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.lambtha * x)
