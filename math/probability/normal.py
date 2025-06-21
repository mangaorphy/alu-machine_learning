#!/usr/bin/env python3
"""
Defines a class for Normal distribution.
"""

import math


class Normal:
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            # Use provided mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Calculate from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = sum(data) / len(data)

            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = math.sqrt(variance)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value

        parameters:
            x: x-value

        return:
            the PDF value for x
        """
        mean = self.mean
        stddev = self.stddev
        e = 2.7182818285
        pi = 3.1415926536
        power = -0.5 * (self.z_score(x) ** 2)
        coefficient = 1 / (stddev * ((2 * pi) ** (1 / 2)))
        pdf = coefficient * (e ** power)
        return pdf

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value

        parameters:
            x: x-value

        return:
            the CDF value for x
        """
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        erf = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        erf = erf - ((value ** 7) / 42) + ((value ** 9) / 216)
        erf *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + erf)
        return cdf
