class Polynomial:
    def __init__(self, *coefficients):
        self.coefficients = coefficients

    def calc(self, x):
        return sum(c * (x ** i) for i, c in enumerate(self.coefficients))


class LinearPolynomial(Polynomial):
    def __init__(self, a, b):
        super().__init__(a, b)


class QuadraticPolynomial(Polynomial):
    def __init__(self, a, b, c):
        super().__init__(a, b, c)


class CubicPolynomial(Polynomial):
    def __init__(self, a, b, c, d):
        super().__init__(a, b, c, d)