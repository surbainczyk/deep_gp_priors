from .baryrat_copy import brasil


class RationalApproximation:
    def __init__(self, interval, k=5):
        self.k = k

        self.interval = interval

    def compute_rat_function(self, fraction):
        true_f = lambda x: x ** fraction
        self.rat_function, info = brasil(true_f, self.interval, self.k, info=True)
        if not info[0]:
            raise Warning("BRASIL algorithm did not converge.")
        
        self.rat_function_inv = self.rat_function.reciprocal()

    def get_rational_approx_coeffs(self):
        poles, residues = self.rat_function.polres()
        gain = self.rat_function.gain()

        return gain, residues, poles

    def get_rational_approx_inv_coeffs(self):
        poles, residues = self.rat_function_inv.polres()
        gain = self.rat_function_inv.gain()

        return gain, residues, poles
