from numpy import log, sqrt, exp
from  scipy.stats import norm


class BsmModel:
    def __init__(self, option_type, price, strike, interest_rate, expiry, volatility, dividend_yield=0):
        self.s = price # Underlying asset price
        self.k = strike # Option strike K
        self.r = interest_rate # Continuous risk fee rate
        self.q = dividend_yield # Dividend continuous rate
        self.T = expiry # time to expiry (year)
        self.sigma = volatility # Underlying volatility
        self.type = option_type # option type "p" put option "c" call option

    def n(self, d):
        # cumulative probability distribution function of standard normal distribution
        return norm.cdf(d)

    def dn(self, d):
        # the first order derivative of n(d)
        return norm.pdf(d)

    def d1(self):
        d1 = (log(self.s / self.k) + (self.r - self.q + self.sigma ** 2 * 0.5) * self.T) / (self.sigma * sqrt(self.T))
        return d1

    def d2(self):
        d2 = (log(self.s / self.k) + (self.r - self.q - self.sigma ** 2 * 0.5) * self.T) / (self.sigma * sqrt(self.T))
        return d2

    def bsm_price(self):
        d1 = self.d1()
        d2 = d1 - self.sigma * sqrt(self.T)
        if self.type == 'c':
            price = exp(-self.r*self.T) * (self.s * exp((self.r - self.q)*self.T) * self.n(d1) - self.k * self.n(d2))
            return price
        elif self.type == 'p':
            price = exp(-self.r*self.T) * (self.k * self.n(-d2) - (self.s * exp((self.r - self.q)*self.T) * self.n(-d1)))
            return price
        else:
            print ("option type can only be c or p")


        ''' Greek letters for European options on an asset that provides a yield at rate q '''

    def delta(self):
        d1 = self.d1()
        if self.type == "c":
            return exp(-self.q * self.T) * self.n(d1)
        elif self.type == "p":
            return exp(-self.q * self.T) * (self.n(d1)-1)


    def gamma(self):
        d1 = self.d1()
        dn1 = self.dn(d1)
        return dn1 * exp(-self.q * self.T) / (self.s * self.sigma * sqrt(self.T))

    def vega(self):
        d1 = self.d1()
        dn1 = self.dn(d1)
        return self.s * sqrt(self.T) * dn1 * exp(-self.q * self.T)

    def theta(self):
        d1 = self.d1()
        d2 = d1 - self.sigma * sqrt(self.T)
        dn1 = self.dn(d1)
        if self.type == "c":
            theta = -self.s * dn1 * self.sigma * exp(-self.q*self.T) / (2 * sqrt(self.T)) \
                        + self.q * self.s * self.n(d1) * exp(-self.q*self.T) \
                        - self.r * self.k * exp(-self.r*self.T) * self.n(d2)
            return theta
        elif self.type == "p":
            theta = -self.s * dn1 * self.sigma * exp(-self.q * self.T) / (2 * sqrt(self.T)) \
                        - self.q * self.s * self.n(-d1) * exp(-self.q * self.T) \
                        + self.r * self.k * exp(-self.r * self.T) * self.n(-d2)
            return theta

    def rho(self):
        d2 = self.d2()
        if self.type == "c":
            rho = self.k * self.T * (exp(-self.r*self.T)) * self.n(d2)
        elif self.type == "p":
            rho = -self.k * self.T * (exp(-self.r*self.T)) * self.n(-d2)
        return rho


if __name__ == "__main__":
    # Example
    option = BsmModel(option_type='c', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.bsm_price())

    # Example
    option = BsmModel(option_type='p', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.bsm_price())

    # Example
    option = BsmModel(option_type='c', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.delta())

    # Example
    option = BsmModel(option_type='c', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.gamma())

    # Example
    option = BsmModel(option_type='c', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.vega())

    # Example
    option = BsmModel(option_type='c', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.theta())

    # Example
    option = BsmModel(option_type='c', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.rho())

    # Example
    option = BsmModel(option_type='p', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.delta())

    # Example
    option = BsmModel(option_type='p', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.gamma())

    # Example
    option = BsmModel(option_type='p', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)
    print (option.vega())

    # Example
    option = BsmModel(option_type='p', price=100, strike=100, interest_rate=0.05, expiry=1, volatility=0.2)