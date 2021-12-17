"""
Grading 10/10 Excellent Job!
"""

# Section 1. Preparation. Import the necessary libraries
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

# Section 2. Define the Growth model as a class
# attributes: para_dict, state_dict
# ----------------------------------------------------------------------------
# para_dict:
# population growth rate, growth rate of labor efficiency, saving rate,
# share of capital, depreciation rate
# ----------------------------------------------------------------------------
# state_dict:
#  employment rate, factor productivity, capital, population, output


class SolowModel:
    state_history = []

    def __init__(self, para_dict, state_dict):
        self.para_dict = para_dict
        self.state_dict = state_dict

        self.state_dict['y'] = self.para_dict['a'] * \
            self.state_dict['k']**self.para_dict['alpha']
        self.state_dict['K'] = self.state_dict['k'] * self.state_dict['L']
        self.state_dict['Y'] = self.state_dict['y'] * self.state_dict['L']
        self.state_dict['i'] = self.state_dict['y'] * self.para_dict['s']
        self.state_dict['I'] = self.state_dict['i'] * self.state_dict['L']
        self.state_dict['dk'] = np.array([0])

        self.steady_state = {}


# methods:
# check_model: print the attribute of the instance
# grwoth: take one argument "year", and drive the economic growth
# get parameters: get the model parameters
# get states: get the states variables
# plot_growth: visualize the growth model


    def check_model(self):
        pprint(self.para_dict)

    def check_state(self):
        pprint(self.state_dict)

    def growth(self, years):

        timeline = np.linspace(0, years, num=years+1, dtype=int)
        differences = []

        for t in timeline:
            n = self.para_dict.get('n')[0]
            s = self.para_dict.get('s')[0]
            delta = self.para_dict.get('delta')[0]
            alpha = self.para_dict.get('alpha')[0]
            a = self.para_dict.get('a')[0]
            g = self.para_dict.get('g')[0]

            k_t = self.state_dict.get('k')
            y_t = self.state_dict.get('y')
            K_t = self.state_dict.get('K')
            Y_t = self.state_dict.get('Y')
            L_t = self.state_dict.get('L')
            i_t = self.state_dict.get('i')
            I_t = self.state_dict.get('I')
            dk_t = self.state_dict.get('dk')

            dk = s*y_t[t] - (delta + n) * k_t[t]

            k_next = np.array(k_t[t] + dk)
            L_next = np.array(L_t[t] * (1+n))
            y_next = np.array(a*k_next**alpha)
            Y_next = np.array(y_next * L_next)
            K_next = np.array(k_next * L_next)
            i_next = np.array(s * y_next)
            I_next = np.array(i_next * L_next)

            k_t = np.append(k_t, k_next)
            y_t = np.append(y_t, y_next)
            K_t = np.append(K_t, K_next)
            Y_t = np.append(Y_t, Y_next)
            L_t = np.append(L_t, L_next)
            i_t = np.append(i_t, i_next)
            I_t = np.append(I_t, I_next)
            dk_t = np.append(dk_t, (delta + n + g) * k_t[t])

            self.state_dict['k'] = k_t
            self.state_dict['y'] = y_t
            self.state_dict['K'] = K_t
            self.state_dict['Y'] = Y_t
            self.state_dict['L'] = L_t
            self.state_dict['i'] = i_t
            self.state_dict['I'] = I_t
            self.state_dict['dk'] = dk_t

    def find_steady_state(self):
        n = self.para_dict.get('n')[0]
        s = self.para_dict.get('s')[0]
        delta = self.para_dict.get('delta')[0]
        alpha = self.para_dict.get('alpha')[0]
        a = self.para_dict.get('a')[0]

        k_t = np.linspace(0, 200, 100)

        break_even = (n + delta) * k_t
        y_t = a * k_t**alpha
        i_t = s * y_t
        compare = i_t - break_even

        steady = np.where(compare < 0)[0][0]

        y_star = y_t[steady]
        i_star = i_t[steady]
        c_star = y_t[steady] - i_t[steady]
        k_star = k_t[steady]

        self.steady_state = {
            'k_star': k_star,
            'c_star': c_star,
            'y_star': y_star,
            'i_star': i_star,
        }

        return steady

    def plot_income():
        pass

    def plot_growth(self, years):
        timeline = np.linspace(0, years, num=years+2, dtype=int)
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(211)
        ax.set_title('Growth in Total Output')
        ax.plot(timeline, self.state_dict['y'], 'b', label='Production')
        ax.set_ylabel('Y - Total Production')
        ax.set_xlabel('Period')
        bx = fig.add_subplot(212)
        bx.set_title('Growth per worker')
        bx.plot(self.state_dict['k'],
                self.state_dict['y'], 'b', label='Production')
        bx.plot(self.state_dict['k'],
                self.state_dict['i'], 'b', label='Investment')
        bx.plot(self.state_dict['k'],
                self.state_dict['dk'], 'r', label='Depreciation')
        bx.set_ylabel('y, i - production/investment per worker')
        bx.set_xlabel('k - capital per worker')
        plt.show()


# Section 3. Specify model parameters and examine economic grwoth in 10 years

parameters = {
    'n': np.array([0.002]),
    's': np.array([0.15]),
    'alpha': np.array([1/3]),
    'delta': np.array([0.05]),
    'a': np.array([1]),
    'g': np.array([0.01]),
}

initial_state = {
    'k': np.array([1]),
    'L': np.array([100])
}

# 3.1 Simulation of growth
economy = SolowModel(parameters, initial_state)
economy.growth(100)

# 3.2 Plotting growth from 3.1
economy.plot_growth(100)

# 3.3, 3.4  Find steady state and the number iteration it took to get there
steady_state_iterations = economy.find_steady_state()
print(f"The steady state is {economy.steady_state}")
print(
    f"It took {steady_state_iterations - 1} iterations of growth to arrive at the steady state")


# Section 4
def compare_consumption_star(society1, society2):
    c_star_1 = society1.steady_state['c_star']
    c_star_2 = society2.steady_state['c_star']

    pct_change = ((c_star_2 - c_star_1)/c_star_1) * 100

    print(
        f"Consumption changed by {pct_change}% when the saving rate changed from s_1={society1.para_dict['s']} to s_2={society2.para_dict['s']}")


# 4-1. Holding all other factors the same as 3-1, what would happen to the steady state consumption (c*) if the saving rate is 33%?
parameters = {
    'n': np.array([0.002]),
    's': np.array([0.33]),
    'alpha': np.array([1/3]),
    'delta': np.array([0.05]),
    'a': np.array([1]),
    'g': np.array([0.01]),
}

economy2 = SolowModel(parameters, initial_state)
economy2.growth(100)
steady_state_iterations_2 = economy2.find_steady_state()
compare_consumption_star(economy, economy2)
# compare_consumption_star(economy, economy2)

# 4-2. how many iterations it takes in 4-1 to converge to the steady state?
print(
    f"It took {steady_state_iterations_2 - 1} iterations of growth to arrive at the steady state")

# 4-3. Holding all other factors the same as 3-1, what would happen to the steady state consumption (c*) if the saving rate is 50%?
parameters = {
    'n': np.array([0.002]),
    's': np.array([0.50]),
    'alpha': np.array([1/3]),
    'delta': np.array([0.05]),
    'a': np.array([1]),
    'g': np.array([0.01]),
}

economy3 = SolowModel(parameters, initial_state)
economy3.growth(100)
steady_state_iterations_3 = economy3.find_steady_state()

compare_consumption_star(economy, economy3)
# 4-4. how many iterations it takes in 4-3 to converge to the steady state?

print(
    f"It took {steady_state_iterations_3 - 1} iterations of growth to arrive at the steady state")

# 4-5. Compare the steady state consumptions and the convergent speed when s=15%, s=33%, and s=50%, do you think what you find makes sense? Use economic knowledge to explain your findings.
"""
I know from ECON 307 that the golden state of an economy's capital per worker
ratio is at the level that maximizes consumption per worker, thus we'll make 
a function that helps us determine the maximal level of consumption so we can best
pick a savings rate
"""


def max_consumption(s_list):
    counter = 0
    for society in s_list:
        if counter == 0:
            max_consumption = society
            counter += 1
            continue
        if society.steady_state['c_star'] > max_consumption.steady_state['c_star']:
            max_consumption = society
    return max_consumption


comparison_group = [economy, economy2, economy3]
max_consumption = max_consumption(comparison_group)
print(
    f"The optimal saving rate was s = {max_consumption.para_dict['s']} because it maximized consumption per worker at {max_consumption.steady_state['c_star']}")
