import numpy as np
from Simulations import Virtual_Student_Model as student
from PSBMR_Framework import PSBMR_Exp3 as riarit
from matplotlib import pyplot as plt
from PSBMR_Framework.M_Matrix import R_table
from Simulations import PES_Method as baselines

ex_type=np.array([[0.7,0.4,0,0,0,0.5],
    [0.7,0.6,0.3,0,0,0.5],
    [0.7,0.7,0.6,0,0,0.5],
    [1,0.7,0.6,0.5,0.3,0.7],
    [1,0.9,0.7,0.7,0.5,0.7],
    [1,1,1,1,1,1]])

price_presentation=np.array([[0.8,1,1,1,1,0.2],
    [1,1,1,1,1,0.6],
    [0.9,1,1,1,1,1]])

cents_notation=np.array([[0.8,1,1,1,1,1],
    [0.9,1,1,1,1,1]])

money_type=np.array([[1,1,1,0.9,0.9,1],
    [0.1,1,1,1,1,1]])

# T = 50
n_c = 6
alpha_c_hat = 0.3
R_table_model = R_table([ex_type, price_presentation, cents_notation, money_type])
n_p = R_table_model.n_p
learning_rates = np.random.uniform(low=0.1, high=0.2, size=n_c)
success_prob = 0.8  # probability of sucess when KC=R_table
alpha = np.log(success_prob / (1 - success_prob))
beta = 8
beta_w = 1  ## coefficient of the previous value w_a
eta_w = 0.2  ## learning rate for w_a
gamma = 0.2
initKC = 0.7 * np.ones(n_c)

############one student on three method result ###############
#
def gen_progresses_iter(current_student, method, T, alpha_c_hat, gamma):

    if method == "Predefined sequence":
        activity_list, c_true, correct_answers = \
            baselines.predefined_sequence(current_student, R_table_model, T)
        c_hat = 0
        return c_true, c_hat, np.sum(correct_answers)

    elif method == "Random":
        reward_list, regret_list, activity_list, c_hat, c_true, _, _, correct_answers = \
            riarit.Exp3(current_student, T, R_table_model, alpha_c_hat, 1, compute_regret=False)

        return c_true, c_hat, reward_list, regret_list

    elif method == "PSBMR":
        reward_list, regret_list, activity_list, c_hat, c_true, _, _, correct_answers = \
            riarit.Exp3(current_student, T, R_table_model, alpha_c_hat, gamma, compute_regret=False)

        return c_true, c_hat, reward_list, regret_list, correct_answers
    else:
        return
