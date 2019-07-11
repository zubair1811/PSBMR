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
gamma = 0.18
initKC = 0.7 * np.ones(n_c)

############one student on three method result ###############
#
def gen_progresses_iter(current_student, method, T, alpha_c_hat, gamma):

    if method == "Predefined sequence":
        activity_list, c_true, correct_answers = \
            baselines.predefined_sequence(current_student, R_table_model, T)
        c_hat = 0
        return c_true, c_hat, np.sum(correct_answers)

    elif method == "PSBMR":
        reward_list, regret_list, activity_list, c_hat, c_true, _, _, correct_answers = \
            riarit.Exp3(current_student, T, R_table_model, alpha_c_hat, gamma, compute_regret=False)

        return c_true, c_hat, reward_list, regret_list, correct_answers
    else:
        return
####for comparision

def run(tic):
    if tic == '1':
        nme ='Algorithm_validation'
        T = 110
        n_itr = 100
        methods = ["PSBMR"]
        true_KC = {method: None for method in methods}
        KC_hat = {method: None for method in methods}
        activities = {method: None for method in methods}
        for i in range(n_itr):
            for method in methods:
                current_student = student.Student(R_table_model, initKC, learning_rates, alpha, beta, lambdas=None)
                KC_iter, KC_hat_iter = gen_progresses_iter(current_student, method, T, alpha_c_hat, gamma)[:2]
                if true_KC[method] is None:
                    true_KC[method] = KC_iter / n_itr
                else:
                    true_KC[method] = true_KC[method] + KC_iter / n_itr

                if KC_hat[method] is None:
                    KC_hat[method] = KC_hat_iter / n_itr
                else:
                    KC_hat[method] = KC_hat[method] + KC_hat_iter / n_itr
        M=["PSBMR"]
        # plt.style.use('seaborn-colorblind')
        for c in range(1, 2):
            plt.figure()
            i = 0
            for method in methods:
                plt.plot(true_KC[method][c],'--g', label=M[i])
                i += 1
            plt.legend()
            plt.title("propsed Task vs Knowledge Skill  ",fontsize=10)
            # plt.title("Evolution of competence IntSum for one student and several teaching methods", fontsize=13)
            plt.xlabel('Exercise Task', fontsize=13)
            plt.ylabel('knowledge Competence', fontsize=13)
            plt.xlim((0, 100))
            plt.ylim()
            plt.savefig('../Result/{}.png'.format(nme))
            plt.show()
    if tic == '2': ######Different Graph ######
        nme = 'Baseline_comparison_new'
        T = 280
        n_itr = 70
        # methods = ["Predefined sequence", "Random", "Exp3"]
        methods = ["Predefined sequence", "PSBMR"]
        # methods = ["Exp3"]
        true_KC = {method: None for method in methods}
        KC_hat = {method: None for method in methods}
        activities = {method: None for method in methods}
        for i in range(n_itr):
            for method in methods:
                current_student = student.Student(R_table_model, initKC, learning_rates, alpha, beta, lambdas=None)
                KC_iter, KC_hat_iter = gen_progresses_iter(current_student, method, T, alpha_c_hat, gamma)[:2]
                if true_KC[method] is None:
                    true_KC[method] = KC_iter / n_itr
                else:
                    true_KC[method] = true_KC[method] + KC_iter / n_itr

                if KC_hat[method] is None:
                    KC_hat[method] = KC_hat_iter / n_itr
                else:
                    KC_hat[method] = KC_hat[method] + KC_hat_iter / n_itr
        # M=["Predefined Expert sequence","Random","Exp3"]
        M = ["Predefined sequence", "PSBMR"]
        # M=["PSBMR"]
        for c in range(1, 2):
            fig, ax = plt.subplots()
            i = 0
            for method in methods:
                if method == "PSBMR":
                    ax.plot(true_KC[method][c], '--g', label="PSBMR")
                if method == "Predefined sequence":
                    ax.plot(true_KC[method][c], '-', label="Predefined sequence")
                i += 1
            ax.legend()
            # plt.title("Comprison of the two Methods on Knowledge competence vs propsed Activites",fontsize=8)
            # plt.title("Evolution of competence IntSum for one student and several teaching methods", fontsize=13)
            plt.xlabel('Exercise Task', fontsize=13)
            plt.ylabel('knowledge Competence', fontsize=13)
            plt.xlim((0, 100))
            plt.savefig('../Result/{}.png'.format(nme))
            plt.show()


run('2')