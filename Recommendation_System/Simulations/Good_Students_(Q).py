import numpy as np
from Simulations import Virtual_Student_Model as student
from PSBMR_Framework import PSBMR_Exp3 as riarit
from matplotlib import pyplot as plt
from PSBMR_Framework.M_Matrix import R_table
from Simulations import PES_Method as baselines
from Simulations.Experiment import *

qstudents = []
nb_students = 50
T = 50  # number of rounds
n_c = 6  # KnowMoney IntSum IntDec DecSum DecDec Memory
R_table_model = R_table([ex_type, price_presentation, cents_notation, money_type])
n_p = R_table_model.n_p
n_a_list = R_table_model.n_a
###########################################################################
# These parameters can vary depending on the student but not for now
success_prob = 0.8  # probability of sucess when KC=R_table
alpha = np.log(success_prob / (1 - success_prob))
beta = 7.678
###########################################################################
for i in range(nb_students):
    ######## PARAMETERS THAT VARY FOR EACH STUDENT #################
    initKC = np.random.uniform(low=0, high=0.5, size=n_c)
    # initKC = np.clip(np.random.normal(loc=0.5, scale=0.1, size=n_c), 0, 1)
    learning_rates = np.random.uniform(low=0.001, high=0.005, size=n_c)
    ################################################################

    qstudents.append(student.Student(R_table_model, initKC, learning_rates, alpha, beta, lambdas=None))
init_KC_class = np.zeros((nb_students,n_c))
for s in range(nb_students):
    init_KC_class[s,:]=qstudents[s].KC

for c in range(1,2):
    plt.hist(init_KC_class[:,c])
    plt.show()

# gamma = 0.23
# n_itr = 40
# T = 35
gamma = 0.22
n_itr = 50
T = 50
# methods = ["Predefined sequence", "Random", "Exp3"]
methods = ["Predefined sequence", "PSBMR"]

# Iterations for each student and each methods
final_KC = {method: list(init_KC_class / n_itr) for method in methods}
answers = {method: np.zeros(nb_students) for method in methods}

for i, c_student in enumerate(qstudents):

    print(i)

    for method in methods[0:2]:
        for iteration in range(n_itr):
            final_KC[method][i] += gen_progresses_iter(c_student, method, T, alpha_c_hat, gamma)[0][:, -1] / n_itr
            answers[method][i] += float(gen_progresses_iter(c_student, method, T, alpha_c_hat, gamma)[-1]) / n_itr
        c_student.reset()
plt.style.use('seaborn-colorblind')
# M = ["Predefined sequence", "Random", "Exp3"]
M = ["Predefined sequence", "PSBMR"]


def run(tic,tm):
    if tic == "1":
        nme = 'q-competence_level_a'
        for c in range(1, 2):

            data = []
            for method in methods[0:2]:
                data.append(np.array(final_KC[method])[:, c])

            data = np.vstack(data).T

            plt.hist(data, label=M[0:2])
            plt.title('Q Students',fontsize=15)
            # plt.title('Histogram of maximum level achieved in competence IntSum \n after 50 exercises for Q students',fontsize=13)
            plt.ylabel('Number of students ', fontsize=13)
            plt.xlabel('Level in competence', fontsize=13)
            plt.legend()
            plt.savefig('../Result/{}{}.png'.format(nme,tm))
            plt.show()

    if tic == "2":
        nme = 'q-competence_progress_b'
        for c in range(1, 2):

            data = []
            for method in methods[0:2]:
                data.append(np.array(final_KC[method])[:, c] - init_KC_class[:, c])

            data = np.vstack(data).T

            plt.hist(data, label=M[0:2])
            plt.legend()
            plt.title('Q Students',fontsize=15)
            # plt.title('Histogram of progress in competence IntSum \n after 50 exercises for Q student', fontsize=13)
            plt.ylabel('Number of students', fontsize=13)
            plt.xlabel('Progress in competence ', fontsize=13)
            plt.savefig('../Result/{}{}.png'.format(nme,tm))
            plt.show()

    if tic == "3":
        nme = 'q-Failures'
        data = []
        for method in methods[0:2]:
            data.append(T - answers[method])
        data = np.vstack(data).T

        plt.hist(data, label=M[0:2])
        plt.legend()
        plt.title('Q Students',fontsize=15)
        plt.xlabel('Failures', fontsize=13)
        plt.ylabel('Number of students', fontsize=13)
        plt.savefig('../Result/{}_{}.png'.format(nme,tm))
        plt.show()

run("3",4)





