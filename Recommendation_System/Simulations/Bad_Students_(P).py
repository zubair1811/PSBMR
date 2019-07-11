from Simulations.Experiment import *
import random


pstudents =[]
nb_students=50
T = 50
n_c = 6
R_table_model=R_table([ex_type,price_presentation,cents_notation,money_type])
n_p=R_table_model.n_p
n_a_list = R_table_model.n_a
###########################################################################
success_prob=0.80
alpha=np.log(success_prob/(1-success_prob))
beta = 8
###########################################################################
nb_block = 5

for i in range(nb_students):
    ######## PARAMETERS THAT VARY FOR EACH STUDENT #################
    initKC = np.random.uniform(low=0, high=0.5, size=n_c)
    learning_rates = np.random.uniform(low=0.0001, high=0.01, size=n_c)
    ################################################################
    random_activity_list = [np.array([np.random.choice(n_a) for n_a in n_a_list]) for j in range(nb_block)]
    pstudents.append(student.Student(R_table_model, initKC, learning_rates, alpha, beta, lambdas=random_activity_list))

init_KC_class = np.zeros((nb_students,n_c))
for s in range(nb_students):
    init_KC_class[s,:]=pstudents[s].KC

for c in range(1,2):
    plt.hist(init_KC_class[:,c])
    plt.show()
gamma = 0.2
n_itr = 10
T = 10
methods = ["Predefined sequence", "PSBMR"]

# Iterations for each student and each methods
final_KC = {method: list(init_KC_class / n_itr) for method in methods}
answers = {method: np.zeros(nb_students) for method in methods}


for i, c_student in enumerate(pstudents):

    print(i)

    for method in methods[0:2]:
        for iteration in range(n_itr):
            final_KC[method][i] += gen_progresses_iter(c_student, method, T, alpha_c_hat, gamma)[0][:, -1] / n_itr
            answers[method][i] += float(gen_progresses_iter(c_student, method, T, alpha_c_hat, gamma)[-1]) / n_itr
        c_student.reset()
plt.style.use('seaborn-colorblind')
M = ["Predefined sequence", "PSBMR"]
def run(tic,tm):
    if tic == "1":
        nme = 'p-competence_level_a'
        for c in range(1, 2):

            data = []
            for method in methods[0:2]:
                data.append(np.array(final_KC[method])[:, c])

            data = np.vstack(data).T

            plt.hist(data, label=M[0:2])
            plt.title('P-Students',fontsize=15)
            plt.ylabel('Number of Students', fontsize=13)
            plt.xlabel('Level in competence ', fontsize=13)
            plt.legend()
            plt.savefig('../Result/{}{}.png'.format(nme,tm))
            plt.show()

    if tic == "2":
        nme = 'p-competence_progress_b'
        for c in range(5, 6):

            data = []
            for method in methods[0:2]:
                data.append(np.array(final_KC[method])[:, c] - init_KC_class[:, c])

            data = np.vstack(data).T

            plt.hist(data, label=M[0:2])
            plt.legend()
            plt.title('P-Students', fontsize=15)
            plt.ylabel('Number of Students', fontsize=13)
            plt.xlabel('Progress in competence ', fontsize=13)
            plt.savefig("../Result/{}{}.png".format(nme,tm))
            plt.show()

    if tic == "3":
        nme = 'p-Failures'
        data = []
        for method in methods[0:2]:
            data.append(T - answers[method])

        data = np.vstack(data).T
        plt.hist(data, label=M[0:2])
        plt.legend()
        plt.title('P-Students', fontsize=15)
        plt.ylabel('Number of Students', fontsize=13)
        plt.xlabel('Failures', fontsize=13)
        plt.savefig("../Result/{}_{}.png".format(nme,tm))
        plt.show()
    if tic == "4": ### change Graph ####
        nme = 'p-Failures'
        data1 = []
        data2 = []
        for method in methods[0:2]:
            if method == 'PSBMR':
                data1.append(T - answers[method])
            if method == 'Predefined sequence':
                data2.append(T - answers[method])

        data1 = np.vstack(data1).T
        print("data 1 value")
        print(data1)
        data2 = np.vstack(data2).T
        print("data 2 value")
        print(data2)
        plt.hist(data1,color='white', edgecolor='k', hatch="////",label=M[0])
        plt.hist(data2,color='white', edgecolor='red', hatch='....', label=M[1])
        # plt.legend()
        plt.title('P-Students', fontsize=15)
        plt.ylabel('Number of Students', fontsize=13)
        plt.xlabel('Failures', fontsize=13)
        plt.savefig("../Result/{}_{}.png".format(nme,tm))
        plt.show()

run("3",5)
# run("4",4)