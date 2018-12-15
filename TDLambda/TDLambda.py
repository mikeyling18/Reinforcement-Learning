import numpy as np
import matplotlib.pyplot as plt
import random as rd


def create_sequence():
    # Each sequence is a binary array detailing which state the learner is currently in
    # [A, B, C, D, E, F, G]
    sequence_list = list()
    sequence = [0, 0, 0, 1, 0, 0, 0]
    sequence_list.append(sequence)
    while(sequence_list[-1][-1] != 1) and (sequence_list[-1][0] != 1):
        index = np.argmax(sequence_list[-1])
        new_index = index + rd.choice([-1,1])
        new_sequence = np.copy(sequence_list[-1])
        new_sequence[index] = 0
        new_sequence[new_index] = 1
        sequence_list.append(new_sequence)
    return sequence_list


def create_training_sets(number_of_training_sets, number_of_sequences):
    training_sets = list()
    training_sequence = list()
    for i in range(0,number_of_training_sets):
        for j in range(0,number_of_sequences):
            training_sequence.append(create_sequence())

        training_sets.append(training_sequence)
        training_sequence = list()

    return training_sets


# This function is based off the equation described at the top of page 16 of the paper
#
def sum_of_lambda(L, P, t):
    _sum = np.zeros(7)

    # This is the last portion of equation 3 on page 15
    # the sum of P_k from k = 1 to t
    for k in range(1, t+1):
        # the gradient of Pt with respect to w is simply X_i, which is the state vector
        # in this case the state vector is P, which is assigned to S when this is called
        _sum = _sum + (L ** (t-k)) * P[k]
    return _sum


"""The weight vector was not updated after each sequence as indicated by equation: W <- W + sum of delta Wt from t = 1 to m.
instead, the delta W's were accumulated over sequences and only used to update the weight vector after the complete
presentation of a training set - Sutton 1988, page 20 """
def TDLambda_Experiment_1(sequence, weights, a, L):
    w_t = weights
    # converged = False
    delta_new = np.zeros(7)
    while(1):
        delta = np.zeros(7)
        for S in sequence:
            for T in range(len(S)-1):
                P_Tplus1 = np.dot(S[T+1], w_t)
                P_T = np.dot(S[T], w_t)
                # The following equation is based off equation 3 in the paper
                delta = delta + (a * (P_Tplus1 - P_T) * sum_of_lambda(L, S, T))

        # wait to update the weights until after the complete presentation of a training set
        w_t = w_t + delta
        if np.sum(np.abs(delta_new - delta)) < 0.001:
            break
        else:
            delta_new = delta
    return w_t


"""...we assume that W is updated only once for each complete observation outcome sequence and thus does not change
during a sequence. For each observation, an increment to W...is determined, and, after a complete sequence has been
processed, W is changed by all the sequence's increments - Sutton 1988, page 13"""
def TDLambda_Experiment_2(sequence, weights, a, L):
    w_t = weights
    for S in sequence:
        for T in range(0,len(S)-1):
            P_Tplus1 = np.dot(S[T+1], w_t) # Pt is the dot product of w and state vectors (bottom of page 13)
            P_T = np.dot(S[T], w_t)
            # The following equation is based off equation 3 in the paper
        w_t = w_t + a * (P_Tplus1 - P_T) * sum_of_lambda(L, S, T)
    return w_t




# ------------------------------------------------------------------------------------------------------
# Driver Code for Figure 3
# ------------------------------------------------------------------------------------------------------


n_training_sets = 100
n_sequences = 10
# Lambda values given in Figure 3 of Sutton 1988...
lambdas = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

# value given in paper - page 20
alpha = 0.01

avg_rms_error = []

# True probabilities given on page 20 of Sutton 1988...
true_probabilities = [1./6., 1./3., 1./2., 2./3., 5./6.]

# Initialize weights accordingly...
init_weights = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

test = create_sequence()
training_sets = create_training_sets(number_of_sequences=n_sequences, number_of_training_sets=n_training_sets)
training_sets = np.array(training_sets)

for _lambda in lambdas:
    rms_error = []
    for sequence in training_sets:
        predicted_weights = TDLambda_Experiment_1(sequence=sequence, weights=init_weights, a=alpha, L=_lambda)

        rms_error.append(np.sqrt(np.mean((predicted_weights[1:-1]-true_probabilities)**2)))

    avg_rms_error.append((np.mean(rms_error)))

plt.plot(lambdas, avg_rms_error, marker = 'o')
plt.margins(0.05, 0.1)
plt.xlabel("Lambda")
plt.ylabel("RMS")
plt.title("Figure 3")
plt.show()

# ------------------------------------------------------------------------------------------------------
# Driver Code for Figure 4
# ------------------------------------------------------------------------------------------------------

alphas = np.arange(0.0, 0.65, 0.05)
lambdas = np.array([0.0, 0.3, 0.8, 1.0])
init_weights = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

avg_rms_error_per_lambda = []
for _lambda in lambdas:
    avg_rms_error = []
    for alpha in alphas:
        rms_error = []
        for sequence in training_sets:

            predicted_weights = TDLambda_Experiment_2(sequence=sequence, weights=init_weights, a=alpha, L=_lambda)
            rms_error.append(np.sqrt(np.mean((predicted_weights[1:-1] - true_probabilities) ** 2)))

        avg_rms_error.append(np.mean(rms_error))

    avg_rms_error_per_lambda.append(avg_rms_error)

plt.plot(alphas, avg_rms_error_per_lambda[0], marker = 'o')
plt.plot(alphas, avg_rms_error_per_lambda[1], marker = 'o')
plt.plot(alphas, avg_rms_error_per_lambda[2], marker = 'o')
plt.plot(alphas, avg_rms_error_per_lambda[3], marker = 'o')
plt.margins(0.05, 0.1)
plt.xlabel("Alpha")
plt.ylabel("ERROR")
plt.ylim([0.0, 0.7])
plt.title("Figure 4")
plt.legend(['Lambda = 0', 'Lambda = 0.3', 'Lambda = 0.8', 'Lambda = 1'])
plt.show()

# ------------------------------------------------------------------------------------------------------
# Driver Code for Figure 5
# ------------------------------------------------------------------------------------------------------


lambdas = np.arange(0.0, 1.1, 0.1)
true_probabilities = [1./6., 1./3., 1./2., 2./3., 5./6.]
init_weights = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1])


# This 'best-alpha' was taken from Figure 4
# The 'best-alpha' is the alpha value where the lowest 'ERROR' occurred
avg_rms_error = []
alpha = 0.25
for _lambda in lambdas:
    rms_error = []
    for sequence in training_sets:
        predicted_weights = TDLambda_Experiment_2(sequence=sequence, weights=init_weights, a=alpha, L=_lambda)
        rms_error.append(np.sqrt(np.mean((predicted_weights[1:-1]-true_probabilities)**2)))

    avg_rms_error.append((np.mean(rms_error)))

plt.plot(lambdas, avg_rms_error, marker = 'o')
plt.margins(0.05, 0.1)
plt.xlabel("Lambda")
plt.ylabel("Error Using Best Alpha")
plt.ylim([0.0, 0.3])
plt.title("Figure 5")
plt.show()
