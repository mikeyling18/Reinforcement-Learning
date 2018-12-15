from q_learner import Q_Learner
from foe_q import Foe_Learner
from friend_q import Friend_Learner
from correlated_q import Correlated_Learner
import datetime as dt

gamma = 0.9
alpha = 0.2
total_time_steps = 1000000

# add times to files to prevent accidental overwriting
time = dt.datetime.now().strftime('%I-%M-%S-%p')

q_learner = Q_Learner()
q_diff = q_learner.learn(total_time_steps, alpha, gamma)
file = open('q_learner - ' + time + '.csv', 'w')
for data_point in q_diff:
    file.write('{} \n'.format(data_point))

foe_learner = Foe_Learner()
foe_q_diff = foe_learner.learn(total_time_steps, alpha, gamma)
file = open('foe_learner - ' + time + '.csv', 'w')
for qdiff in foe_q_diff:
    file.write('{}\n'.format(qdiff))

friend_learner = Friend_Learner()
friend_q_diff = friend_learner.learn(total_time_steps, alpha, gamma)
file = open('friend_learner - ' + time + '.csv', 'w')
for qdiff in friend_q_diff:
    file.write('{}\n'.format(qdiff))

# Doesn't work :(
correlated_learner = Correlated_Learner()
corr_q_diff = correlated_learner.learn(total_time_steps, alpha, gamma)
file = open('corr_learner - ' + time + '.csv', 'w')
for qdiff in corr_q_diff:
    file.write('{}\n'.format(qdiff))

