import numpy as np
import matplotlib.pyplot as plt
# les  arrays  sont  batis  avec  les  dimensions  suivantes :
# pluie ,  arroseur ,  watson ,  holmes
# et  chaque  dimension :  faux ,  vrai

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print ("Pr(Pluie)={}\n".format(np.squeeze(prob_pluie)))
prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print ("Pr( Arroseur )={}\n".format(np.squeeze(prob_arroseur)))
watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print ("Pr(Watson | Pluie )={}\n".format(np.squeeze(watson)))
holmes = np.array([[1, 0], [0.1, 0.9], [0, 1], [0, 1]]).reshape(2, 2, 1, 2) # TODO
print ("Pr( Holmes | Pluie , arroseur )={}\n".format(np.squeeze(holmes)))

#a)
print ("Pr(Watson = 1)={}\n".format((watson * prob_pluie).sum(0).squeeze()[1]))

#b)
print ("Pr(Watson = 1 | Holmes = 1)={}\n".format((prob_pluie * prob_arroseur * watson * holmes)[:, :, 1, 1].sum() /
                                                 (prob_pluie * prob_arroseur * watson * holmes)[:, :, :, 1].sum()))

#c)
print ("Pr(Watson = 1 | Holmes = 1, Arroseur = 0)={}\n".format((prob_pluie * prob_arroseur * watson * holmes)[:, 0, 1, 1].sum() /
                                                 (prob_pluie * prob_arroseur * watson * holmes)[:, 0, :, 1].sum()))

#d)
print ("Pr(Watson = 1 | Arroseur = 0)={}\n".format((prob_pluie * prob_arroseur * watson * holmes)[:, 0, 1, :].sum() /
                                                 (prob_pluie * prob_arroseur * watson * holmes)[:, 0, :, :].sum()))

#e)
print ("Pr(Watson = 1 | Pluie = 1)={}\n".format(watson[1, :, 1, :].squeeze()))
