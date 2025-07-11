# learner.py
import pickle
from data_store import load_events
def learn(mdl='model.pkl'):
    hist=[v.get('intensity') for v in load_events()]; pickle.dump(hist,open(mdl,'wb'))
if __name__=='__main__': learn()
