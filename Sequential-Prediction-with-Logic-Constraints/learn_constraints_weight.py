from pracmln import MLN, Database
from pracmln import query, learn
from pracmln.mlnlearn import EVIDENCE_PREDS
import time

data_dir = "../logic_constraints"
PRACMLN_HOME = "."
def inference_surgeme():
    mln = MLN(mlnfile=data_dir+'/gym_pickplace_trained.mln',
              grammar='StandardGrammar')
    
    db = Database(mln, dbfile=data_dir+'/gym_pickplace.db')
    for method in ['GibbsSampler']:
        # for multicore in (False, True):
        print('=== INFERENCE TEST:', method, '===')
        query(queries='Has',
              method=method,
              mln=mln,
              db=db,
              verbose=True,
              multicore=True).run()
    
def learning_surgeme(mlnfile, dbfile):
    mln = MLN(mlnfile=mlnfile, grammar='StandardGrammar') # .mln file is the domain knowledge (FOL - verify)
    mln.write()
    db = Database(mln, dbfile=dbfile)
    savefile = data_dir+'/save.mln' # not working now
    # for method in ('BPLL', 'BPLL_CG', 'CLL'):
    for method in ['BPLL_CG']:
        # for multicore in (True, False):
        print('=== LEARNING TEST:', method, '===')
        learn(method=method,
              mln=mln,
              db=db,
              verbose=True,
              multicore=True,
              output_filename=savefile,
                  save=True).run()

def runall(ROBOT="Gym"):
    start = time.time()
    if ROBOT == "Gym":
        dbfile=data_dir+'/gym_kinematics_train.db'
        mlnfile=data_dir+'/gym.mln'
    else:
        dbfile=data_dir+'/taurus_kinematics_train.db'
        mlnfile=data_dir+'/taurus.mln'

    learning_surgeme(mlnfile=mlnfile, dbfile=dbfile)

    print()
    print('All test finished after', time.time() - start, 'secs')

if __name__ == '__main__':
    ROBOT = "Gym" # Gym/Taurus
    runall(ROBOT=ROBOT)
    


