import pickle

# saving optimizer config
def save_optimizer_config(filename, optimizer):
      with open(filename+'.data', 'wb') as f:
          pickle.dump(optimizer.get_config(), f)
      f.close()
      del f
def load_optimizer_config(path, optimizer):
      with open(path+'.data', 'rb') as f:
              config = pickle.load(f)
              optimizer.from_config(config)
      f.close()
      del f      

# saving the number of iterations
def save_global_step(filename, global_step):
        with open(filename+'.data', 'wb') as f:
            pickle.dump(global_step, f)
        f.close()
        del f

def load_global_step(path):
        with open(path+'.data', 'rb') as f:
          global_step = pickle.load(f)
        f.close()
        del f
        return global_step     