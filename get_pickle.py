import cloudpickle

def get_pickle(predict_fn, out_file):
    p = cloudpickle.dumps(predict_fn)
    with open(out_file, "wb") as f: 
        f.write(p)
    
    