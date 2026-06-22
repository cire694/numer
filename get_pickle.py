import cloudpickle


def get_pickle(predict_fn, out_file):
    """Serialize a prediction function to disk using cloudpickle.

    Args:
        predict_fn: Callable prediction function or pipeline object.
        out_file: File path to write the serialized object.
    """
    p = cloudpickle.dumps(predict_fn)
    with open(out_file, "wb") as f: 
        f.write(p)
    
    