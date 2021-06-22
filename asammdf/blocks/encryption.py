import importlib
import importlib.util
from pathlib import Path
from traceback import format_exc

encrypt = decrypt = None


def load_user_encryption_module(path):
    """
    loads a user defined module that must contain an `encrypt` and
    a `decrypt` function
    """
    global encrypt
    global decrypt
    path = Path(path)
    try:
        spec = importlib.util.spec_from_file_location(path.with_suffix("").name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        encrypt = module.encrypt
        decrypt = module.decrypt
    except:
        print(format_exc())
