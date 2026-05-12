import os

envs_path = str(os.getenv("MAMBA_ROOT_PREFIX")) + "/envs"

CACHE_LOCATION = os.path.expanduser(".cache")

ACHILLES_ENV_PATH = os.path.expanduser(f"{envs_path}/achilles_heels/bin/python")
DOMIAS_ENV_PATH = os.path.expanduser(f"{envs_path}/domias/bin/python")
SYNTHCITY_ENV_PATH = os.path.expanduser(f"{envs_path}/synthcity/bin/python")
REPROSYN_ENV_PATH = ACHILLES_ENV_PATH

SYNTHCITY_SCRIPT = "src/synthcity_generator.py"
REPROSYN_SCRIPT = "src/rsyn_generator.py"
