eval "$(conda shell.bash hook)"
conda activate gid
set -a
source "$(dirname "$0")/.env"
set +a
python -m db.init_db
python -m db.load_fixtures
uvicorn main:app --reload

