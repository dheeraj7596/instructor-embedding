export PYTHONPATH="${PYTHONPATH}:/home/dheeraj/instructor-embedding/evaluation/MTEB/mteb"
sh examples/scripts/classification.sh
sh examples/scripts/clustering.sh
sh examples/scripts/pairwiseclassification.sh
sh examples/scripts/reranking.sh
