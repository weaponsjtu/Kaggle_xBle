all:
	echo "fold, featu, [model, opt], [ensemble,rank, stack, subm], show"

fold:
	python gen_fold.py

featu:
	python gen_feat.py | tee -a log/feat.log

model:
	python model_library.py "train" | tee -a log/model.log

opt:
	python model_library.py "hyperopt" 2>&1 | tee -a log/hyperopt.log

stack:
	python gen_stacking.py | tee -a log/stacking.log

ensemble:
	python gen_ensemble.py "ensemble" | tee -a log/ensemble.log

subm:
	python gen_ensemble.py "submission" | tee -a log/subm.log

rank:
	python gen_ensemble.py "rankavg" | tee -a log/rank.log

show:
	python utils.py | tee -a log/utils.log
