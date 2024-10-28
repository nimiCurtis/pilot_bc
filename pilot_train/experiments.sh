# python3 train.py data.context_size=0
# python3 train.py data.context_size=1 
# python3 train.py data.context_size=2 

# standard normtype
python3 train.py data.context_size=0 data.distance.min_dist_cat=12  data.distance.max_dist_cat=20 data.action.min_dist_cat=12  data.action.max_dist_cat=20 data.norm_type=standard 
python3 train.py data.context_size=1 data.distance.min_dist_cat=12  data.distance.max_dist_cat=20 data.action.min_dist_cat=12  data.action.max_dist_cat=20 data.norm_type=standard
python3 train.py data.context_size=2 data.distance.min_dist_cat=12  data.distance.max_dist_cat=20 data.action.min_dist_cat=12  data.action.max_dist_cat=20 data.norm_type=standard
python3 train.py data.context_size=0 data.distance.min_dist_cat=24  data.distance.max_dist_cat=40 data.action.min_dist_cat=24  data.action.max_dist_cat=40 data.norm_type=standard
python3 train.py data.context_size=1 data.distance.min_dist_cat=24  data.distance.max_dist_cat=40 data.action.min_dist_cat=24  data.action.max_dist_cat=40 data.norm_type=standard
python3 train.py data.context_size=2 data.distance.min_dist_cat=24  data.distance.max_dist_cat=40 data.action.min_dist_cat=24  data.action.max_dist_cat=40 data.norm_type=standard

# maxmin normtype
python3 train.py data.context_size=0 data.distance.min_dist_cat=12  data.distance.max_dist_cat=20 data.action.min_dist_cat=12  data.action.max_dist_cat=20 data.norm_type=maxmin
python3 train.py data.context_size=1 data.distance.min_dist_cat=12  data.distance.max_dist_cat=20 data.action.min_dist_cat=12  data.action.max_dist_cat=20 data.norm_type=maxmin 
python3 train.py data.context_size=2 data.distance.min_dist_cat=12  data.distance.max_dist_cat=20 data.action.min_dist_cat=12  data.action.max_dist_cat=20 data.norm_type=maxmin
python3 train.py data.context_size=0 data.distance.min_dist_cat=24  data.distance.max_dist_cat=40 data.action.min_dist_cat=24  data.action.max_dist_cat=40 data.norm_type=maxmin
python3 train.py data.context_size=1 data.distance.min_dist_cat=24  data.distance.max_dist_cat=40 data.action.min_dist_cat=24  data.action.max_dist_cat=40 data.norm_type=maxmin
python3 train.py data.context_size=2 data.distance.min_dist_cat=24  data.distance.max_dist_cat=40 data.action.min_dist_cat=24  data.action.max_dist_cat=40 data.norm_type=maxmin