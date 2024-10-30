# c0:
# maxmin
python3 train.py data.context_size=0 training.epochs=1 
python3 train.py --config-name=train_cnn_mlp_policy training.epochs=1 data.context_size=0
python3 train.py --config-name=train_vint_policy training.epochs=1 data.context_size=0

# standard
python3 train.py data.context_size=0 data.norm_type=standard training.epochs=1
python3 train.py --config-name=train_cnn_mlp_policy training.epochs=1 data.context_size=0 data.norm_type=standard
python3 train.py --config-name=train_vint_policy training.epochs=1 data.context_size=0 data.norm_type=standard

# goal condition = false | no target info
python3 train.py data.context_size=1 training.epochs=1 data.goal_condition=False data.target_context_enable=False data.norm_type=standard
python3 train.py data.context_size=1 training.epochs=1 data.goal_condition=True data.target_context_enable=False data.norm_type=standard
python3 train.py --config-name=train_vint_policy training.epochs=1 data.context_size=1 data.goal_condition=True data.target_context_enable=False data.norm_type=standard
python3 train.py --config-name=train_cnn_mlp_policy training.epochs=1 data.context_size=1 data.target_context_enable=False data.norm_type=standard
