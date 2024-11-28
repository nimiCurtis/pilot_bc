# c0:
# maxmin
# python3 train.py data.context_size=0 training.epochs=1 
# python3 train.py --config-name=train_cnn_mlp_policy training.epochs=1 data.context_size=0
# python3 train.py --config-name=train_vint_policy training.epochs=1 data.context_size=0

# # standard
# python3 train.py data.context_size=0 data.norm_type=standard training.epochs=1
# python3 train.py --config-name=train_cnn_mlp_policy training.epochs=1 data.context_size=0 data.norm_type=standard
# python3 train.py --config-name=train_vint_policy training.epochs=1 data.context_size=0 data.norm_type=standard

# # goal condition = false | no target info
# python3 train.py data.context_size=1 training.epochs=1 data.goal_condition=False data.target_context_enable=False data.norm_type=standard
# python3 train.py data.context_size=1 training.epochs=1 data.goal_condition=True data.target_context_enable=False data.norm_type=standard
# python3 train.py --config-name=train_vint_policy training.epochs=1 data.context_size=1 data.goal_condition=True data.target_context_enable=False data.norm_type=standard
# python3 train.py --config-name=train_cnn_mlp_policy training.epochs=1 data.context_size=1 data.target_context_enable=False data.norm_type=standard



# python3 train.py data.pred_horizon=16
# python3 train.py data.pred_horizon=32
# python3 train.py data.pred_horizon=64

# python3 train.py data.pred_horizon=32 training.goal_mask_prob=0.75 training.modal_dropout_prob=0.5

# python3 train.py data.pred_horizon=64



#finetune
python3 train.py --config-name=train_pilot_finetune training.fine_tune.model=pidiff_bsz64_c1_ac1_gcTrue_gcp0.7_ah16_ph16_tceTrue_ntmaxmin_2024-11-27_18-11-30 training.goal_mask_prob=0.7 training.modal_dropout_prob=0.7 data.pred_horizon=16 
python3 train.py --config-name=train_pilot_finetune training.fine_tune.model=pidiff_bsz64_c1_ac1_gcTrue_gcp0.7_ah16_ph64_tceTrue_ntmaxmin_2024-11-27_18-48-19 training.goal_mask_prob=0.7 training.modal_dropout_prob=0.7 data.pred_horizon=64 
