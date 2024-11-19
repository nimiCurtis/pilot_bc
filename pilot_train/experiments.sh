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



#finetune
python3 train.py --config-name=train_pilot_finetune training.fine_tune.model=pidiff_bsz128_c2_ac1_gcTrue_gcp0.75_ah16_ph16_tceTrue_ntmaxmin_2024-11-18_20-30-21 data.pred_horizon=16
python3 train.py --config-name=train_pilot_finetune training.fine_tune.model=pidiff_bsz128_c2_ac1_gcTrue_gcp0.75_ah16_ph32_tceTrue_ntmaxmin_2024-11-18_20-54-00 data.pred_horizon=32
python3 train.py --config-name=train_pilot_finetune training.fine_tune.model=pidiff_bsz128_c2_ac1_gcTrue_gcp0.75_ah16_ph64_tceTrue_ntmaxmin_2024-11-18_21-17-50 data.pred_horizon=64