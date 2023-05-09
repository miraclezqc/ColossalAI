from colossalai.amp import AMP_TYPE

# hyperparameters config
# BATCH_SIZE is as per GPU, global batch size = BATCH_SIZE x data parallel size
SEQ_LEN = 1024
VOCAB_SIZE = 32000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
TRAIN_STEPS = 10    #NUM_EPOCHS
WARMUP_STEPS = 2    #WARMUP_EPOCHS

# Features 1 (Auto Mixed Precision)
# use Torch AMP
fp16 = dict(
    mode=AMP_TYPE.TORCH,
    # default value of grad scaler
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True)

# use naive AMP
# fp16=dict(
#     mode = AMP_TYPE.NAIVE
# )

# use NVIDIA Apex AMP
# fp16=dict(
#     mode = AMP_TYPE.APEX
# )

# Features 2 (Gradient Accumulation)
# gradient_accumulation = 4 # don't use it together with GeminiDPP

# Features 3 (Gradient Clipping)
clip_grad_norm = 1.0

# Features 4 (Gradient Handler)
gradient_handler = [dict(type='MyGradientHandler')]

# parallel setting
TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '2d'

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']