from dataclasses import dataclass, field

@dataclass
class Setting:
    data_file: str
    test_file: str
    labels_mapping: str
    extra_label_file: str
    model_name: str 
    model_type: str # default, mixed
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    criterion: str
    optimizer: str
    lr_scheduler: str
    warmup_ratio: float

    layer_lr_decay: float = 1
    madgrad_weight_decay: float = 0
    warmup_step: int = None
    max_norm: int = 1
    seed: int = 42
    small_data: bool = False
    use_amp: bool = False
    logging: bool = False
    test_ratio: float = 0.3
    device_pref: str = 'cuda'
    project: str = None
    entity: str = None
    name: str = None
    tags: list = field(default_factory=lambda: [])
    is_testing: bool = False
    no_empty_label: bool = False
    single_col_data: str = None
    freezing_layer_num: int = -1 # only for mixed model,  -1, 0, 1, 2, 3, 4, 5, 6
                                # -1 = no freezing, 0 = freeze only word emb
    use_intermediate_cache: bool = False
    freeze_emb: bool = False
    gate_th: float = 0.5
    gate_44: bool = False
    gate_44_th: float = 0.5
    big_mixed_use_each: bool =  False

    use_swa: bool = False
    swa_lr: float = 1e-4
    swa_annel_steps_ratio: float = 0.1
    swa_annel_steps: int = None
    swa_start_epoch: int = 7
    
    tokenize_max_seq: int = 500

    line_stuff: bool = False
    line_file: str = None

    clean_dirty: bool = False
    special_tms: int = 64

    mixed_col_keys_to_use: list = field(
        default_factory=lambda: ['사업명', '사업_부처명', '내역사업명', '과제명', '요약문_한글키워드', '요약문_영문키워드']
    )

    use_weight_in_loss: bool = False

    use_extra_label: bool= False
    el_single_output: bool = False

    mixed_two_models: bool= False
    mixed_second_model_index: list = field(
        default_factory=lambda: [1]
    )

    use_llrd: bool= False
    llrd_decay: float= 0.95

    multi_sample_dropout_n: int=1
    output_dropout_p: float=0.1