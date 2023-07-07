# Config Options
This file defines how to change values for each of the config options.

## Dataset
To use a different dataset check that the config for the dataset you want to use exists in `configs/dataset`.
You can then change the dataset by adding the following to the run command:
```
... dataset=<name_of_dataset_config>
```

## Model
```
... model=<name_of_model_config>
```

## Optimizer (Default: Adam)
```
... optimizer=<name_of_optim_config>
```

## LR Scheduler (Default: None)
```
... lr_scheduler=<name_of_lr_scheduler_config>
```

## Adversarial Training (Default: None)
```
... adv_training=<name_of_adv_training_config>
```

## Augmentations (Default: resize)
```
... "augmentations@train_augmentations=[resize, random_rotation]"
```
This works either with `train_augmentations` or `eval_augmentations`.

> **_Note:_** You need the quotation marks here. Otherwise, you will get a parsing error.

## Normalization (default: imagenet)
```
... normalization=<name_of_normalization_config>
```

## Other Possible Values
You can change any of these options using:
```
<value_name>=<value>
```

| **Value Name**                         | **Default Value**                                      | **Possible Values** | **Description**                                                                                  |
|----------------------------------------|--------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------|
| seed                                   | true                                                   | bool                | The number of epochs after which the training is stopped if the validation loss did not improve. |
| wandb.entity                           | null                                                   | str                 | The entity to log the WandB run to. Using the default entity by default.                         |
| wandb.experiment_name                  | 'Template Experiment'                                  | str                 | The name of the experiment.                                                                      |
| wandb.run_name                         | ${model.name} ${dataset.name} ${wandb.experiment_name} | str                 | The name of the run.                                                                             |
| wandb.offline                          | false                                                  | bool                | Whether WandB will sync with the runs with the server.                                           |
| wandb.project                          | 'Template Project'                                     | str                 | The name of the WandB project under which the run is going to be logged.                         |
| datamodule.dataloader_args.num_workers | 8                                                      | int                 | The number of workers for the dataloader.                                                        |
| datamodule.dataloader_args.shuffle     | True                                                   | bool                | Whether to shuffle the training data.                                                            |
| datamodule.dataloader_args.pin_memory  | True                                                   | bool                | Whether to pin the memory when using cuda.                                                       |
| rtpt.initials                          | 'DH'                                                   | str                 | The initials for rtpt.                                                                           |


## Training
Using hydra you can assign values using `.` and specify configs using `/`.

### Early Stopping (Default: None)
By default early stopping is not used. To use early stopping you can speficy using the
early stopping config by using: 
```
... training/early_stopping=early_stopping
```

Possible Options:

| **Value Name**            | **Default Value** | **Description**                                                                                       |
|---------------------------|-------------------|-------------------------------------------------------------------------------------------------------|
| early_stopping_patience   |                   | The number of epochs after which the training is stopped if the validation loss did not improve.      |
| early_stopping_min_delta  |                   | The amount by which the validation loss has to improve such that it is registered as an improvement.  |

> **_Note:_** You can change these options by adding them **after** the command above:
> 
> ```
> ... training/early_stopping=early_stopping training.early_stopping.early_stopping_patience=10
> ```

### Other Possible Values
You can change any of these options using:
```
training.<value_name>=<value>
```

| **Value Name** | **Default Value** | **Possible Values** | **Description**                                                                                       |
|----------------|-------------------|---------------------|-------------------------------------------------------------------------------------------------------|
| save_model     | true              | true, false         | The number of epochs after which the training is stopped if the validation loss did not improve.      |
| epochs         | 100               | int                 | The amount by which the validation loss has to improve such that it is registered as an improvement.  |
| batch_size     | 128               | int                 |                                                                                                       |


